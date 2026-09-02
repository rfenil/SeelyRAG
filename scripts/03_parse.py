#!/usr/bin/env python
"""Stage 2 -- parse the acquired corpus into per-page records.

build-plan.md sections 4.2 to 4.5.

Reads ``data/00_raw/manifest.jsonl``, parses every unique document, and writes
``data/01_interim/pages.jsonl`` plus rendered page images.

Iterates **documents, not articles**. A manual attached to five articles is
parsed once; parsing per article would repeat the work five times, including the
table-detection pass and the page renders.

Resumes by default: documents already present in ``pages.jsonl`` are skipped.

Scanned and diagram-heavy pages are recorded with ``needs_vision`` set and
whatever text they carry. Nothing is dropped -- the work is queued for
``parse/vision.py``, which is not implemented yet.

Exit codes:
    0 -- parse completed.
    1 -- nothing to parse, or the manifest is missing.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import os
import sys

from seeley_rag.acquire.manifest import load_manifest, unique_documents
from seeley_rag.exceptions import ManifestError, ParseError
from seeley_rag.logging_conf import configure_logging, get_logger
from seeley_rag.parse.base import Page, PagesWriter, parsed_doc_ids, read_pages
from seeley_rag.parse.html import ingest_articles
from seeley_rag.parse.pdf import parse_document_to_dicts
from seeley_rag.paths import PAGES_PATH, ensure_dirs

log = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Parse acquired PDFs and diagnostic articles into pages.jsonl.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--limit", type=int, default=None, help="Parse at most N documents.")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Worker processes. Table detection and rendering are CPU-bound.",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Skip page rendering. Much faster, but 'show me the diagram' will "
        "not work and the vision tiers will have no input.",
    )
    parser.add_argument(
        "--no-articles",
        action="store_true",
        help="Skip the diagnostic-article stream; parse PDFs only.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-parse everything, replacing pages.jsonl. Without this, "
        "documents already parsed are skipped.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be parsed, then stop.",
    )
    parser.add_argument(
        "--refresh-metadata",
        action="store_true",
        help="Recompute product family, doc type and model codes in pages.jsonl "
        "from config/models.yaml. Seconds, not hours: no PDF is reopened. Use "
        "this after editing the lexicon.",
    )
    parser.add_argument(
        "--log-format", choices=("json", "console"), default="console", help="Log output format."
    )
    return parser.parse_args()


def refresh_metadata() -> int:
    """Recompute product family, doc type and model codes in place.

    ``config/models.yaml`` is hand-maintained and grows as the crawl reveals
    codes -- the build plan explicitly expects that. Every edit invalidates the
    metadata already written into ``pages.jsonl``, but *only* the metadata: the
    text, tables, labels and page images are unaffected.

    Re-running the full parse to pick up a lexicon change would redo table
    detection over 12,526 pages, which costs hours for a change that costs
    seconds. This rewrites the three derived fields and nothing else.

    Returns:
        A process exit code.
    """
    from seeley_rag.parse.base import resolve_doc_type, resolve_model_series, resolve_product_family

    try:
        pages = list(read_pages())
    except ParseError as exc:
        print(f"Nothing to refresh: {exc}")
        return 1

    changes: collections.Counter = collections.Counter()
    for page in pages:
        family = resolve_product_family(page.category, page.folder, page.title)
        if family != page.product_family:
            changes[f"{page.product_family} -> {family}"] += 1
        page.product_family = family
        page.doc_type = resolve_doc_type(page.folder, page.title)
        page.model_series = resolve_model_series(page.title)

    with PagesWriter(overwrite=True) as writer:
        writer.write_all(pages)

    total = sum(changes.values())
    print(f"Refreshed metadata on {len(pages):,} page(s) from config/models.yaml.")
    if not total:
        print("  No family labels changed.")
        return 0

    print(f"  {total:,} family label(s) changed ({total/len(pages):.1%}):")
    for transition, n in changes.most_common(12):
        print(f"    {transition:<26} {n:>6,}")
    print()
    print("  Text, tables, page labels and images were not touched.")
    return 0


def print_plan(documents: list, articles: int, skipped: int) -> None:
    """Print what the run intends to do.

    Args:
        documents: Documents queued for parsing.
        articles: Content articles queued for ingestion.
        skipped: Documents skipped because they were already parsed.
    """
    # ~160 KB per page, measured across this corpus (1.99 GB / 12,526 pages).
    pages_estimate = sum(max(1, d.size_bytes // 160_000) for d in documents)
    print("Parse plan")
    print("----------")
    print(f"  documents to parse   {len(documents)}")
    print(f"  already parsed       {skipped}")
    print(f"  content articles     {articles}")
    print(f"  estimated pages      ~{pages_estimate:,}")
    print()


def summarise_pages(counts: collections.Counter, families: collections.Counter) -> None:
    """Print a summary of what was produced.

    Args:
        counts: Tier and stream counters.
        families: Product-family counters.
    """
    total = counts["pages"]
    print()
    print("Parsed corpus")
    print("-------------")
    print(f"  pages written          {total:,}")
    print(f"    from PDFs            {counts['pdf']:,}")
    print(f"    from articles        {counts['diagnostic_article']:,}")
    print(f"  tables detected        {counts['tables']:,}")
    print(f"  page images rendered   {counts['images']:,}")
    print()
    if total:
        print("  by tier:")
        for tier in ("plain_text", "diagram_heavy", "scanned"):
            n = counts[tier]
            print(f"    {tier:<16} {n:>7,}  {n/total:>6.1%}")
        vision = counts["diagram_heavy"] + counts["scanned"]
        print(f"  awaiting vision        {vision:,} ({vision/total:.1%})")
    print()
    pdf_pages = counts["pdf"]
    print("  printed page labels (PDF pages only):")
    for source, caption in (
        ("embedded", "from the PDF's label tree"),
        ("text", "read from the page footer"),
        ("index", "GUESSED as index+1"),
    ):
        n = counts[f"label_{source}"]
        share = f"{n/pdf_pages:>6.1%}" if pdf_pages else "     -"
        print(f"    {caption:<26} {n:>7,}  {share}")
    trusted = counts["label_embedded"] + counts["label_text"]
    if pdf_pages:
        print(f"    -> citable labels          {trusted:,} ({trusted/pdf_pages:.1%})")
    print()
    print("  product families:")
    for family, n in families.most_common():
        print(f"    {family:<12} {n:>7,}")


def parse_documents(
    queued: list,
    args: argparse.Namespace,
    writer: PagesWriter,
    counts: collections.Counter,
    families: collections.Counter,
) -> None:
    """Parse the queued PDFs, fanning out across processes.

    Table detection and page rendering are both CPU-bound, so this is the one
    place in the project where parallelism pays. (The *crawl* is deliberately
    single-threaded -- that is politeness, not performance.)

    Args:
        queued: Documents to parse.
        args: Parsed command-line arguments.
        writer: Open pages writer.
        counts: Counter to update.
        families: Product-family counter to update.
    """
    if not queued:
        return

    payloads = [{"document": vars(d), "render_images": not args.no_images} for d in queued]
    print(f"Parsing {len(payloads)} document(s) across {args.workers} worker(s)...")

    if args.workers <= 1:
        stream = map(parse_document_to_dicts, payloads)
        for done, rows in enumerate(stream, start=1):
            _write_rows(rows, writer, counts, families)
            _report(done, len(payloads), counts)
        return

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for done, rows in enumerate(pool.map(parse_document_to_dicts, payloads), start=1):
            _write_rows(rows, writer, counts, families)
            _report(done, len(payloads), counts)


def _report(done: int, total: int, counts: collections.Counter) -> None:
    """Print a periodic progress line.

    Args:
        done: Documents completed.
        total: Documents queued.
        counts: Running counters.
    """
    if done % 10 == 0 or done == total:
        print(f"  ... {done}/{total} docs, {counts['pages']:,} pages", flush=True)


def main() -> int:
    """Run the parse.

    Returns:
        A process exit code.
    """
    args = parse_args()
    configure_logging(fmt=args.log_format)
    ensure_dirs()

    if args.refresh_metadata:
        return refresh_metadata()

    try:
        documents = unique_documents()
        articles = load_manifest()
    except ManifestError as exc:
        print(f"Cannot parse: {exc}")
        return 1

    already = set() if args.overwrite else parsed_doc_ids()
    pdf_docs = [d for d in documents if d.stored_path.endswith(".pdf")]
    non_pdf = len(documents) - len(pdf_docs)
    queued = [d for d in pdf_docs if d.sha256 not in already]
    if args.limit:
        queued = queued[: args.limit]

    content_articles = [] if args.no_articles else [a for a in articles if not a.is_stub]

    if non_pdf:
        print(f"Note: skipping {non_pdf} non-PDF attachment(s) (images / Office documents).")
    print_plan(queued, len(content_articles), len(pdf_docs) - len(queued))

    if args.dry_run:
        print("Dry run: nothing parsed, nothing written.")
        return 0

    if not queued and not content_articles:
        print("Nothing to do. Everything is already parsed; use --overwrite to redo it.")
        return 1

    counts: collections.Counter = collections.Counter()
    families: collections.Counter = collections.Counter()

    with PagesWriter(overwrite=args.overwrite) as writer:
        # Diagnostic articles are cheap -- no PDF work at all -- so they land
        # first and the run has useful output immediately.
        for page in ingest_articles(content_articles):
            writer.write(page)
            _tally(page, counts, families)

        parse_documents(queued, args, writer, counts, families)

    summarise_pages(counts, families)
    print()
    print(f"Pages: {PAGES_PATH}")
    print("Next: Stage 3 chunking (scripts/04_index.py), or implement parse/vision.py")
    print("      to transcribe the pages flagged needs_vision.")
    return 0


def _write_rows(
    rows: list[dict],
    writer: PagesWriter,
    counts: collections.Counter,
    families: collections.Counter,
) -> None:
    """Write one document's page rows and tally them.

    Args:
        rows: Page records as dicts, from a worker.
        writer: Open pages writer.
        counts: Counter to update.
        families: Product-family counter to update.
    """
    for row in rows:
        try:
            page = Page.model_validate(row)
        except ParseError:
            continue
        writer.write(page)
        _tally(page, counts, families)


def _tally(page: Page, counts: collections.Counter, families: collections.Counter) -> None:
    """Update the run counters for one page.

    Args:
        page: The page just written.
        counts: Counter to update.
        families: Product-family counter to update.
    """
    counts["pages"] += 1
    counts[page.content_stream] += 1
    counts[page.tier] += 1
    counts["tables"] += len(page.tables)
    if page.image_path:
        counts["images"] += 1
    families[page.product_family] += 1

    if page.content_stream != "pdf":
        return
    # Read the recorded provenance rather than re-deriving it. A label that
    # happens to equal index+1 is not the same as a fallback, and treating them
    # alike would hide whether the page-accuracy gate is measurable at all.
    counts[f"label_{page.label_source}"] += 1


if __name__ == "__main__":
    sys.exit(main())
