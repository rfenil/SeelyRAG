#!/usr/bin/env python
"""Stage 3 -- chunk the parsed corpus and build the fault-code table.

build-plan.md sections 5.1 and 5.3.

Reads ``data/01_interim/pages.jsonl`` and writes ``data/02_processed/chunks.jsonl``
plus ``codes.jsonl``.

Chunks are page-anchored: a chunk never spans two pages, so every citation the
system emits resolves to exactly one printed page. The single exception is a
table merged across consecutive pages, which stays anchored to the page it
started on and records a ``page_range``.

Chunk ids are deterministic and every chunk carries the SHA-256 of its final
text, so re-running after a change reports what actually moved instead of
rewriting the corpus blind. That is what makes the later vision backfill --
3,459 pages currently awaiting transcription -- an incremental update rather
than a rebuild.

Stage 4 (embedding and the LanceDB index) is a separate step and is not run
here; ``--stats`` reports what it would cost.

Exit codes:
    0 -- chunking completed.
    1 -- pages.jsonl is missing or empty.
"""

from __future__ import annotations

import argparse
import collections
import sys

from seeley_rag.chunk.base import Chunk, JsonlWriter, chunk_hashes
from seeley_rag.chunk.chunker import chunk_corpus
from seeley_rag.chunk.codes import annotate_chunks, build_code_table
from seeley_rag.exceptions import ParseError
from seeley_rag.logging_conf import configure_logging, get_logger
from seeley_rag.parse.base import read_pages
from seeley_rag.paths import CHUNKS_PATH, CODES_PATH, PAGES_PATH, ensure_dirs
from seeley_rag.settings import get_settings

log = get_logger(__name__)

#: USD per million input tokens for text-embedding-3-large. Used only for the
#: cost line in the summary; the index stage does not read it.
EMBED_USD_PER_MTOK = 0.13


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Chunk parsed pages into chunks.jsonl and codes.jsonl.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--limit", type=int, default=None, help="Chunk at most N pages.")
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Report what would be produced without writing anything.",
    )
    parser.add_argument(
        "--include-vision-pending",
        action="store_true",
        help=(
            "Also chunk pages still awaiting vision transcription. Off by "
            "default: their text is empty or fragmentary, so they would add "
            "citable-but-contentless rows to the index."
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="Log at DEBUG level.")
    return parser.parse_args()


def summarise(chunks: list[Chunk]) -> dict[str, object]:
    """Build the human-facing summary of a chunking run.

    Args:
        chunks: Every chunk produced.

    Returns:
        A mapping of summary fields.
    """
    tokens = sum(c.token_count for c in chunks)
    tables = sum(1 for c in chunks if c.is_table)
    merged = sum(1 for c in chunks if c.page_span > 1)
    families = collections.Counter(c.product_family for c in chunks)
    streams = collections.Counter(c.content_stream for c in chunks)
    oversized = sum(1 for c in chunks if c.token_count > get_settings().chunk.max_tokens)
    with_codes = sum(1 for c in chunks if c.fault_codes)
    return {
        "chunks": len(chunks),
        "tables": tables,
        "merged_tables": merged,
        "prose": len(chunks) - tables,
        "tokens": tokens,
        "mean_tokens": round(tokens / len(chunks)) if chunks else 0,
        "oversized_prose": oversized - tables if oversized > tables else 0,
        "chunks_with_codes": with_codes,
        "families": dict(families.most_common()),
        "streams": dict(streams),
        "embed_usd": round(tokens / 1_000_000 * EMBED_USD_PER_MTOK, 2),
    }


def main() -> int:
    """Chunk the corpus and write the Stage 3 outputs.

    Returns:
        Process exit code.
    """
    args = parse_args()
    configure_logging(level="DEBUG" if args.verbose else None)
    ensure_dirs()

    try:
        pages = list(read_pages())
    except ParseError as exc:
        print(f"Cannot chunk: {exc}")
        return 1

    if not pages:
        print(f"No pages in {PAGES_PATH}. Run `python scripts/03_parse.py` first.")
        return 1

    skipped_vision = 0
    skipped_empty = 0
    selected = []
    for page in pages:
        if not page.has_content:
            skipped_empty += 1
            continue
        if page.needs_vision and not args.include_vision_pending and not page.text.strip():
            skipped_vision += 1
            continue
        selected.append(page)
        if args.limit and len(selected) >= args.limit:
            break

    previous = chunk_hashes()
    chunks = annotate_chunks(list(chunk_corpus(selected)))
    stats = summarise(chunks)

    fresh = {c.chunk_id: c.content_hash for c in chunks}
    unchanged = sum(1 for cid, h in fresh.items() if previous.get(cid) == h)
    added = sum(1 for cid in fresh if cid not in previous)
    changed = sum(1 for cid, h in fresh.items() if cid in previous and previous[cid] != h)
    removed = sum(1 for cid in previous if cid not in fresh)

    print(f"Pages read:        {len(pages):,}")
    print(f"  chunked:         {len(selected):,}")
    print(f"  skipped (empty): {skipped_empty:,}")
    print(f"  skipped (vision pending, no text): {skipped_vision:,}")
    print()
    print(f"Chunks:            {stats['chunks']:,}")
    print(f"  prose:           {stats['prose']:,}")
    print(
        f"  tables:          {stats['tables']:,} "
        f"({stats['merged_tables']:,} merged across pages)"
    )
    print(f"  carrying codes:  {stats['chunks_with_codes']:,}")
    print(f"Tokens:            {stats['tokens']:,} (mean {stats['mean_tokens']}/chunk)")
    print(f"Embedding cost:    ${stats['embed_usd']:.2f} at ${EMBED_USD_PER_MTOK}/M tokens")
    print()
    # This split is the point of the whole incremental design: Stage 4 embeds
    # only `new + changed`, which is what turns a re-index from hours into
    # minutes -- the eventual vision backfill included.
    print("Against existing chunks.jsonl:")
    print(f"  unchanged:       {unchanged:,} (embeddings still valid)")
    print(f"  changed:         {changed:,} (must re-embed)")
    print(f"  new:             {added:,} (must embed)")
    print(f"  gone:            {removed:,} (drop from the index)")
    print(f"Families: {stats['families']}")

    if args.stats:
        print("\n--stats: nothing written.")
        return 0

    codes = build_code_table(chunks)

    with JsonlWriter(CHUNKS_PATH) as writer:
        writer.write_all(chunks)
    with JsonlWriter(CODES_PATH) as writer:
        writer.write_all(codes)

    print()
    print(f"Wrote {len(chunks):,} chunks -> {CHUNKS_PATH}")
    print(f"Wrote {len(codes):,} fault codes -> {CODES_PATH}")
    log.info(
        "chunking_complete",
        extra={"chunks": len(chunks), "codes": len(codes), "tokens": stats["tokens"]},
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
