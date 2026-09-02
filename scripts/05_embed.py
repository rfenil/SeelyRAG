#!/usr/bin/env python
"""Stage 4 -- embed chunks and build the LanceDB vector + full-text index.

build-plan.md section 6.

Reads ``data/02_processed/chunks.jsonl`` and writes ``data/03_index/``.

Incremental by default. Chunks already in the index whose ``content_hash`` is
unchanged are not re-embedded, chunks that have changed are updated in place on
``chunk_id``, and rows with no counterpart on disk are deleted. On an unchanged
corpus this costs zero API calls, which is what makes iterating on chunk
boundaries -- and the deferred vision backfill -- cheap.

Embeddings are cached under ``data/cache/embeddings/{model}-{dim}/`` by the hash
of the exact text sent. The cache survives a deleted index, so rebuilding the
store from scratch after a schema change is free.

Start with ``--smoke``: it embeds 200 chunks into a throwaway table, runs a
dense and a BM25 query against them, and reports what it found. Roughly one US
cent, and it verifies the whole path before the full run.

Exit codes:
    0 -- completed.
    1 -- chunks.jsonl missing, no API key, or a verification check failed.
"""

from __future__ import annotations

import argparse
import sys
import time

from seeley_rag.chunk.base import read_chunks
from seeley_rag.exceptions import ConfigurationError, ParseError, SeeleyRagError
from seeley_rag.index.build import build_index, plan_build, search_smoke, verify_index
from seeley_rag.index.embedder import Embedder
from seeley_rag.index.store import open_store
from seeley_rag.logging_conf import configure_logging, get_logger
from seeley_rag.paths import CHUNKS_PATH, INDEX_DIR, ensure_dirs
from seeley_rag.settings import get_settings

log = get_logger(__name__)

#: USD per million input tokens for text-embedding-3-large.
EMBED_USD_PER_MTOK = 0.13

#: Table used by --smoke, kept separate so a trial never touches the real index.
SMOKE_TABLE = "chunks_smoke"

#: Queries the smoke test runs. Chosen to exercise both retrieval channels: a
#: fault code (which BM25 should nail and dense search should struggle with) and
#: a natural-language symptom (the reverse).
SMOKE_QUERIES = (
    "TQ heater fault code FC7 flame sensing",
    "gas heater will not ignite and locks out",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Embed chunks and build the LanceDB index.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Embed 200 chunks into a throwaway table and query them. ~1 cent.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Index at most N chunks.")
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Report what would be embedded and what it would cost. No API calls.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Texts per embedding request.")
    parser.add_argument(
        "--no-indexes",
        action="store_true",
        help="Skip the ANN and FTS index builds (they can be built later).",
    )
    parser.add_argument("--verbose", action="store_true", help="Log at DEBUG level.")
    return parser.parse_args()


def _progress(done: int, total: int) -> None:
    """Print a single-line progress indicator.

    Args:
        done: Texts embedded so far.
        total: Texts to embed.
    """
    pct = done / total * 100 if total else 100.0
    print(f"\r  embedding {done:,}/{total:,} ({pct:.1f}%)", end="", flush=True)


def run_plan(chunks: list, limit: int | None) -> int:
    """Report the build plan without calling the API.

    Args:
        chunks: Chunks read from disk.
        limit: Optional cap.

    Returns:
        Process exit code.
    """
    plan = plan_build(chunks, open_store())
    tokens = sum(c.token_count for c in plan.to_embed)
    print(f"Chunks on disk:    {len(chunks):,}" + (f" (limited to {limit:,})" if limit else ""))
    print()
    print(f"  unchanged:       {plan.summary()['unchanged']:,} (already embedded)")
    print(f"  changed:         {plan.summary()['changed']:,} (must re-embed)")
    print(f"  new:             {plan.summary()['new']:,} (must embed)")
    print(f"  gone:            {plan.summary()['removed']:,} (drop from the index)")
    print()
    print(f"Would embed:       {len(plan.to_embed):,} chunks, {tokens:,} tokens")
    print(f"Estimated cost:    ${tokens / 1_000_000 * EMBED_USD_PER_MTOK:.2f}")
    print("\n--plan: no API calls made, nothing written.")
    return 0


def run_smoke(chunks: list, batch_size: int | None) -> int:
    """Embed a small sample into a throwaway table and query it.

    Args:
        chunks: Chunks read from disk.
        batch_size: Texts per request.

    Returns:
        Process exit code.
    """
    settings = get_settings().index
    sample = chunks[:200]
    tokens = sum(c.token_count for c in sample)
    print(f"Smoke test: {len(sample)} chunks, {tokens:,} tokens, table {SMOKE_TABLE!r}")
    print(f"Estimated cost: ${tokens / 1_000_000 * EMBED_USD_PER_MTOK:.4f}\n")

    store = open_store(table_name=SMOKE_TABLE)
    # A smoke test must exercise a real embed, so any stale table is dropped
    # first -- otherwise the incremental logic would correctly skip everything
    # and verify nothing.
    if store.exists():
        store.db.drop_table(SMOKE_TABLE)
        store._table = None

    embedder = Embedder()
    started = time.monotonic()
    report = build_index(
        sample,
        embedder=embedder,
        batch_size=batch_size,
        build_indexes=False,
        on_progress=_progress,
        store=store,
    )
    print(f"\n\nEmbedded in {time.monotonic() - started:.1f}s")
    print(f"  API requests:    {report['embedder']['requests']}")
    print(f"  from cache:      {report['embedder']['cached']:,}")
    print(f"  rows written:    {report['rows_written']:,}")

    checks = verify_index(store, settings.embedding_dim)
    print(f"\nVerification: {checks['rows']} rows, vector width {checks['vector_dim']}")
    if not checks["dim_matches"]:
        print(f"FAIL: expected width {settings.embedding_dim}, got {checks['vector_dim']}.")
        return 1
    if not checks["ids_present"]:
        print("FAIL: rows are missing chunk_id.")
        return 1

    store.create_fts_index()
    for query in SMOKE_QUERIES:
        print(f"\nQuery: {query!r}")
        results = search_smoke(store, embedder, query, top_k=3)
        for channel in ("dense", "bm25"):
            print(f"  {channel}:")
            if not results[channel]:
                print("    (no results)")
            for row in results[channel]:
                page = f"p.{row['page_label']}" if row["page_label"] else "no page"
                print(
                    f"    {row['score']:.4f} [{row['product_family']}/{row['kind']}] "
                    f"{row['title'][:52]} ({page})"
                )

    print(f"\nSmoke test passed. Table {SMOKE_TABLE!r} left in place for inspection.")
    print("Run without --smoke to build the real index.")
    return 0


def print_report(report: dict, elapsed: float) -> None:
    """Print the human-facing summary of an index build.

    Args:
        report: The report returned by :func:`build_index`.
        elapsed: Wall-clock seconds the build took.
    """
    stats = report["embedder"]
    print(f"\n\nDone in {elapsed:.1f}s")
    print()
    print(f"  unchanged:       {report['unchanged']:,} (no API call)")
    print(f"  changed:         {report['changed']:,}")
    print(f"  new:             {report['new']:,}")
    print(f"  removed:         {report['removed']:,}")
    print()
    print(f"  API requests:    {stats.get('requests', 0):,}")
    print(f"  cache hits:      {stats.get('cached', 0):,}")
    print(f"  rows written:    {report['rows_written']:,}")
    print(f"  rows in index:   {report['rows_total']:,}")

    # "unchanged" and "not built" are different states and must not read the
    # same: nothing moved, versus the table being too small for an ANN index to
    # be trainable at all.
    moved = report["rows_written"] or report["rows_deleted"]
    fts = "built" if report["fts_index_built"] else ("unchanged" if not moved else "skipped")
    if report["vector_index_built"]:
        ann = "built"
    elif not moved:
        ann = "unchanged"
    else:
        ann = "brute force (too few rows to train)"
    print(f"  FTS index:       {fts}")
    print(f"  vector index:    {ann}")


def main() -> int:
    """Embed the corpus and build the index.

    Returns:
        Process exit code.
    """
    args = parse_args()
    configure_logging(level="DEBUG" if args.verbose else None)
    ensure_dirs()
    settings = get_settings()

    try:
        chunks = list(read_chunks())
    except ParseError as exc:
        print(f"Cannot index: {exc}")
        return 1

    if not chunks:
        print(f"No chunks in {CHUNKS_PATH}. Run `python scripts/04_index.py` first.")
        return 1
    if args.limit:
        chunks = chunks[: args.limit]

    if args.plan:
        return run_plan(chunks, args.limit)

    if not settings.openai_api_key:
        print("OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in.")
        return 1

    try:
        if args.smoke:
            return run_smoke(chunks, args.batch_size)

        tokens = sum(c.token_count for c in chunks)
        print(f"Indexing {len(chunks):,} chunks ({tokens:,} tokens) -> {INDEX_DIR}")
        print(f"Model: {settings.index.embedding_model} at {settings.index.embedding_dim}-d\n")

        started = time.monotonic()
        report = build_index(
            chunks,
            batch_size=args.batch_size,
            build_indexes=not args.no_indexes,
            on_progress=_progress,
        )
        elapsed = time.monotonic() - started

        print_report(report, elapsed)

        if report["rows_total"]:
            checks = verify_index(open_store(), settings.index.embedding_dim)
            if not checks["dim_matches"]:
                print(f"\nFAIL: vector width is {checks['vector_dim']}.")
                return 1
            print(f"\nVerified: {checks['rows']:,} rows at {checks['vector_dim']}-d.")
    except (ConfigurationError, SeeleyRagError) as exc:
        print(f"\nIndexing failed: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
