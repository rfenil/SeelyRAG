"""Stage 4 -- index build.

build-plan.md section 6.

Embeds chunks with ``text-embedding-3-large`` (3,072-d) and builds both the
vector index and ``create_fts_index("text")`` on LanceDB's native Rust FTS.

The work this module actually does is *deciding what not to embed*. Given the
chunks on disk and the rows already in the store, it partitions the corpus into
unchanged, changed, new and removed, and touches only the last three. On an
unchanged corpus a rebuild costs zero API calls; after the vision backfill it
will cost only the transcribed pages.

That partition is possible because Stage 3 guarantees two things: ``chunk_id``
is deterministic, and ``content_hash`` covers exactly the text that gets
embedded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, NamedTuple, Sequence

from seeley_rag.index.embedder import Embedder
from seeley_rag.index.store import LanceDBStore, open_store
from seeley_rag.logging_conf import get_logger

log = get_logger(__name__)


class IndexPlan(NamedTuple):
    """What a build would do, before it does it.

    Attributes:
        new: Chunks absent from the store.
        changed: Chunks whose text has changed since they were indexed.
        unchanged: Chunks whose stored vector is still correct.
        removed: Chunk ids in the store with no counterpart on disk.
    """

    new: list[Any]
    changed: list[Any]
    unchanged: list[Any]
    removed: list[str]

    @property
    def to_embed(self) -> list[Any]:
        """Chunks that need an embedding call."""
        return self.new + self.changed

    def summary(self) -> dict[str, int]:
        """Return the counts, for reporting.

        Returns:
            One entry per partition.
        """
        return {
            "new": len(self.new),
            "changed": len(self.changed),
            "unchanged": len(self.unchanged),
            "removed": len(self.removed),
            "to_embed": len(self.to_embed),
        }


def indexed_hashes(store: LanceDBStore) -> dict[str, str]:
    """Return ``chunk_id -> content_hash`` for everything already indexed.

    Args:
        store: The store to inspect.

    Returns:
        The mapping. Empty when the table does not exist.
    """
    if not store.exists():
        return {}
    # A projected scan through LanceDB's own query builder, not `to_lance()`:
    # that route needs the separate `pylance` package, and selecting the two
    # columns keeps 3,072-float vectors out of a scan that only needs hashes.
    table = store.table
    rows = table.search().select(["chunk_id", "content_hash"]).limit(table.count_rows()).to_list()
    return {row["chunk_id"]: row["content_hash"] for row in rows}


def plan_build(chunks: Iterable[Any], store: LanceDBStore) -> IndexPlan:
    """Work out which chunks actually need embedding.

    Args:
        chunks: Every chunk from ``chunks.jsonl``.
        store: The target store.

    Returns:
        The partition.
    """
    existing = indexed_hashes(store)
    new: list[Any] = []
    changed: list[Any] = []
    unchanged: list[Any] = []

    seen: set[str] = set()
    for chunk in chunks:
        seen.add(chunk.chunk_id)
        stored = existing.get(chunk.chunk_id)
        if stored is None:
            new.append(chunk)
        elif stored != chunk.content_hash:
            changed.append(chunk)
        else:
            unchanged.append(chunk)

    removed = [cid for cid in existing if cid not in seen]
    plan = IndexPlan(new=new, changed=changed, unchanged=unchanged, removed=removed)
    log.info("index_plan", extra=plan.summary())
    return plan


def embed_chunks(
    chunks: list[Any],
    batch_size: int | None = None,
    embedder: Embedder | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[list[float]]:
    """Embed chunk texts, consulting the cache first.

    Args:
        chunks: Chunk records.
        batch_size: Embedding request batch size. Defaults to configured.
        embedder: Injected embedder. Defaults to a new one, which needs a key.
        on_progress: Optional ``(done, total)`` callable.

    Returns:
        One vector per chunk, in order.
    """
    client = embedder or Embedder()
    return client.embed_chunks(chunks, batch_size=batch_size, on_progress=on_progress)


def build_index(
    chunks: Iterable[Any],
    index_dir: Path | None = None,
    embedder: Embedder | None = None,
    batch_size: int | None = None,
    build_indexes: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
    store: LanceDBStore | None = None,
) -> dict[str, Any]:
    """Build or update the vector and full-text indexes.

    Args:
        chunks: Chunk records.
        index_dir: Destination. Defaults to ``data/03_index``.
        embedder: Injected embedder. Defaults to a new one.
        batch_size: Embedding request batch size.
        build_indexes: Whether to (re)build the ANN and FTS indexes afterwards.
            Off during a smoke test, where the table is deliberately tiny.
        on_progress: Optional ``(done, total)`` callable for the embedding pass.
        store: An already-opened store. Pass one to target a table other than
            the configured default -- the smoke test does, so a trial run
            cannot write into the real index.

    Returns:
        A report: the plan's counts, rows written, and embedder statistics.
    """
    store = store if store is not None else open_store(index_dir)
    plan = plan_build(list(chunks), store)

    written = 0
    if plan.to_embed:
        client = embedder or Embedder()
        vectors = client.embed_chunks(plan.to_embed, batch_size=batch_size, on_progress=on_progress)
        written = store.upsert(plan.to_embed, vectors)
        stats = client.stats()
    else:
        stats = embedder.stats() if embedder else {"requests": 0, "embedded": 0, "cached": 0}

    deleted = store.delete_ids(plan.removed)

    fts_built = False
    ann_built = False
    if build_indexes and store.exists() and store.count():
        # Only worth rebuilding when rows actually moved; both passes scan the
        # whole table.
        if written or deleted:
            store.create_fts_index()
            fts_built = True
            ann_built = store.create_vector_index()

    report: dict[str, Any] = {
        **plan.summary(),
        "rows_written": written,
        "rows_deleted": deleted,
        "rows_total": store.count(),
        "fts_index_built": fts_built,
        "vector_index_built": ann_built,
        "embedder": stats,
    }
    log.info("index_build_complete", extra={k: v for k, v in report.items() if k != "embedder"})
    return report


def verify_index(store: LanceDBStore, expected_dim: int, sample: int = 5) -> dict[str, Any]:
    """Sanity-check a built index.

    A vector of the wrong width, or a row whose ``chunk_id`` is missing, fails
    at query time with an error that looks like a retrieval bug. Checking here
    keeps the diagnosis where the cause is.

    Args:
        store: The store to check.
        expected_dim: Vector width the configuration asked for.
        sample: How many rows to inspect.

    Returns:
        Row count, observed vector width, and whether it matches.

    Raises:
        StoreError: If the table is missing or empty.
    """
    from seeley_rag.index.store import StoreError

    if not store.exists():
        raise StoreError(f"No index at {store.path}.")
    rows = store.table.search().select(["chunk_id", "vector"]).limit(sample).to_list()
    if not rows:
        raise StoreError(f"Index at {store.path} has no rows.")

    widths = {len(row["vector"]) for row in rows}
    observed = widths.pop() if len(widths) == 1 else -1
    return {
        "rows": store.count(),
        "sampled": len(rows),
        "vector_dim": observed,
        "dim_matches": observed == expected_dim,
        "ids_present": all(row["chunk_id"] for row in rows),
    }


def search_smoke(
    store: LanceDBStore, embedder: Embedder, query: str, top_k: int = 3
) -> dict[str, list[dict[str, Any]]]:
    """Run one dense and one BM25 query, for post-build verification.

    Args:
        store: The built store.
        embedder: Embedder for the query vector.
        query: Query text.
        top_k: Results per channel.

    Returns:
        ``{"dense": [...], "bm25": [...]}`` with trimmed rows.
    """
    vector = embedder.embed_texts([query])[0]
    return {
        "dense": _trim(store.search_dense(vector, top_k)),
        "bm25": _trim(store.search_bm25(query, top_k)),
    }


def _trim(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reduce result rows to what a human check needs.

    Args:
        rows: Search results.

    Returns:
        Just the citation fields and the score.
    """
    return [
        {
            "score": round(row.get("score", 0.0), 4),
            "title": row.get("title", ""),
            "page_label": row.get("page_label"),
            "product_family": row.get("product_family"),
            "kind": row.get("kind"),
            "text": (row.get("text", "")[:160] + "...") if row.get("text") else "",
        }
        for row in rows
    ]
