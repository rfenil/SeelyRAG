"""Stage 4 -- vector store protocol and the LanceDB implementation.

build-plan.md section 6.

LanceDB is the POC choice: embedded, no server, native vector + FTS + hybrid,
one pip install. For a short build the ops cost of Qdrant or Postgres+pgvector
is time we do not have.

Everything the retrieval cascade needs is declared on :class:`VectorStore`, and
nothing outside this module imports ``lancedb``. That is what keeps the
production swap -- pgvector inside the RosteredAI Postgres estate, or Qdrant
past single-node -- a one-file change.

Two properties this implementation is built around:

* **Upsert, not append.** Rows are merged on ``chunk_id`` via LanceDB's
  ``merge_insert``, so re-indexing a changed subset updates those rows in place
  instead of duplicating them. Combined with Stage 3's deterministic ids, that
  is what makes the deferred vision backfill an update rather than a rebuild.
* **Native Rust FTS, not Tantivy.** LanceDB is removing Tantivy support, and the
  legacy path is local-disk only and fully reindexes on every write.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

from seeley_rag.exceptions import ConfigurationError, SeeleyRagError
from seeley_rag.logging_conf import get_logger
from seeley_rag.settings import get_settings

log = get_logger(__name__)

#: Columns carried in the store. Mirrors the Chunk model field for field, so
#: retrieval reads one shape from acquisition through to citation.
CHUNK_COLUMNS: tuple[str, ...] = (
    "chunk_id",
    "doc_id",
    "text",
    "content_hash",
    "token_count",
    "kind",
    "page_index",
    "page_label",
    "label_source",
    "page_range",
    "page_span",
    "page_image",
    "fault_codes",
    "source_article_ids",
    "product_family",
    "model_series",
    "doc_type",
    "title",
    "source_url",
    "article_url",
    "category",
    "folder",
    "tier",
    "content_stream",
    "needs_vision",
    "is_table",
)


class StoreError(SeeleyRagError):
    """The vector store could not be opened, written or queried."""


class VectorStore(Protocol):
    """The minimum surface the retrieval cascade needs from a store."""

    def search_dense(
        self, vector: Sequence[float], top_k: int, where: str | None = None
    ) -> list[dict[str, Any]]:
        """Vector similarity search.

        Args:
            vector: Query embedding.
            top_k: Results to return.
            where: Optional pre-filter predicate.

        Returns:
            Matching chunk records with scores.
        """
        ...

    def search_bm25(self, query: str, top_k: int, where: str | None = None) -> list[dict[str, Any]]:
        """Full-text search. Catches model numbers, part numbers, code strings.

        Args:
            query: Query text.
            top_k: Results to return.
            where: Optional pre-filter predicate.

        Returns:
            Matching chunk records with scores.
        """
        ...


def _chunk_to_row(chunk: Any, vector: Sequence[float]) -> dict[str, Any]:
    """Flatten a chunk and its vector into a store row.

    Args:
        chunk: A :class:`~seeley_rag.chunk.base.Chunk`.
        vector: Its embedding.

    Returns:
        A row keyed by :data:`CHUNK_COLUMNS` plus ``vector``.
    """
    row = chunk.model_dump(mode="json")
    # `is_table` is a property on the model, so it is absent from the dump, but
    # retrieval filters on it and a Protocol-level filter cannot call a Python
    # property. Materialise it.
    row["is_table"] = chunk.is_table
    row["vector"] = list(vector)
    return row


class LanceDBStore:
    """A LanceDB-backed :class:`VectorStore`.

    Args:
        path: Store directory. Defaults to ``data/03_index``.
        table_name: Table to use. Defaults to configured.

    Attributes:
        path: Where the store lives.
        table_name: The table's name.
    """

    def __init__(self, path: Path | None = None, table_name: str | None = None) -> None:
        from seeley_rag.paths import INDEX_DIR

        self.path = path or INDEX_DIR
        self.table_name = table_name or get_settings().index.table_name
        self._db: Any = None
        self._table: Any = None

    # -- connection --------------------------------------------------------

    @property
    def db(self) -> Any:
        """The LanceDB connection, opened on first use.

        Returns:
            The connection.

        Raises:
            ConfigurationError: If lancedb is not installed.
        """
        if self._db is None:
            try:
                import lancedb
            except ImportError as exc:
                raise ConfigurationError(
                    'lancedb is not installed. Run: pip install -e ".[downstream]"'
                ) from exc
            self.path.mkdir(parents=True, exist_ok=True)
            self._db = lancedb.connect(str(self.path))
        return self._db

    @property
    def table(self) -> Any:
        """The chunks table.

        Returns:
            The table handle.

        Raises:
            StoreError: If the table does not exist yet.
        """
        if self._table is None:
            if not self.exists():
                raise StoreError(
                    f"No table {self.table_name!r} in {self.path}. "
                    "Run `python scripts/05_embed.py` to build the index."
                )
            self._table = self.db.open_table(self.table_name)
        return self._table

    def table_names(self) -> list[str]:
        """Return the names of every table in the store.

        LanceDB deprecated ``table_names()`` in favour of a paginated
        ``list_tables()`` that returns a response object rather than a list.
        Iterating that object yields *field* tuples, not names, so it is unwrapped
        explicitly here. The old call is kept as a fallback because
        ``pyproject.toml`` admits lancedb from 0.24.

        Returns:
            Table names.
        """
        lister = getattr(self.db, "list_tables", None)
        if lister is None:
            return list(self.db.table_names())

        names: list[str] = []
        token: str | None = None
        while True:
            response = lister(page_token=token) if token else lister()
            names.extend(getattr(response, "tables", []) or [])
            token = getattr(response, "page_token", None)
            if not token:
                return names

    def exists(self) -> bool:
        """Whether the table has been created.

        Returns:
            True when the table is present.
        """
        return self.table_name in self.table_names()

    def count(self) -> int:
        """Return the number of rows in the table.

        Returns:
            Row count, or 0 when the table does not exist.
        """
        return self.table.count_rows() if self.exists() else 0

    # -- writing -----------------------------------------------------------

    def upsert(self, chunks: Iterable[Any], vectors: Sequence[Sequence[float]]) -> int:
        """Insert or update rows, matched on ``chunk_id``.

        Upsert rather than append is the whole basis of incremental indexing:
        re-running over a changed subset must update those rows, not duplicate
        them.

        Args:
            chunks: Chunk records.
            vectors: One vector per chunk, in the same order.

        Returns:
            How many rows were written.

        Raises:
            StoreError: If the counts disagree.
        """
        materialised = list(chunks)
        if len(materialised) != len(vectors):
            raise StoreError(
                f"Got {len(vectors)} vectors for {len(materialised)} chunks; "
                "they must correspond one to one."
            )
        if not materialised:
            return 0

        rows = [_chunk_to_row(c, v) for c, v in zip(materialised, vectors)]

        if not self.exists():
            self._table = self.db.create_table(self.table_name, data=rows)
            log.info("index_table_created", extra={"rows": len(rows), "path": str(self.path)})
            return len(rows)

        (
            self.table.merge_insert("chunk_id")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(rows)
        )
        log.info("index_rows_upserted", extra={"rows": len(rows)})
        return len(rows)

    def delete_ids(self, chunk_ids: Sequence[str]) -> int:
        """Remove rows by chunk id.

        Needed by the incremental path: a chunk that disappears from
        ``chunks.jsonl`` -- because its page was re-chunked into fewer pieces --
        must leave the index too, or retrieval keeps serving a stale row that no
        longer corresponds to any source text.

        Args:
            chunk_ids: Ids to remove.

        Returns:
            How many ids were requested for deletion.
        """
        if not chunk_ids or not self.exists():
            return 0
        # Chunk ids are hex digests and colons -- no quotes to escape -- but the
        # quoting is explicit rather than assumed.
        quoted = ", ".join("'" + cid.replace("'", "''") + "'" for cid in chunk_ids)
        self.table.delete(f"chunk_id IN ({quoted})")
        log.info("index_rows_deleted", extra={"rows": len(chunk_ids)})
        return len(chunk_ids)

    def create_fts_index(self, replace: bool = True) -> None:
        """Build the full-text index over ``text``.

        BM25 is not a nicety here: it is what catches model numbers, part
        numbers and code strings, which are exactly the tokens dense retrieval
        handles worst.

        Args:
            replace: Rebuild an existing index.
        """
        # lancedb >= 0.25 deprecates create_fts_index in favour of
        # `create_index(config=FTS())` -- but that form exists only on the async
        # table API; the synchronous LanceTable.create_index takes vector-index
        # arguments and has no `config` parameter at all. So this is still the
        # only route on the sync client, and the warning is silenced here rather
        # than project-wide, so a genuine deprecation elsewhere still fails the
        # suite. Revisit when the sync API grows the new form.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*create_fts_index is deprecated.*")
            self.table.create_fts_index("text", replace=replace, use_tantivy=False)
        log.info("fts_index_built", extra={"table": self.table_name})

    def create_vector_index(self, replace: bool = True) -> bool:
        """Build the ANN index over ``vector``.

        Skipped below a few thousand rows: LanceDB's IVF-PQ needs enough vectors
        to train its partitions, and under that threshold a brute-force scan is
        both exact and fast. Returning the decision rather than raising lets the
        caller report it honestly.

        Args:
            replace: Rebuild an existing index.

        Returns:
            True when an index was built, False when the table is too small.
        """
        rows = self.count()
        if rows < 5000:
            log.info("vector_index_skipped", extra={"rows": rows, "reason": "too few rows"})
            return False
        self.table.create_index(metric="cosine", replace=replace)
        log.info("vector_index_built", extra={"rows": rows})
        return True

    # -- reading -----------------------------------------------------------

    def search_dense(
        self, vector: Sequence[float], top_k: int = 30, where: str | None = None
    ) -> list[dict[str, Any]]:
        """Vector similarity search.

        Args:
            vector: Query embedding.
            top_k: Results to return.
            where: Optional SQL predicate, applied as a **pre**-filter so the
                search runs over matching rows rather than trimming its output.
                Post-filtering a top-k list returns nothing when the matches sit
                below the cut, which is a filter that only works when it was not
                needed.

        Returns:
            Matching chunk records, each with a ``score``.
        """
        results = self.table.search(list(vector), vector_column_name="vector")
        if where:
            results = results.where(where, prefilter=True)
        return _normalise(results.limit(top_k).to_list(), distance_key="_distance")

    def search_bm25(
        self, query: str, top_k: int = 30, where: str | None = None
    ) -> list[dict[str, Any]]:
        """Full-text search over the chunk text.

        Args:
            query: Query text.
            top_k: Results to return.
            where: Optional SQL predicate, applied as a pre-filter.

        Returns:
            Matching chunk records, each with a ``score``.
        """
        results = self.table.search(query, query_type="fts")
        if where:
            results = results.where(where, prefilter=True)
        return _normalise(results.limit(top_k).to_list(), distance_key="_score")

    def get(self, chunk_id: str) -> dict[str, Any] | None:
        """Fetch one row by id, for pinning a fault-code hit into context.

        Args:
            chunk_id: The chunk to fetch.

        Returns:
            The row, or ``None`` when absent.
        """
        quoted = chunk_id.replace("'", "''")
        rows = self.table.search().where(f"chunk_id = '{quoted}'").limit(1).to_list()
        return _strip_vector(rows[0]) if rows else None


def _strip_vector(row: dict[str, Any]) -> dict[str, Any]:
    """Drop the embedding from a result row.

    A 3,072-float vector in every result would dominate logs and API responses,
    and nothing downstream of retrieval reads it.

    Args:
        row: A result row.

    Returns:
        The row without ``vector``.
    """
    return {k: v for k, v in row.items() if k != "vector"}


def _normalise(rows: list[dict[str, Any]], distance_key: str) -> list[dict[str, Any]]:
    """Give every result a uniform ``score`` and drop its vector.

    Dense search returns a cosine *distance* (lower is better) while FTS returns
    a relevance *score* (higher is better). RRF fuses on rank so it is immune to
    the difference, but anything reading ``score`` directly would silently
    invert one of them.

    Args:
        rows: Raw result rows.
        distance_key: ``_distance`` for dense, ``_score`` for FTS.

    Returns:
        Rows with a comparable ``score``, highest first.
    """
    out: list[dict[str, Any]] = []
    for row in rows:
        cleaned = _strip_vector(row)
        raw = row.get(distance_key)
        if raw is None:
            cleaned["score"] = 0.0
        elif distance_key == "_distance":
            cleaned["score"] = 1.0 - float(raw)
        else:
            cleaned["score"] = float(raw)
        out.append(cleaned)
    return out


def open_store(path: Path | None = None, table_name: str | None = None) -> LanceDBStore:
    """Open the configured vector store.

    Args:
        path: Store location. Defaults to ``data/03_index``.
        table_name: Table to use. Defaults to configured.

    Returns:
        A :class:`LanceDBStore`.
    """
    return LanceDBStore(path=path, table_name=table_name)
