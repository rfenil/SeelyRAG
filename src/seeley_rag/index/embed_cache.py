"""Stage 4 -- embedding cache.

build-plan.md section 6.

Keyed by ``sha256(final_chunk_text)`` -- the *final* text, breadcrumb prefix
included, because that is what actually gets embedded. Stage 3 already computes
exactly that value and stores it on every chunk as
:attr:`~seeley_rag.chunk.base.Chunk.content_hash`, so the cache key is not
recomputed here; it is read off the chunk.

This one hour is what makes the rest of the work possible: it turns each
re-index from hours into minutes, and iteration on chunk boundaries, table
sizing and the eventual vision backfill is entirely re-index cycles. Written
*before* ``build.py`` deliberately -- a cache added after the first full
embedding run has already failed at its job.

Storage
-------
One shard file per two-character key prefix under ``data/cache/embeddings/``,
holding ``{key: [float, ...]}`` as JSON. 256 shards over ~16k chunks is roughly
60 vectors per shard.

Sharding rather than one big file, because the alternative fails at this corpus
size: 16,189 vectors at 3,072 float32 dimensions is ~200 MB, and rewriting all
of it after every batch is slower than the API call it exists to avoid. Sharding
rather than one file per key, because 16,189 loose files on NTFS is pathological
to enumerate and back up.

Vectors are stored at whatever width the model returned. A change to
``index.embedding_dim`` or ``index.embedding_model`` therefore invalidates the
cache -- which :func:`namespace` handles by putting the model and width in the
directory name, so the two never mix and switching back and forth costs nothing.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Iterable

from seeley_rag.exceptions import ConfigurationError
from seeley_rag.logging_conf import get_logger
from seeley_rag.settings import get_settings

log = get_logger(__name__)

#: Characters of the key used to pick a shard. Two hex characters give 256
#: shards, which keeps both the file count and the per-shard rewrite small.
SHARD_PREFIX_LEN = 2


def cache_key(chunk_text: str) -> str:
    """Return the cache key for a chunk's final text.

    Args:
        chunk_text: The exact text that will be embedded, breadcrumb included.

    Returns:
        Hex SHA-256. Identical to
        :attr:`~seeley_rag.chunk.base.Chunk.content_hash`, which is where
        callers should normally read it from rather than re-hashing.
    """
    from seeley_rag.chunk.base import content_hash

    return content_hash(chunk_text)


def namespace(model: str | None = None, dimensions: int | None = None) -> str:
    """Return the cache sub-directory name for a model and width.

    Vectors from different models, or the same model truncated to different
    widths, are not interchangeable. Keying the directory on both means a
    dimension experiment cannot silently read another run's vectors, and
    switching back recovers the old cache instead of re-embedding.

    Args:
        model: Embedding model. Defaults to the configured one.
        dimensions: Vector width. Defaults to the configured one.

    Returns:
        e.g. ``text-embedding-3-large-3072``.
    """
    settings = get_settings().index
    resolved_model = model or settings.embedding_model
    resolved_dim = dimensions or settings.embedding_dim
    return f"{resolved_model}-{resolved_dim}"


class EmbeddingCache:
    """A sharded, on-disk store of chunk-text hashes to vectors.

    Not thread-safe by accident -- writes are guarded by a lock because the
    embedding client may fan out batches, and two threads flushing the same
    shard would lose one of them.

    Args:
        root: Cache directory. Defaults to ``data/cache/embeddings/{namespace}``.
        model: Embedding model, for the namespace. Defaults to configured.
        dimensions: Vector width, for the namespace. Defaults to configured.

    Attributes:
        hits: Lookups served from disk.
        misses: Lookups that had to be embedded.
        writes: Vectors stored this session.
    """

    def __init__(
        self,
        root: Path | None = None,
        model: str | None = None,
        dimensions: int | None = None,
    ) -> None:
        from seeley_rag.paths import EMBEDDING_CACHE_DIR

        self.root = root or (EMBEDDING_CACHE_DIR / namespace(model, dimensions))
        self.hits = 0
        self.misses = 0
        self.writes = 0
        self._shards: dict[str, dict[str, list[float]]] = {}
        self._dirty: set[str] = set()
        self._lock = threading.Lock()

    # -- shard plumbing ----------------------------------------------------

    def _shard_name(self, key: str) -> str:
        """Return the shard a key belongs to.

        Args:
            key: Cache key.

        Returns:
            The shard's prefix.

        Raises:
            ConfigurationError: If the key is too short to shard.
        """
        if len(key) < SHARD_PREFIX_LEN:
            raise ConfigurationError(
                f"Cache key {key!r} is shorter than {SHARD_PREFIX_LEN} characters; "
                "keys must be hex SHA-256 digests."
            )
        return key[:SHARD_PREFIX_LEN]

    def _shard_path(self, shard: str) -> Path:
        """Return the file backing a shard.

        Args:
            shard: Shard prefix.

        Returns:
            Path to the shard's JSON file.
        """
        return self.root / f"{shard}.json"

    def _load_shard(self, shard: str) -> dict[str, list[float]]:
        """Load a shard, memoised for the session.

        A shard that is missing or unreadable is treated as empty. The cache is
        an optimisation: a corrupt shard must cost an embedding call, never the
        run.

        Args:
            shard: Shard prefix.

        Returns:
            The shard's contents.
        """
        if shard in self._shards:
            return self._shards[shard]

        path = self._shard_path(shard)
        loaded: dict[str, list[float]] = {}
        if path.exists():
            try:
                parsed = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(parsed, dict):
                    loaded = parsed
                else:
                    log.warning("cache_shard_malformed", extra={"shard": shard})
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("cache_shard_unreadable", extra={"shard": shard, "error": str(exc)})

        self._shards[shard] = loaded
        return loaded

    # -- public surface ----------------------------------------------------

    def get(self, key: str) -> list[float] | None:
        """Look up a cached embedding.

        Args:
            key: Cache key.

        Returns:
            The vector, or ``None`` on a miss.
        """
        vector = self._load_shard(self._shard_name(key)).get(key)
        if vector is None:
            self.misses += 1
            return None
        self.hits += 1
        return vector

    def get_many(self, keys: Iterable[str]) -> dict[str, list[float]]:
        """Look up several embeddings at once.

        Args:
            keys: Cache keys.

        Returns:
            Only the keys that hit, mapped to their vectors.
        """
        found: dict[str, list[float]] = {}
        for key in keys:
            vector = self.get(key)
            if vector is not None:
                found[key] = vector
        return found

    def put(self, key: str, vector: list[float]) -> None:
        """Store an embedding.

        Held in memory until :meth:`flush`, so a batch of 256 costs one file
        write rather than 256.

        Args:
            key: Cache key.
            vector: The embedding.
        """
        with self._lock:
            shard = self._shard_name(key)
            self._load_shard(shard)[key] = list(vector)
            self._dirty.add(shard)
            self.writes += 1

    def put_many(self, vectors: dict[str, list[float]]) -> None:
        """Store several embeddings.

        Args:
            vectors: Cache keys mapped to vectors.
        """
        for key, vector in vectors.items():
            self.put(key, vector)

    def flush(self) -> int:
        """Write every dirty shard to disk.

        Called after each batch, not only at the end: a long embedding run that
        is interrupted must not lose the work already paid for.

        Returns:
            How many shards were written.
        """
        with self._lock:
            if not self._dirty:
                return 0
            self.root.mkdir(parents=True, exist_ok=True)
            written = 0
            for shard in sorted(self._dirty):
                path = self._shard_path(shard)
                # Write-then-replace, so a kill mid-write cannot leave a
                # truncated shard that would read as a silent cache miss for
                # every key it holds.
                temporary = path.with_suffix(".json.tmp")
                temporary.write_text(
                    json.dumps(self._shards[shard], separators=(",", ":")), encoding="utf-8"
                )
                temporary.replace(path)
                written += 1
            self._dirty.clear()
            return written

    def __enter__(self) -> EmbeddingCache:
        """Enter the context."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Flush on the way out, including after an exception."""
        self.flush()

    @property
    def hit_rate(self) -> float:
        """Fraction of lookups served from disk this session."""
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def stats(self) -> dict[str, float | int | str]:
        """Return a summary for logging.

        Returns:
            Hits, misses, writes, hit rate and the cache location.
        """
        return {
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "hit_rate": round(self.hit_rate, 4),
            "root": str(self.root),
        }


# ---------------------------------------------------------------------------
# Module-level convenience, for the stub API the build plan names
# ---------------------------------------------------------------------------


_DEFAULT: EmbeddingCache | None = None


def default_cache() -> EmbeddingCache:
    """Return the process-wide cache for the configured model and width.

    Returns:
        The shared :class:`EmbeddingCache`.
    """
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = EmbeddingCache()
    return _DEFAULT


def get(key: str) -> list[float] | None:
    """Look up a cached embedding in the default cache.

    Args:
        key: Cache key.

    Returns:
        The vector, or ``None`` on a miss.
    """
    return default_cache().get(key)


def put(key: str, vector: list[float]) -> None:
    """Store an embedding in the default cache.

    Args:
        key: Cache key.
        vector: The embedding.
    """
    default_cache().put(key, vector)
    default_cache().flush()
