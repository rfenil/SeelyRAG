"""Stage 4 embedding cache.

build-plan section 6. The cache is what turns each re-index from hours into
minutes, so its correctness is worth more than its speed: a false hit attaches
the wrong vector to a chunk, which is a silent retrieval failure rather than a
loud one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from seeley_rag.chunk.base import content_hash
from seeley_rag.exceptions import ConfigurationError
from seeley_rag.index.embed_cache import EmbeddingCache, cache_key, namespace

VECTOR = [0.1, 0.2, 0.3]


@pytest.fixture
def cache(tmp_path: Path) -> EmbeddingCache:
    """A cache rooted in a temporary directory.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        An empty cache.
    """
    return EmbeddingCache(root=tmp_path / "embeddings")


class TestKeying:
    """The key must be the hash of the text that is actually embedded."""

    def test_key_matches_the_chunk_content_hash(self) -> None:
        """Stage 3 already computed this; the cache must agree with it.

        If these ever diverge, every chunk misses on its first lookup and the
        cache silently does nothing.
        """
        text = "Ducted Gas Heating > Service Guides > TQ\n\nCheck the flame sensor."
        assert cache_key(text) == content_hash(text)

    def test_namespace_separates_models_and_widths(self) -> None:
        """Vectors of different width or model are not interchangeable."""
        assert namespace("text-embedding-3-large", 3072) == "text-embedding-3-large-3072"
        assert namespace("text-embedding-3-large", 1024) != namespace(
            "text-embedding-3-large", 3072
        )
        assert namespace("text-embedding-3-small", 3072) != namespace(
            "text-embedding-3-large", 3072
        )


class TestRoundTrip:
    """Store and retrieve."""

    def test_miss_on_an_empty_cache(self, cache: EmbeddingCache) -> None:
        """Nothing stored, nothing returned."""
        assert cache.get("a" * 64) is None
        assert cache.misses == 1

    def test_put_then_get(self, cache: EmbeddingCache) -> None:
        """The basic contract."""
        key = "a" * 64
        cache.put(key, VECTOR)
        assert cache.get(key) == VECTOR
        assert cache.hits == 1

    def test_vectors_survive_a_flush_and_reopen(self, tmp_path: Path) -> None:
        """The whole point: the cache outlives the process that filled it."""
        root = tmp_path / "embeddings"
        key = "b" * 64
        with EmbeddingCache(root=root) as first:
            first.put(key, VECTOR)
        assert EmbeddingCache(root=root).get(key) == VECTOR

    def test_cache_outlives_a_deleted_index(self, tmp_path: Path) -> None:
        """Rebuilding the store from scratch must not cost API calls.

        The cache is keyed by text, not by anything the store knows, so
        dropping the table and rebuilding is free.
        """
        root = tmp_path / "embeddings"
        with EmbeddingCache(root=root) as cache:
            cache.put(content_hash("chunk text"), VECTOR)
        rebuilt = EmbeddingCache(root=root)
        assert rebuilt.get(content_hash("chunk text")) == VECTOR
        assert rebuilt.misses == 0

    def test_get_many_returns_only_hits(self, cache: EmbeddingCache) -> None:
        """Callers need to know what is left to embed."""
        cache.put("a" * 64, VECTOR)
        found = cache.get_many(["a" * 64, "c" * 64])
        assert found == {"a" * 64: VECTOR}

    def test_put_many(self, cache: EmbeddingCache) -> None:
        """Batch writes are the normal path."""
        cache.put_many({"a" * 64: VECTOR, "b" * 64: [0.4]})
        assert cache.get("a" * 64) == VECTOR
        assert cache.get("b" * 64) == [0.4]


class TestSharding:
    """One file per two-character prefix."""

    def test_keys_land_in_prefix_shards(self, cache: EmbeddingCache) -> None:
        """256 shards over ~16k chunks keeps each rewrite small."""
        cache.put("ab" + "0" * 62, VECTOR)
        cache.flush()
        assert (cache.root / "ab.json").exists()

    def test_different_prefixes_use_different_files(self, cache: EmbeddingCache) -> None:
        """Sharding must actually distribute."""
        cache.put("ab" + "0" * 62, VECTOR)
        cache.put("cd" + "0" * 62, VECTOR)
        cache.flush()
        assert {p.name for p in cache.root.glob("*.json")} == {"ab.json", "cd.json"}

    def test_a_short_key_is_rejected(self, cache: EmbeddingCache) -> None:
        """Keys must be digests; a short one signals a caller bug."""
        with pytest.raises(ConfigurationError, match="SHA-256"):
            cache.put("a", VECTOR)

    def test_flush_reports_shards_written(self, cache: EmbeddingCache) -> None:
        """Used by the run summary."""
        cache.put("ab" + "0" * 62, VECTOR)
        assert cache.flush() == 1
        assert cache.flush() == 0, "a clean cache must not rewrite anything"


class TestResilience:
    """The cache is an optimisation; a bad shard must cost a call, not the run."""

    def test_corrupt_shard_is_treated_as_empty(self, tmp_path: Path) -> None:
        """A truncated shard must not raise mid-run."""
        root = tmp_path / "embeddings"
        root.mkdir(parents=True)
        (root / "ab.json").write_text('{"ab000": [0.1,', encoding="utf-8")
        assert EmbeddingCache(root=root).get("ab" + "0" * 62) is None

    def test_non_object_shard_is_treated_as_empty(self, tmp_path: Path) -> None:
        """Valid JSON of the wrong shape must not raise either."""
        root = tmp_path / "embeddings"
        root.mkdir(parents=True)
        (root / "ab.json").write_text("[1, 2, 3]", encoding="utf-8")
        assert EmbeddingCache(root=root).get("ab" + "0" * 62) is None

    def test_flush_is_atomic(self, cache: EmbeddingCache) -> None:
        """Write-then-replace, so a kill cannot leave a truncated shard.

        A truncated shard would read as a silent miss for every key it holds --
        the most expensive possible failure, since it is invisible.
        """
        cache.put("ab" + "0" * 62, VECTOR)
        cache.flush()
        assert not list(cache.root.glob("*.tmp")), "temporary file left behind"
        assert json.loads((cache.root / "ab.json").read_text(encoding="utf-8"))

    def test_context_manager_flushes_after_an_exception(self, tmp_path: Path) -> None:
        """Work already paid for must not be lost to a later failure."""
        root = tmp_path / "embeddings"
        key = "ab" + "0" * 62
        with pytest.raises(RuntimeError):
            with EmbeddingCache(root=root) as cache:
                cache.put(key, VECTOR)
                raise RuntimeError("interrupted")
        assert EmbeddingCache(root=root).get(key) == VECTOR


class TestStats:
    """Reported in the run summary, so they have to be right."""

    def test_hit_rate_is_zero_before_any_lookup(self, cache: EmbeddingCache) -> None:
        """No division by zero on a fresh cache."""
        assert cache.hit_rate == 0.0

    def test_hit_rate_counts_both_outcomes(self, cache: EmbeddingCache) -> None:
        """One hit and one miss is 50%."""
        cache.put("a" * 64, VECTOR)
        cache.get("a" * 64)
        cache.get("z" * 64)
        assert cache.hit_rate == 0.5

    def test_stats_names_the_cache_location(self, cache: EmbeddingCache) -> None:
        """So a run's log says which namespace it was reading."""
        assert str(cache.root) in str(cache.stats()["root"])
