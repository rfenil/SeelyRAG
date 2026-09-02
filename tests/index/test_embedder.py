"""Stage 4 embedding client.

No test here touches the network: the OpenAI client is injected as a fake, and
``conftest.py``'s ``no_network`` fixture fails anything that tries otherwise.

What is worth testing is the behaviour around the API call rather than the call
itself -- batching against two hard limits, deduplication, retry classification,
and the ordering guarantee. Every one of those, when wrong, produces vectors
attached to the wrong chunks, which retrieval cannot detect and an eval reports
as a mysterious quality problem.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from seeley_rag.chunk.base import Chunk
from seeley_rag.index.embed_cache import EmbeddingCache
from seeley_rag.index.embedder import (
    MAX_BATCH_TOKENS,
    MAX_INPUT_TOKENS,
    Embedder,
    EmbeddingError,
    _is_retryable,
)


class FakeEmbeddings:
    """Stands in for ``client.embeddings``.

    Args:
        dimensions: Width of the vectors to return.
        fail_times: Raise ``error`` this many times before succeeding.
        error: Exception to raise.
        shuffle: Return ``data`` in reverse index order, to prove the client
            sorts on ``index`` rather than trusting arrival order.
    """

    def __init__(
        self,
        dimensions: int = 4,
        fail_times: int = 0,
        error: Exception | None = None,
        shuffle: bool = False,
    ) -> None:
        self.dimensions = dimensions
        self.fail_times = fail_times
        self.error = error or RuntimeError("boom")
        self.shuffle = shuffle
        self.calls: list[dict[str, Any]] = []

    def create(self, **payload: Any) -> Any:
        """Return one deterministic vector per input.

        Args:
            **payload: The request.

        Returns:
            An object shaped like an OpenAI embeddings response.

        Raises:
            Exception: The configured error, while ``fail_times`` remains.
        """
        self.calls.append(payload)
        if self.fail_times > 0:
            self.fail_times -= 1
            raise self.error

        texts = payload["input"]
        width = payload.get("dimensions", self.dimensions)
        items = [
            type("Item", (), {"index": i, "embedding": [float(len(t))] * width})()
            for i, t in enumerate(texts)
        ]
        if self.shuffle:
            items = list(reversed(items))
        return type("Response", (), {"data": items})()


class FakeClient:
    """An OpenAI-shaped client wrapping :class:`FakeEmbeddings`."""

    def __init__(self, **kwargs: Any) -> None:
        self.embeddings = FakeEmbeddings(**kwargs)


@pytest.fixture
def embedder(tmp_path: Path) -> Embedder:
    """An embedder with a fake client and a temporary cache.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        The embedder.
    """
    return Embedder(
        client=FakeClient(),
        cache=EmbeddingCache(root=tmp_path / "cache"),
        dimensions=4,
    )


def make_chunk(text: str, chunk_id: str = "c1") -> Chunk:
    """Build a finalised chunk.

    Args:
        text: Chunk text.
        chunk_id: Identifier.

    Returns:
        The chunk.
    """
    return Chunk(chunk_id=chunk_id, doc_id="d", text=text).finalise()


class TestBatching:
    """Two hard API limits, both of which fail rather than degrade."""

    def test_batches_respect_the_count_limit(self, embedder: Embedder) -> None:
        """The configured batch size is an upper bound."""
        groups = embedder.batches(["short text"] * 10, batch_size=3)
        assert [len(g) for g in groups] == [3, 3, 3, 1]

    def test_batches_respect_the_token_budget(self, embedder: Embedder) -> None:
        """256 table chunks near their ceiling would breach the request total.

        This is why batching is by tokens as well as by count.
        """
        big = "word " * 60_000
        groups = embedder.batches([big] * 6, batch_size=256)
        assert len(groups) > 1, "token budget was not enforced"
        for group in groups:
            from seeley_rag.chunk.tokens import count_tokens

            assert sum(count_tokens(big) for _ in group) <= MAX_BATCH_TOKENS

    def test_every_index_appears_exactly_once(self, embedder: Embedder) -> None:
        """Batching must not drop or duplicate an input."""
        groups = embedder.batches(["t"] * 25, batch_size=7)
        flat = [i for g in groups for i in g]
        assert sorted(flat) == list(range(25))

    def test_empty_input_yields_no_batches(self, embedder: Embedder) -> None:
        """Nothing to embed, nothing to send."""
        assert embedder.batches([]) == []


class TestOrdering:
    """Vectors must land on the chunks they were computed from."""

    def test_response_is_sorted_by_index(self, tmp_path: Path) -> None:
        """A silently reordered batch attaches every vector to the wrong chunk.

        The fake returns data reversed; the client must put it back.
        """
        client = FakeClient(shuffle=True)
        embedder = Embedder(
            client=client, cache=EmbeddingCache(root=tmp_path / "cache"), dimensions=4
        )
        vectors = embedder.embed_texts(["a", "bb", "ccc"])
        # The fake encodes each text's length, so the mapping is checkable.
        assert [v[0] for v in vectors] == [1.0, 2.0, 3.0]

    def test_vectors_come_back_in_input_order(self, embedder: Embedder) -> None:
        """Across several batches, not just within one."""
        texts = ["a", "bb", "ccc", "dddd", "eeeee"]
        vectors = embedder.embed_texts(texts, batch_size=2)
        assert [v[0] for v in vectors] == [1.0, 2.0, 3.0, 4.0, 5.0]


class TestCaching:
    """The cache sits in front of every call."""

    def test_second_run_makes_no_api_call(self, embedder: Embedder) -> None:
        """The property the whole incremental design depends on."""
        embedder.embed_texts(["hello world"])
        assert embedder.requests == 1
        embedder.embed_texts(["hello world"])
        assert embedder.requests == 1, "a cached text was re-embedded"
        assert embedder.cached == 1

    def test_duplicate_texts_are_embedded_once(self, embedder: Embedder) -> None:
        """The same boilerplate appears on hundreds of pages.

        Deduplication happens within a run too, not only against the cache.
        """
        embedder.embed_texts(["same text"] * 50)
        assert embedder.embedded == 1
        assert len(embedder._client.embeddings.calls[0]["input"]) == 1

    def test_duplicates_still_each_get_their_vector(self, embedder: Embedder) -> None:
        """Deduplication must not shorten the result."""
        vectors = embedder.embed_texts(["same"] * 5)
        assert len(vectors) == 5
        assert all(v == vectors[0] for v in vectors)

    def test_partial_cache_embeds_only_the_rest(self, embedder: Embedder) -> None:
        """The common case after a small change."""
        embedder.embed_texts(["first"])
        embedder.embed_texts(["first", "second"])
        assert embedder.embedded == 2, "only the new text should have been embedded"

    def test_chunks_are_keyed_by_their_existing_hash(self, embedder: Embedder) -> None:
        """Stage 3 already hashed exactly this text; do not re-hash it."""
        chunk = make_chunk("Some chunk body text.")
        embedder.embed_chunks([chunk])
        assert embedder.cache.get(chunk.content_hash) is not None


class TestRetries:
    """Retry transient failures; fail fast on the rest."""

    @pytest.mark.parametrize(
        "name", ["RateLimitError", "APITimeoutError", "APIConnectionError", "InternalServerError"]
    )
    def test_transient_errors_are_retryable(self, name: str) -> None:
        """These are worth another attempt."""
        assert _is_retryable(type(name, (Exception,), {})())

    @pytest.mark.parametrize(
        "name", ["AuthenticationError", "PermissionDeniedError", "BadRequestError"]
    )
    def test_permanent_errors_are_not_retryable(self, name: str) -> None:
        """Retrying a bad key just burns the run before failing anyway."""
        assert not _is_retryable(type(name, (Exception,), {})())

    def test_status_codes_are_classified(self) -> None:
        """A 429 or 5xx is transient; a 400 is not."""
        for status, expected in ((429, True), (500, True), (503, True), (400, False)):
            exc = Exception()
            exc.status_code = status  # type: ignore[attr-defined]
            assert _is_retryable(exc) is expected

    def test_a_transient_failure_is_retried(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One failure then success must produce vectors, not an error."""
        monkeypatch.setattr("seeley_rag.index.embedder.time.sleep", lambda _: None)
        error = type("RateLimitError", (Exception,), {})()
        client = FakeClient(fail_times=2, error=error)
        embedder = Embedder(
            client=client, cache=EmbeddingCache(root=tmp_path / "cache"), dimensions=4
        )
        assert len(embedder.embed_texts(["text"])) == 1

    def test_a_permanent_failure_raises_immediately(self, tmp_path: Path) -> None:
        """No backoff loop on an auth failure."""
        error = type("AuthenticationError", (Exception,), {})()
        client = FakeClient(fail_times=99, error=error)
        embedder = Embedder(
            client=client, cache=EmbeddingCache(root=tmp_path / "cache"), dimensions=4
        )
        with pytest.raises(EmbeddingError, match="will not be retried"):
            embedder.embed_texts(["text"])
        assert len(client.embeddings.calls) == 1

    def test_exhausted_retries_raise(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Give up eventually rather than looping forever."""
        monkeypatch.setattr("seeley_rag.index.embedder.time.sleep", lambda _: None)
        error = type("RateLimitError", (Exception,), {})()
        client = FakeClient(fail_times=99, error=error)
        embedder = Embedder(
            client=client, cache=EmbeddingCache(root=tmp_path / "cache"), dimensions=4
        )
        with pytest.raises(EmbeddingError, match="after 5 attempts"):
            embedder.embed_texts(["text"])


class TestInputLimits:
    """The per-input cap is a 400, not a truncation, so enforce it locally."""

    def test_oversized_input_is_truncated_before_sending(self, embedder: Embedder) -> None:
        """Better a shortened chunk than a failed batch."""
        from seeley_rag.chunk.tokens import count_tokens

        embedder.embed_texts(["word " * 20_000])
        sent = embedder._client.embeddings.calls[0]["input"][0]
        assert count_tokens(sent) <= MAX_INPUT_TOKENS

    def test_mismatched_keys_are_rejected(self, embedder: Embedder) -> None:
        """A key list that does not correspond would poison the cache."""
        with pytest.raises(EmbeddingError, match="must correspond"):
            embedder.embed_texts(["a", "b"], keys=["only-one-key"])


class TestDimensions:
    """The width parameter is only sent when it differs from the native one."""

    def test_native_width_omits_the_parameter(self, tmp_path: Path) -> None:
        """Sending it always would fail on models that reject it."""
        client = FakeClient(dimensions=3072)
        embedder = Embedder(
            client=client, cache=EmbeddingCache(root=tmp_path / "cache"), dimensions=3072
        )
        embedder.embed_texts(["text"])
        assert "dimensions" not in client.embeddings.calls[0]

    def test_reduced_width_sends_the_parameter(self, tmp_path: Path) -> None:
        """The 1024-d experiment has to actually request 1024."""
        client = FakeClient()
        embedder = Embedder(
            client=client, cache=EmbeddingCache(root=tmp_path / "cache"), dimensions=1024
        )
        embedder.embed_texts(["text"])
        assert client.embeddings.calls[0]["dimensions"] == 1024

    def test_widths_do_not_share_a_cache(self, tmp_path: Path) -> None:
        """A 1024-d run must not read 3072-d vectors, or the index is corrupt."""
        big = EmbeddingCache(root=tmp_path / "c" / "large-3072")
        small = EmbeddingCache(root=tmp_path / "c" / "large-1024")
        key = "a" * 64
        big.put(key, [0.0] * 3072)
        big.flush()
        assert small.get(key) is None


class TestProgress:
    """Long runs need a progress signal."""

    def test_progress_callback_reaches_the_total(self, embedder: Embedder) -> None:
        """A 16,000-chunk run without feedback looks like a hang."""
        seen: list[tuple[int, int]] = []
        embedder.embed_texts(
            [f"text {i}" for i in range(10)],
            batch_size=3,
            on_progress=lambda d, t: seen.append((d, t)),
        )
        assert seen[-1] == (10, 10)
