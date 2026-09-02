"""Stage 4 -- the OpenAI embedding client.

build-plan.md section 6.

Wraps ``text-embedding-3-large`` with the three things a 16,000-chunk run needs
and a bare API call does not: the cache in front of it, batching, and retries
that distinguish "try again" from "stop".

Separated from :mod:`seeley_rag.index.build` so the store-building logic can be
tested without an API client at all, and so a later swap of embedding provider
touches one file.

⚠ **Two hard limits, both of which fail the request rather than degrading it.**
``text-embedding-3-large`` rejects any single input above 8,191 tokens, and
rejects a batch whose *total* exceeds roughly 300,000 tokens. Stage 3 already
caps individual chunks well below the first, but a batch of 256 table chunks
near their 6,000-token ceiling would breach the second, so batches are packed by
token budget rather than by a fixed count.
"""

from __future__ import annotations

import time
from typing import Any, Iterable, Sequence

from seeley_rag.chunk.tokens import count_tokens, truncate_to_tokens
from seeley_rag.exceptions import ConfigurationError, SeeleyRagError
from seeley_rag.index.embed_cache import EmbeddingCache
from seeley_rag.logging_conf import get_logger
from seeley_rag.settings import get_settings

log = get_logger(__name__)

#: Hard per-input cap for text-embedding-3-*. Exceeding it is a 400, not a
#: truncation, so the client enforces it before the request leaves.
MAX_INPUT_TOKENS = 8191

#: Conservative ceiling on a batch's combined tokens. The documented limit is
#: near 300k; 250k leaves room for the tokeniser disagreeing with theirs at the
#: margin, which costs one extra request and avoids a failed one.
MAX_BATCH_TOKENS = 250_000

#: Attempts per batch before giving up.
MAX_ATTEMPTS = 5

#: Base seconds for exponential backoff between attempts.
BACKOFF_BASE_SECONDS = 2.0


class EmbeddingError(SeeleyRagError):
    """An embedding request failed in a way retrying will not fix."""


def _is_retryable(exc: Exception) -> bool:
    """Whether an exception is worth another attempt.

    Rate limits, timeouts and 5xx are transient. An authentication failure or a
    malformed request is not, and retrying it just burns the run's time before
    failing anyway.

    Args:
        exc: The raised exception.

    Returns:
        True when a retry could plausibly succeed.
    """
    name = type(exc).__name__
    if name in {"AuthenticationError", "PermissionDeniedError", "BadRequestError"}:
        return False
    if name in {"RateLimitError", "APITimeoutError", "APIConnectionError", "InternalServerError"}:
        return True
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status == 429 or status >= 500
    return False


class Embedder:
    """Embeds chunk texts, consulting the cache first.

    Args:
        client: An OpenAI-compatible client exposing ``embeddings.create``.
            Injected so tests never construct a real one -- no test in this
            project may touch the network.
        cache: Embedding cache. Defaults to one for the configured model.
        model: Embedding model. Defaults to configured.
        dimensions: Vector width. Defaults to configured.

    Attributes:
        requests: API calls made.
        embedded: Texts sent to the API.
        cached: Texts served from cache.
    """

    def __init__(
        self,
        client: Any | None = None,
        cache: EmbeddingCache | None = None,
        model: str | None = None,
        dimensions: int | None = None,
    ) -> None:
        settings = get_settings()
        self.model = model or settings.index.embedding_model
        self.dimensions = dimensions or settings.index.embedding_dim
        self.cache = (
            cache
            if cache is not None
            else EmbeddingCache(model=self.model, dimensions=self.dimensions)
        )
        self._client = client
        self.requests = 0
        self.embedded = 0
        self.cached = 0

    @property
    def client(self) -> Any:
        """The OpenAI client, constructed on first use.

        Deferred so that importing this module, or building an
        :class:`Embedder` with a cache that satisfies every lookup, needs no API
        key at all.

        Returns:
            The client.

        Raises:
            ConfigurationError: If no key is configured or the SDK is missing.
        """
        if self._client is None:
            api_key = get_settings().openai_api_key
            if not api_key:
                raise ConfigurationError(
                    "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in; "
                    "Stage 4 cannot embed without it."
                )
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ConfigurationError(
                    'The openai package is not installed. Run: pip install -e ".[downstream]"'
                ) from exc
            self._client = OpenAI(api_key=api_key)
        return self._client

    # -- batching ----------------------------------------------------------

    def _prepare(self, text: str) -> str:
        """Make one text safe to send.

        Args:
            text: Chunk text.

        Returns:
            The text, truncated if it would breach the per-input cap.
        """
        if count_tokens(text) <= MAX_INPUT_TOKENS:
            return text
        log.warning("chunk_truncated_for_embedding", extra={"limit": MAX_INPUT_TOKENS})
        return truncate_to_tokens(text, MAX_INPUT_TOKENS)

    def batches(self, texts: Sequence[str], batch_size: int | None = None) -> list[list[int]]:
        """Group text indices into batches that fit both API limits.

        Packs by token budget as well as count, because 256 table chunks near
        their 6,000-token ceiling would exceed the per-request total even though
        the count is fine.

        Args:
            texts: Texts to be embedded.
            batch_size: Maximum items per batch. Defaults to configured.

        Returns:
            Batches of indices into ``texts``.
        """
        limit = batch_size or get_settings().index.batch_size
        groups: list[list[int]] = []
        current: list[int] = []
        current_tokens = 0

        for index, text in enumerate(texts):
            tokens = count_tokens(text)
            too_many = len(current) >= limit
            too_big = current and current_tokens + tokens > MAX_BATCH_TOKENS
            if too_many or too_big:
                groups.append(current)
                current, current_tokens = [], 0
            current.append(index)
            current_tokens += tokens

        if current:
            groups.append(current)
        return groups

    # -- the API call ------------------------------------------------------

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Send one batch, retrying transient failures.

        Args:
            texts: Texts to embed.

        Returns:
            One vector per text, in order.

        Raises:
            EmbeddingError: If the batch fails and retrying will not help, or
                every attempt is exhausted.
        """
        payload: dict[str, Any] = {"model": self.model, "input": texts}
        # Only send `dimensions` when it differs from the model's native width;
        # sending it always would fail against models that do not support the
        # parameter at all.
        if self.dimensions and self.dimensions != 3072:
            payload["dimensions"] = self.dimensions

        last: Exception | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = self.client.embeddings.create(**payload)
                self.requests += 1
                # The API documents order preservation, but the index field is
                # authoritative and cheap to honour. A silently reordered batch
                # would attach every vector to the wrong chunk.
                items = sorted(response.data, key=lambda item: item.index)
                return [list(item.embedding) for item in items]
            except Exception as exc:  # noqa: BLE001 - SDK raises many types
                last = exc
                if not _is_retryable(exc):
                    raise EmbeddingError(
                        f"Embedding request failed and will not be retried: {exc}"
                    ) from exc
                if attempt == MAX_ATTEMPTS:
                    break
                delay = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                log.warning(
                    "embedding_retry",
                    extra={"attempt": attempt, "delay": delay, "error": str(exc)},
                )
                time.sleep(delay)

        raise EmbeddingError(f"Embedding failed after {MAX_ATTEMPTS} attempts: {last}")

    # -- public surface ----------------------------------------------------

    def embed_texts(
        self,
        texts: Sequence[str],
        keys: Sequence[str] | None = None,
        batch_size: int | None = None,
        on_progress: Any | None = None,
    ) -> list[list[float]]:
        """Embed texts, serving what the cache already holds.

        Args:
            texts: Texts to embed.
            keys: Cache key per text. Defaults to hashing each text, but callers
                holding chunks should pass ``content_hash`` -- Stage 3 already
                computed it over exactly this string.
            batch_size: Items per API request. Defaults to configured.
            on_progress: Optional callable taking ``(done, total)`` after each
                batch.

        Returns:
            One vector per input text, in input order.

        Raises:
            EmbeddingError: If a batch cannot be embedded.
        """
        from seeley_rag.chunk.base import content_hash

        resolved_keys = list(keys) if keys is not None else [content_hash(t) for t in texts]
        if len(resolved_keys) != len(texts):
            raise EmbeddingError(
                f"Got {len(resolved_keys)} cache keys for {len(texts)} texts; they must correspond."
            )

        vectors: list[list[float] | None] = [None] * len(texts)

        # Pass 1: the cache. Duplicate texts across the corpus -- the same
        # safety notice on 900 pages -- collapse to one key and are answered
        # once, before any request is made.
        pending: list[int] = []
        for index, key in enumerate(resolved_keys):
            hit = self.cache.get(key)
            if hit is None:
                pending.append(index)
            else:
                vectors[index] = hit
                self.cached += 1

        # Pass 2: embed what is left, deduplicated by key so a repeated chunk
        # is paid for once even within a single run.
        unique: dict[str, int] = {}
        for index in pending:
            unique.setdefault(resolved_keys[index], index)
        unique_indices = list(unique.values())
        unique_texts = [self._prepare(texts[i]) for i in unique_indices]

        done = 0
        for group in self.batches(unique_texts, batch_size):
            embedded = self._embed_batch([unique_texts[i] for i in group])
            fresh: dict[str, list[float]] = {}
            for position, vector in zip(group, embedded):
                key = resolved_keys[unique_indices[position]]
                fresh[key] = vector
            self.cache.put_many(fresh)
            # Flush per batch: a long run that is interrupted must keep the
            # work already paid for.
            self.cache.flush()
            self.embedded += len(group)
            done += len(group)
            if on_progress is not None:
                on_progress(done, len(unique_texts))

        # Pass 3: fan the deduplicated vectors back out to every position.
        for index in pending:
            if vectors[index] is None:
                vectors[index] = self.cache.get(resolved_keys[index])

        missing = [i for i, v in enumerate(vectors) if v is None]
        if missing:
            raise EmbeddingError(
                f"{len(missing)} texts have no vector after embedding; first at index {missing[0]}."
            )
        return [v for v in vectors if v is not None]

    def embed_chunks(
        self,
        chunks: Iterable[Any],
        batch_size: int | None = None,
        on_progress: Any | None = None,
    ) -> list[list[float]]:
        """Embed chunk records, keyed by their existing content hashes.

        Args:
            chunks: :class:`~seeley_rag.chunk.base.Chunk` records.
            batch_size: Items per API request.
            on_progress: Optional ``(done, total)`` callable.

        Returns:
            One vector per chunk, in order.
        """
        materialised = list(chunks)
        return self.embed_texts(
            [c.text for c in materialised],
            keys=[c.content_hash for c in materialised],
            batch_size=batch_size,
            on_progress=on_progress,
        )

    def stats(self) -> dict[str, Any]:
        """Return a summary for logging.

        Returns:
            Request, cache and embedding counters.
        """
        return {
            "model": self.model,
            "dimensions": self.dimensions,
            "requests": self.requests,
            "embedded": self.embedded,
            "cached": self.cached,
            **self.cache.stats(),
        }
