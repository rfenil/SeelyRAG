"""Token counting for chunk sizing.

build-plan.md section 5.1.

Chunk sizes are not stylistic. ``text-embedding-3-large`` rejects inputs above
8,191 tokens outright, so a table chunk measured wrongly does not degrade -- it
fails, on exactly the fault-code content this system exists to serve. That makes
the count worth getting right rather than estimating.

The real count comes from ``tiktoken`` with the encoding the embedding model
actually uses. ``tiktoken`` downloads its BPE table on first use and caches it,
which is fine for a pipeline run and unacceptable inside the test suite -- no
test may make a real network request. So the encoder is loaded lazily, once, and
any failure to obtain it falls back to a character-ratio estimate.

The fallback is deliberately **conservative**: it divides by 3.4 rather than the
customary 4.0, so it over-counts on ordinary prose. Over-counting yields chunks
slightly smaller than the target; under-counting yields chunks that exceed a
hard API limit. Only one of those two errors is recoverable.
"""

from __future__ import annotations

import functools
from typing import Any, Protocol

from seeley_rag.logging_conf import get_logger
from seeley_rag.settings import get_settings

log = get_logger(__name__)

#: Characters per token assumed when ``tiktoken`` is unavailable. Lower than the
#: usual 4.0 on purpose -- see the module docstring.
FALLBACK_CHARS_PER_TOKEN = 3.4

#: Encoding used by both ``text-embedding-3-large`` and ``text-embedding-3-small``.
EMBEDDING_ENCODING = "cl100k_base"


class _Encoder(Protocol):
    """The single method this module needs from a tokeniser."""

    def encode(self, text: str) -> list[int]:
        """Return token ids for ``text``."""
        ...


@functools.lru_cache(maxsize=1)
def get_encoder() -> _Encoder | None:
    """Return the tiktoken encoder for the configured embedding model.

    Cached: building an encoder parses a multi-megabyte BPE table, and chunking
    calls this once per page.

    Returns:
        An encoder, or ``None`` when tiktoken is not installed or its BPE table
        cannot be obtained (no network and no cache). Callers fall back to
        :data:`FALLBACK_CHARS_PER_TOKEN`.
    """
    try:
        import tiktoken
    except ImportError:
        log.warning("tiktoken_unavailable", extra={"reason": "not installed", "fallback": "chars"})
        return None

    model = get_settings().index.embedding_model
    try:
        return tiktoken.encoding_for_model(model)
    except (KeyError, ValueError):
        # An unrecognised model name is not fatal: every current OpenAI
        # embedding model uses cl100k_base, so name it directly.
        pass
    except Exception as exc:  # noqa: BLE001 - tiktoken raises bare network errors
        log.warning("tiktoken_unavailable", extra={"reason": str(exc), "fallback": "chars"})
        return None

    try:
        return tiktoken.get_encoding(EMBEDDING_ENCODING)
    except Exception as exc:  # noqa: BLE001 - download failure has no typed class
        log.warning("tiktoken_unavailable", extra={"reason": str(exc), "fallback": "chars"})
        return None


def count_tokens(text: str) -> int:
    """Count tokens in ``text`` as the embedding model will.

    Args:
        text: The text to measure.

    Returns:
        Token count. Exact when tiktoken is available, a conservative
        over-estimate otherwise.
    """
    if not text:
        return 0
    encoder = get_encoder()
    if encoder is None:
        return estimate_tokens(text)
    return len(encoder.encode(text))


def estimate_tokens(text: str) -> int:
    """Estimate tokens from character count, without tiktoken.

    Args:
        text: The text to measure.

    Returns:
        A deliberately conservative (high) token estimate.
    """
    if not text:
        return 0
    return max(1, int(len(text) / FALLBACK_CHARS_PER_TOKEN) + 1)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Hard-truncate ``text`` so it cannot exceed ``max_tokens``.

    The last line of defence before an embedding call. Splitting on semantic
    boundaries is the chunker's job; this exists so a pathological input -- an
    unbroken 40,000-character table cell, say -- cannot reach the API and 400.

    Args:
        text: Text to truncate.
        max_tokens: Hard ceiling.

    Returns:
        ``text`` unchanged when it already fits, else a prefix that fits.
    """
    if max_tokens <= 0:
        return ""
    encoder = get_encoder()
    if encoder is None:
        limit = int(max_tokens * FALLBACK_CHARS_PER_TOKEN)
        return text if len(text) <= limit else text[:limit]

    tokens: Any = encoder.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoder.decode(tokens[:max_tokens])  # type: ignore[attr-defined]
