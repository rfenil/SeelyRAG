"""Token counting, and the fallback that keeps the suite off the network.

``tiktoken`` downloads its BPE table on first use. That is fine for a pipeline
run and unacceptable in the test suite, so every path here is exercised with the
encoder both present and absent -- the absent case being what CI actually hits.
"""

from __future__ import annotations

import pytest

from seeley_rag.chunk import tokens as tokens_module
from seeley_rag.chunk.tokens import (
    FALLBACK_CHARS_PER_TOKEN,
    count_tokens,
    estimate_tokens,
    get_encoder,
    truncate_to_tokens,
)


@pytest.fixture
def no_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the character-ratio fallback.

    Args:
        monkeypatch: pytest's patcher.
    """
    monkeypatch.setattr(tokens_module, "get_encoder", lambda: None)


class TestCounting:
    """Counting with the real encoder, when one is available."""

    def test_empty_text_is_zero(self) -> None:
        """No text, no tokens."""
        assert count_tokens("") == 0

    def test_longer_text_counts_higher(self) -> None:
        """Monotonicity is what chunk sizing actually relies on."""
        assert count_tokens("word " * 100) > count_tokens("word")

    def test_encoder_is_cached(self) -> None:
        """Building an encoder parses a multi-megabyte table; do it once."""
        get_encoder.cache_clear()
        assert get_encoder() is get_encoder()


class TestFallback:
    """The path taken when tiktoken is missing or its table cannot be fetched."""

    def test_estimate_is_conservative(self) -> None:
        """It divides by 3.4, not 4.0, so it over-counts ordinary prose.

        Over-counting yields chunks slightly under target. Under-counting
        yields chunks over a hard API limit. Only one of those is recoverable.
        """
        text = "a" * 340
        assert estimate_tokens(text) > len(text) / 4.0
        assert estimate_tokens(text) >= len(text) / FALLBACK_CHARS_PER_TOKEN

    def test_empty_text_estimates_zero(self) -> None:
        """Consistent with the real counter."""
        assert estimate_tokens("") == 0

    def test_count_falls_back_when_no_encoder(self, no_encoder: None) -> None:
        """The suite must work with no network and no cached BPE table."""
        assert count_tokens("some text here") == estimate_tokens("some text here")

    def test_truncate_falls_back_when_no_encoder(self, no_encoder: None) -> None:
        """Truncation still has to cap, encoder or not."""
        truncated = truncate_to_tokens("word " * 1000, 10)
        assert len(truncated) <= int(10 * FALLBACK_CHARS_PER_TOKEN)

    def test_short_text_is_untouched_by_fallback_truncation(self, no_encoder: None) -> None:
        """Nothing that already fits may be altered."""
        assert truncate_to_tokens("short", 100) == "short"


class TestTruncation:
    """The last line of defence before an embedding call."""

    def test_text_within_the_cap_is_unchanged(self) -> None:
        """Truncation is an exception, not a normal step."""
        assert truncate_to_tokens("a short sentence", 1000) == "a short sentence"

    def test_oversized_text_is_cut_to_fit(self) -> None:
        """A pathological input must not reach the API and 400."""
        assert count_tokens(truncate_to_tokens("word " * 5000, 50)) <= 50

    def test_zero_budget_yields_nothing(self) -> None:
        """A degenerate cap must not return the whole string."""
        assert truncate_to_tokens("anything at all", 0) == ""
