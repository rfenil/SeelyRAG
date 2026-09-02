"""Stage 5 reranking.

build-plan section 7.2 step 6. Three backends, preferred in this order: Cohere
``rerank-v3.5`` with a key, the plan's own listwise LLM fallback when enabled,
and an identity pass returning the boosted fusion order otherwise.

The identity backend is the one worth being careful about. It is not a
placeholder that pretends to rerank -- it returns a genuinely reasonable order
(fused across both channels, boosted on four signals) and labels itself so an
eval can attribute its numbers. Tests assert that labelling, because a silent
identity pass reported as reranking would inflate every quality measurement
taken before a real reranker arrives.

The LLM backend's tests are mostly about what it is not allowed to do: invent a
passage index, duplicate one, or silently drop a passage it did not mention.
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

from seeley_rag.retrieve.rerank import (
    MAX_DOCUMENT_CHARS,
    cohere_rerank,
    identity_rerank,
    llm_rerank,
    rerank,
    rerank_backend,
)

#: The rerank *module*. `seeley_rag.retrieve` re-exports the `rerank` function
#: under the same name, so both `import seeley_rag.retrieve.rerank as m` and
#: `from seeley_rag.retrieve import rerank as m` bind the callable instead.
rerank_module = sys.modules["seeley_rag.retrieve.rerank"]


def row(chunk_id: str, score: float, text: str = "body") -> dict[str, Any]:
    """Build a boosted candidate row.

    Args:
        chunk_id: Identifier.
        score: Its boosted score.
        text: Chunk text.

    Returns:
        The row.
    """
    return {"chunk_id": chunk_id, "boosted_score": score, "text": text}


class FakeCohere:
    """Stands in for the Cohere client.

    Args:
        order: Indices into the candidate list, best first.
        error: Raise this instead of returning.
    """

    def __init__(self, order: list[int] | None = None, error: Exception | None = None) -> None:
        self.order = order or []
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def rerank(self, **kwargs: Any) -> Any:
        """Return a Cohere-shaped response.

        Raises:
            Exception: The configured error, if any.
        """
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        results = [
            type("Result", (), {"index": index, "relevance_score": 1.0 - position * 0.1})()
            for position, index in enumerate(self.order)
        ]
        return type("Response", (), {"results": results})()


class TestIdentityBackend:
    """The no-key path."""

    def test_boosted_order_is_preserved(self) -> None:
        """The list is already fused and boosted; do not disturb it."""
        candidates = [row("a", 0.9), row("b", 0.5), row("c", 0.1)]
        assert [r["chunk_id"] for r in identity_rerank("q", candidates)] == ["a", "b", "c"]

    def test_truncates_to_top_k(self) -> None:
        """Step 6 is where the list finally shortens."""
        assert len(identity_rerank("q", [row(str(i), 1.0) for i in range(20)], top_k=5)) == 5

    def test_results_are_labelled_identity(self) -> None:
        """An unlabelled identity pass would inflate every eval number."""
        ranked = identity_rerank("q", [row("a", 0.9)])
        assert ranked[0]["rerank_backend"] == "identity"

    def test_rerank_score_carries_the_boosted_score(self) -> None:
        """Downstream reads one field whichever backend ran."""
        assert identity_rerank("q", [row("a", 0.42)])[0]["rerank_score"] == 0.42

    def test_empty_candidates(self) -> None:
        """No candidates, no results, no error."""
        assert identity_rerank("q", []) == []

    def test_input_is_not_mutated(self) -> None:
        """The caller keeps the fused list for debugging output."""
        candidates = [row("a", 0.9)]
        identity_rerank("q", candidates)
        assert "rerank_backend" not in candidates[0]


class TestCohereBackend:
    """The keyed path."""

    def test_cohere_order_is_applied(self) -> None:
        """The reranker may reorder freely; index maps back to the candidate."""
        candidates = [row("a", 0.9), row("b", 0.5), row("c", 0.1)]
        client = FakeCohere(order=[2, 0, 1])
        ranked = cohere_rerank("q", candidates, top_k=3, client=client)
        assert [r["chunk_id"] for r in ranked] == ["c", "a", "b"]

    def test_results_are_labelled_cohere(self) -> None:
        """So an eval can tell which backend produced its numbers."""
        ranked = cohere_rerank("q", [row("a", 0.9)], client=FakeCohere(order=[0]))
        assert ranked[0]["rerank_backend"] == "cohere"
        assert ranked[0]["rerank_score"] == 1.0

    def test_documents_are_truncated_before_sending(self) -> None:
        """Cohere truncates near 4k tokens; a merged table can approach it."""
        client = FakeCohere(order=[0])
        cohere_rerank("q", [row("a", 0.9, text="x" * 50_000)], client=client)
        assert len(client.calls[0]["documents"][0]) == MAX_DOCUMENT_CHARS

    def test_top_n_never_exceeds_the_candidate_count(self) -> None:
        """Asking for more than exists is an API error."""
        client = FakeCohere(order=[0])
        cohere_rerank("q", [row("a", 0.9)], top_k=8, client=client)
        assert client.calls[0]["top_n"] == 1

    def test_a_failure_falls_back_rather_than_raising(self) -> None:
        """A reranker outage should cost quality, not the answer.

        The installer still gets the fused, boosted list.
        """
        candidates = [row("a", 0.9), row("b", 0.5)]
        ranked = cohere_rerank("q", candidates, client=FakeCohere(error=RuntimeError("503")))
        assert [r["chunk_id"] for r in ranked] == ["a", "b"]
        assert ranked[0]["rerank_backend"] == "identity"

    def test_empty_candidates_make_no_call(self) -> None:
        """Nothing to rerank, nothing to spend."""
        client = FakeCohere(order=[])
        assert cohere_rerank("q", [], client=client) == []
        assert client.calls == []


class TestBackendSelection:
    """Which path runs, and how a caller finds out."""

    def test_backend_is_identity_when_nothing_is_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No key and ``use_llm_rerank`` off: nothing reranks, and it says so."""
        from seeley_rag import settings as settings_module

        monkeypatch.setattr(settings_module.get_settings().retrieve, "use_llm_rerank", False)
        assert rerank_backend() == "identity"

    def test_shipped_config_has_the_llm_backend_on(self) -> None:
        """Turning reranking on is a decision, so it is asserted rather than assumed.

        production-readiness B-4. Enabled 2026-08-31 on the movement and latency
        measured by ``scripts/09_rerank_ab.py``; accuracy still needs the SME
        question set. If this fails, someone reverted the flag -- check
        ``config/config.yaml`` before changing the test.
        """
        from seeley_rag.settings import get_settings

        assert get_settings().retrieve.use_llm_rerank is True

    def test_backend_reports_cohere_when_keyed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The script prints this so a run is attributable."""
        from seeley_rag import settings as settings_module

        resolved = settings_module.get_settings()
        monkeypatch.setattr(resolved, "cohere_api_key", "test-key")
        monkeypatch.setattr(rerank_module, "cohere_installed", lambda: True)
        assert rerank_backend() == "cohere"

    def test_backend_reports_llm_when_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The plan's own fallback, opt-in because it doubles per-query cost."""
        from seeley_rag import settings as settings_module

        monkeypatch.setattr(settings_module.get_settings().retrieve, "use_llm_rerank", True)
        assert rerank_backend() == "llm"

    def test_cohere_wins_over_llm_when_both_are_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """build-plan 7.2 prefers Cohere; the LLM path is the fallback."""
        from seeley_rag import settings as settings_module

        resolved = settings_module.get_settings()
        monkeypatch.setattr(resolved, "cohere_api_key", "test-key")
        monkeypatch.setattr(resolved.retrieve, "use_llm_rerank", True)
        monkeypatch.setattr(rerank_module, "cohere_installed", lambda: True)
        assert rerank_backend() == "cohere"

    def test_dispatch_follows_the_backend_not_the_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Configuration decides the path; the client only supplies transport.

        Injecting a client must not silently switch backends, or a test could
        exercise a path production never takes.
        """
        from seeley_rag import settings as settings_module

        monkeypatch.setattr(settings_module.get_settings(), "cohere_api_key", "test-key")
        monkeypatch.setattr(rerank_module, "cohere_installed", lambda: True)
        ranked = rerank("q", [row("a", 0.9)], client=FakeCohere(order=[0]))
        assert ranked[0]["rerank_backend"] == "cohere"

    def test_a_key_without_the_sdk_is_not_reported_as_cohere(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`cohere` lives in the `downstream` extra, so a venv can hold the key alone.

        Reporting `cohere` there would credit a ranking that never happened:
        every query falls through `cohere_rerank`'s except clause to identity.
        """
        from seeley_rag import settings as settings_module

        monkeypatch.setattr(settings_module.get_settings(), "cohere_api_key", "test-key")
        monkeypatch.setattr(rerank_module, "cohere_installed", lambda: False)
        assert rerank_backend() != "cohere"

    def test_no_client_and_no_key_uses_identity(self) -> None:
        """The default path today."""
        assert rerank("q", [row("a", 0.9)])[0]["rerank_backend"] == "identity"


class FakeLLM:
    """An OpenAI-shaped client returning a fixed ordering."""

    def __init__(self, content: str) -> None:
        self.calls: list[dict[str, Any]] = []
        outer = self

        class Completions:
            @staticmethod
            def create(**payload: Any) -> Any:
                outer.calls.append(payload)
                message = type("Message", (), {"content": content})()
                return type(
                    "Response", (), {"choices": [type("Choice", (), {"message": message})()]}
                )()

        self.chat = type("Chat", (), {"completions": Completions()})()


class TestLlmBackend:
    """The listwise fallback build-plan 7.2 step 6 describes."""

    def test_model_order_is_applied(self) -> None:
        """The point of reranking: the model may reorder freely."""
        candidates = [row("a", 0.9), row("b", 0.5), row("c", 0.1)]
        ranked = llm_rerank("q", candidates, top_k=3, client=FakeLLM('{"order": [2, 0, 1]}'))
        assert [r["chunk_id"] for r in ranked] == ["c", "a", "b"]
        assert ranked[0]["rerank_backend"] == "llm"

    def test_passage_headers_carry_the_boost_signals(self) -> None:
        """The reranker cannot weigh what it cannot see.

        It receives a title and 600 characters. Without the family and the
        diagnostic-article tag it silently undoes the boosts that put a chunk
        where it is -- measured on the query log, where it demoted a diagnostic
        article beneath training slides on "TQ heater has no flame".
        """
        client = FakeLLM('{"order": [0]}')
        llm_rerank(
            "no flame",
            [row("a", 0.9) | {"product_family": "DGH", "content_stream": "diagnostic_article"}],
            top_k=1,
            client=client,
        )
        sent = client.calls[0]["messages"][-1]["content"]
        assert "DGH" in sent
        assert "DIAGNOSTIC ARTICLE" in sent

    def test_rerank_model_is_configurable_apart_from_the_router(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reranking is judgement; the router model was chosen for latency."""
        from seeley_rag import settings as settings_module

        monkeypatch.setattr(settings_module.get_settings().retrieve, "llm_rerank_model", "gpt-4.1")
        client = FakeLLM('{"order": [0]}')
        llm_rerank("q", [row("a", 0.9)], top_k=1, client=client)
        assert client.calls[0]["model"] == "gpt-4.1"

    def test_rerank_model_falls_back_to_the_router_model(self) -> None:
        """Unset must not change behaviour: that was the state before the field."""
        from seeley_rag.settings import get_settings

        client = FakeLLM('{"order": [0]}')
        llm_rerank("q", [row("a", 0.9)], top_k=1, client=client)
        assert client.calls[0]["model"] == get_settings().generate.router_model

    def test_hallucinated_indices_are_dropped(self) -> None:
        """A model naming passage 99 must not crash or invent a result."""
        ranked = llm_rerank(
            "q", [row("a", 0.9), row("b", 0.5)], top_k=2, client=FakeLLM('{"order": [99, 1, 0]}')
        )
        assert [r["chunk_id"] for r in ranked] == ["b", "a"]

    def test_repeated_indices_are_ignored(self) -> None:
        """Duplication must not duplicate a chunk into the answer's context."""
        ranked = llm_rerank(
            "q", [row("a", 0.9), row("b", 0.5)], top_k=2, client=FakeLLM('{"order": [0, 0, 1]}')
        )
        assert [r["chunk_id"] for r in ranked] == ["a", "b"]

    def test_omitted_passages_are_appended_not_lost(self) -> None:
        """Silence is not evidence of irrelevance.

        A passage the model simply did not mention keeps its fused rank rather
        than disappearing from consideration.
        """
        candidates = [row("a", 0.9), row("b", 0.5), row("c", 0.1)]
        ranked = llm_rerank("q", candidates, top_k=3, client=FakeLLM('{"order": [2]}'))
        assert [r["chunk_id"] for r in ranked] == ["c", "a", "b"]

    def test_a_malformed_response_falls_back(self) -> None:
        """Quality degrades; the answer does not disappear."""
        ranked = llm_rerank("q", [row("a", 0.9)], client=FakeLLM("not json"))
        assert ranked[0]["rerank_backend"] == "identity"

    def test_a_non_list_order_falls_back(self) -> None:
        """Valid JSON of the wrong shape is still unusable."""
        ranked = llm_rerank("q", [row("a", 0.9)], client=FakeLLM('{"order": "best first"}'))
        assert ranked[0]["rerank_backend"] == "identity"

    def test_the_candidate_pool_is_bounded(self) -> None:
        """Beyond the cap the prompt grows faster than the ranking improves."""
        from seeley_rag import settings as settings_module

        cap = settings_module.get_settings().retrieve.llm_rerank_candidates
        client = FakeLLM('{"order": [0]}')
        llm_rerank("q", [row(str(i), 1.0) for i in range(cap + 15)], top_k=5, client=client)
        prompt = client.calls[0]["messages"][1]["content"]
        assert f"[{cap - 1}]" in prompt
        assert f"[{cap}]" not in prompt

    def test_empty_candidates_make_no_call(self) -> None:
        """Nothing to rerank, nothing to spend."""
        client = FakeLLM('{"order": []}')
        assert llm_rerank("q", [], client=client) == []
        assert client.calls == []
