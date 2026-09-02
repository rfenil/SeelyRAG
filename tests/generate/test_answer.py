"""Stage 6 answer synthesis.

build-plan section 8. No test here calls a model: the point is the code that
runs *after* the model has spoken.

A system prompt is a request, not a guarantee, and this one guards answers about
gas carriage and mains electrical work. Everything asserted below is a rule the
prompt asks for and cannot enforce, so it is enforced here instead.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from seeley_rag.generate.answer import (
    answer,
    assemble,
    build_citation,
    cited_numbers,
    log_query,
    new_query_id,
    normalise_sections,
    plain_ascii,
    strip_inline_citations,
)


def chunk(chunk_id: str, **overrides: Any) -> dict[str, Any]:
    """Build a retrieved chunk row.

    Args:
        chunk_id: Identifier.
        **overrides: Fields to replace.

    Returns:
        The row.
    """
    base: dict[str, Any] = {
        "chunk_id": chunk_id,
        "text": f"Breadcrumb > Path\n\nBody text of {chunk_id}.",
        "title": f"Manual {chunk_id}",
        "page_label": "42",
        "page_image": f"images/{chunk_id}.png",
        "source_url": f"https://example.invalid/doc/{chunk_id}",
        "article_url": f"https://example.invalid/article/{chunk_id}",
        "product_family": "DGH",
        "kind": "prose",
    }
    base.update(overrides)
    return base


def payload(**overrides: Any) -> dict[str, Any]:
    """Build a model output payload.

    Args:
        **overrides: Fields to replace.

    Returns:
        The payload.
    """
    base: dict[str, Any] = {"answer": "The answer [1].", "confidence": "high", "answered": True}
    base.update(overrides)
    return base


class FakeClient:
    """An OpenAI-shaped client returning a fixed payload."""

    def __init__(self, content: str) -> None:
        self.calls: list[dict[str, Any]] = []
        outer = self

        class Completions:
            @staticmethod
            def create(**kwargs: Any) -> Any:
                outer.calls.append(kwargs)
                message = type("Message", (), {"content": content})()
                return type(
                    "Response", (), {"choices": [type("Choice", (), {"message": message})()]}
                )()

        self.chat = type("Chat", (), {"completions": Completions()})()


class TestQueryId:
    """`/feedback` takes one of these, so it must exist before the answer does."""

    def test_ids_are_unique(self) -> None:
        """Two questions must never collide in the query log."""
        assert len({new_query_id() for _ in range(200)}) == 200

    def test_ids_are_prefixed(self) -> None:
        """Recognisable in a log line without a schema."""
        assert new_query_id().startswith("q_")


class TestCitationParsing:
    """Markers in prose are the only link between an answer and its evidence."""

    def test_single_markers(self) -> None:
        """The common form."""
        assert cited_numbers("Check the sensor [1]. Then the valve [3].") == [1, 3]

    def test_grouped_markers(self) -> None:
        """Models write [1, 2] when two passages agree."""
        assert cited_numbers("Both say so [1, 2].") == [1, 2]

    def test_adjacent_markers(self) -> None:
        """And sometimes [1][2]."""
        assert cited_numbers("Both say so [1][2].") == [1, 2]

    def test_repeats_are_collapsed(self) -> None:
        """One source cited five times is still one source."""
        assert cited_numbers("[1] a [1] b [1]") == [1]

    def test_no_markers(self) -> None:
        """An uncited answer parses to nothing, and is marked down elsewhere."""
        assert cited_numbers("No citations at all.") == []


def test_plain_ascii_normalises_model_punctuation() -> None:
    """UI answers should copy as plain field notes."""
    assert plain_ascii("4-6\u202fmm - non\u2011condensing \u201cprobe\u201d \u2022 ok") == (
        '4-6 mm - non-condensing "probe" - ok'
    )


def test_strip_inline_citations_leaves_readable_answer_text() -> None:
    """Citation cards stay, but markers should not clutter the answer body."""
    assert strip_inline_citations("Check flame [1]. Then ignition [2, 3].") == (
        "Check flame. Then ignition."
    )


class TestGuarantees:
    """The rules the prompt asks for and cannot enforce."""

    def test_only_cited_passages_become_citations(self) -> None:
        """Listing all eight sources under a two-source answer implies
        corroboration that does not exist."""
        chunks = [chunk(f"c{i}") for i in range(8)]
        response = assemble("q_1", payload(answer="A [1] and B [3]."), chunks, "DGH", 10)
        assert [c.n for c in response.citations] == [1, 3]

    def test_out_of_range_markers_are_removed_from_the_prose(self) -> None:
        """A [9] with eight passages resolves to nothing.

        Left in place it looks verified, which is worse than no citation.
        """
        chunks = [chunk("c1")]
        response = assemble("q_1", payload(answer="Real [1]. Invented [9]."), chunks, "DGH", 10)
        assert "[9]" not in response.answer
        assert "[1]" not in response.answer
        assert response.answer == "Real. Invented."
        assert [c.n for c in response.citations] == [1]

    def test_a_partly_invalid_group_keeps_the_valid_part(self) -> None:
        """[1, 9] must not lose the legitimate citation with the bogus one."""
        chunks = [chunk("c1")]
        response = assemble("q_1", payload(answer="Both [1, 9]."), chunks, "DGH", 10)
        assert "[1]" not in response.answer
        assert response.answer == "Both."
        assert [c.n for c in response.citations] == [1]

    def test_an_uncited_answer_is_marked_low_confidence(self) -> None:
        """Citations are the one property that makes this system trustworthy.

        An answer without them is not presented as confident, whatever the
        model claimed.
        """
        response = assemble(
            "q_1", payload(answer="Just trust me.", confidence="high"), [chunk("c1")], "DGH", 10
        )
        assert response.confidence == "low"
        assert response.citations == []

    def test_a_hallucinated_citation_downgrades_high_confidence(self) -> None:
        """The answer was less grounded than it claimed."""
        response = assemble(
            "q_1",
            payload(answer="Real [1]. Invented [9].", confidence="high"),
            [chunk("c1")],
            "DGH",
            10,
        )
        assert response.confidence == "low"

    def test_citations_are_sorted_by_marker(self) -> None:
        """The reader scans the list for the number they just read."""
        chunks = [chunk(f"c{i}") for i in range(5)]
        response = assemble("q_1", payload(answer="B [3] then A [1]."), chunks, "DGH", 10)
        assert [c.n for c in response.citations] == [1, 3]

    def test_an_unknown_confidence_value_is_normalised(self) -> None:
        """A model inventing "very high" must not reach the API response."""
        response = assemble("q_1", payload(confidence="extremely high"), [chunk("c1")], "DGH", 10)
        assert response.confidence == "unknown"


class TestSectionLayout:
    """The answer layout is a contract with the field UI, so it is enforced.

    The prompt asks for "Answer:" as the first line. Models reliably write the
    answering sentence first and emit the heading above the bullet list instead,
    which leaves the reader with an unlabelled paragraph followed by a section
    called "Answer" that is not one. Four rounds of rewording did not fix it.
    """

    def test_a_misplaced_heading_moves_to_the_top(self) -> None:
        """Observed on "can i put dual spark electrode on a single probe unit"."""
        text = (
            "Yes. Seeley supplies retrofit kits [3]\n"
            "\n"
            "Answer:\n"
            "- 651799 - condensing kit [3]\n"
            "\n"
            "What to check:\n"
            "1. Confirm the PCB [3]"
        )
        out = normalise_sections(text)
        assert out.startswith("Answer:\n")
        assert out.count("Answer:") == 1
        assert "651799" in out
        assert out.index("Yes. Seeley") < out.index("651799")

    def test_a_correct_layout_is_left_alone(self) -> None:
        """The normaliser must not churn an answer that already complies."""
        text = "Answer:\nYes [3]\n\nWhat to check:\n1. Thing [3]"
        assert normalise_sections(text) == text

    def test_plain_prose_is_left_alone(self) -> None:
        """A one-fact answer or a decline carries no headings at all."""
        text = "The retrieved manuals do not cover that."
        assert normalise_sections(text) == text

    def test_only_blank_lines_above_the_heading_are_dropped(self) -> None:
        """Leading whitespace is not content to reorder around."""
        assert normalise_sections("\n\nAnswer:\nYes [1]") == "Answer:\nYes [1]"


class TestDeclining:
    """Saying "not in the manuals" is a correct answer, and section 8 requires it."""

    def test_answered_false_produces_no_citations(self) -> None:
        """A decline cites nothing, because nothing supported an answer."""
        response = assemble(
            "q_1",
            {"answer": "Not covered.", "answered": False, "missing": "the TQ Service Guide"},
            [chunk("c1")],
            "DGH",
            10,
        )
        assert response.citations == []
        assert response.confidence == "low"

    def test_markers_are_stripped_from_a_decline(self) -> None:
        """Leaving [1] in prose points at a list that is deliberately empty."""
        response = assemble(
            "q_1",
            {"answer": "The passages only cover warranty terms [1], [2].", "answered": False},
            [chunk("c1"), chunk("c2")],
            "DGH",
            10,
        )
        assert "[1]" not in response.answer
        assert "[2]" not in response.answer
        assert "warranty terms" in response.answer

    def test_an_empty_answer_becomes_a_decline(self) -> None:
        """An empty string must never be returned as an answer."""
        response = assemble("q_1", payload(answer="  "), [chunk("c1")], "DGH", 10)
        assert response.answer
        assert response.confidence == "low"

    def test_the_missing_document_is_named(self) -> None:
        """Section 8: say what would have had it."""
        response = assemble(
            "q_1",
            {"answer": "", "answered": False, "missing": "the CW-6S service manual"},
            [chunk("c1")],
            "EVAP",
            10,
        )
        assert "CW-6S service manual" in response.answer

    def test_no_retrieval_results_declines_without_calling_a_model(self) -> None:
        """Nothing to ground an answer in means nothing to ask about."""
        client = FakeClient('{"answer": "should not be called"}')
        response = answer("obscure question", chunks=[], pinned=[], client=client)
        assert response.citations == []
        assert client.calls == []


class TestCitationContent:
    """Every citation resolves to a page image and a link back to the article."""

    def test_citation_carries_verification_links(self) -> None:
        """Two taps to verify is what earns trust -- section 8."""
        citation = build_citation(1, chunk("c1", doc_id="d" * 64, page_index=41))
        assert citation.page_image == "images/c1.png"
        assert citation.page_url == f"/pages/{'d' * 64}/41.png"
        assert citation.article_url == "https://example.invalid/article/c1"
        assert citation.doc_url == "https://example.invalid/doc/c1"
        assert citation.page_label == "42"

    def test_page_range_wins_over_page_label(self) -> None:
        """A merged multi-page table must cite the span it actually covers."""
        citation = build_citation(1, chunk("c1", page_range="42-44"))
        assert citation.page_label == "42-44"

    def test_snippet_drops_the_breadcrumb(self) -> None:
        """The breadcrumb is metadata the citation already shows structurally."""
        citation = build_citation(1, chunk("c1"))
        assert "Breadcrumb" not in citation.snippet
        assert "Body text" in citation.snippet

    def test_remote_page_urls_match_object_storage_keys(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Object storage links use the zero-padded rendered image filename."""
        from seeley_rag.settings import get_settings

        monkeypatch.setenv("PAGE_IMAGE_BASE_URL", "https://assets.example.test/pages/")
        get_settings.cache_clear()
        try:
            citation = build_citation(1, chunk("c1", doc_id="article:123", page_index=7))
            assert citation.page_url == "https://assets.example.test/pages/123/0007.png"
        finally:
            get_settings.cache_clear()


class TestQueryLog:
    """build-plan section 9: the first week of real queries beats any eval."""

    def test_a_record_is_appended(self, tmp_path: Path) -> None:
        """One JSON object per line."""
        path = tmp_path / "queries.jsonl"
        log_query({"query_id": "q_1", "query": "a"}, path)
        log_query({"query_id": "q_2", "query": "b"}, path)
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert [r["query_id"] for r in rows] == ["q_1", "q_2"]

    def test_an_unwritable_log_does_not_fail_the_answer(self, tmp_path: Path) -> None:
        """Never lose an answer because a log could not be written."""
        log_query({"query_id": "q_1"}, tmp_path / "missing" / "deep" / "queries.jsonl")

    def test_answering_records_the_chunk_ids(self, tmp_path: Path) -> None:
        """The eval needs to know what was retrieved, not only what was said."""
        path = tmp_path / "queries.jsonl"
        client = FakeClient(json.dumps(payload(answer="Answer [1].")))
        response = answer(
            "a question",
            chunks=[chunk("c1"), chunk("c2")],
            client=client,
            log_path=path,
        )
        record = json.loads(path.read_text(encoding="utf-8").strip())
        assert record["chunk_ids"] == ["c1", "c2"]
        assert record["query_id"] == response.query_id
        assert record["citations"] == [1]

    def test_eval_id_is_recorded_when_supplied(self, tmp_path: Path) -> None:
        """Stage 8 joins on a stable case id, not brittle question text."""
        path = tmp_path / "queries.jsonl"
        client = FakeClient(json.dumps(payload(answer="Answer [1].")))
        answer(
            "a question",
            chunks=[chunk("c1")],
            client=client,
            log_path=path,
            eval_id="dgh-001",
        )

        record = json.loads(path.read_text(encoding="utf-8").strip())
        assert record["eval_id"] == "dgh-001"

    def test_utf8_survives_the_log(self, tmp_path: Path) -> None:
        """Windows defaults to cp1252 and would corrupt article titles."""
        path = tmp_path / "queries.jsonl"
        log_query({"query": "MDHX – VRF ⚠️"}, path)
        assert "MDHX – VRF ⚠️" in path.read_text(encoding="utf-8")


class TestGenerationFailure:
    """A model outage costs the answer, not the process."""

    def test_a_model_failure_returns_a_response(self, tmp_path: Path) -> None:
        """Callers get an AskResponse, never an exception from the SDK."""

        class Broken:
            class chat:  # noqa: N801 - mirrors the SDK's shape
                class completions:  # noqa: N801
                    @staticmethod
                    def create(**kwargs: Any) -> Any:
                        raise RuntimeError("model down")

        response = answer("q", chunks=[chunk("c1")], client=Broken(), log_path=tmp_path / "q.jsonl")
        assert response.query_id
        assert response.confidence == "low"
        assert response.citations == []

    def test_a_non_json_response_is_handled(self, tmp_path: Path) -> None:
        """The model ignoring JSON mode must not raise."""
        response = answer(
            "q",
            chunks=[chunk("c1")],
            client=FakeClient("I'm sorry, I can't do that."),
            log_path=tmp_path / "q.jsonl",
        )
        assert response.confidence == "low"


class TestEndToEnd:
    """The whole stage, with the model faked."""

    def test_a_grounded_answer_round_trips(self, tmp_path: Path) -> None:
        """Answer text, citations and metadata all reach the response."""
        client = FakeClient(
            json.dumps(
                {
                    "answer": "Check the flame sensor gap is 4-6 mm [1].",
                    "confidence": "high",
                    "answered": True,
                }
            )
        )
        response = answer(
            "TQ FC7",
            chunks=[chunk("c1"), chunk("c2")],
            product_family="DGH",
            client=client,
            log_path=tmp_path / "q.jsonl",
        )
        assert "4-6 mm" in response.answer
        assert response.confidence == "high"
        assert response.product_family == "DGH"
        assert len(response.citations) == 1
        assert response.latency_ms >= 0

    def test_the_question_reaches_the_prompt(self, tmp_path: Path) -> None:
        """Sanity check on the wiring between stages."""
        client = FakeClient(json.dumps(payload()))
        answer(
            "TQ FC7 ignition", chunks=[chunk("c1")], client=client, log_path=tmp_path / "q.jsonl"
        )
        user_message = client.calls[0]["messages"][1]["content"]
        assert "TQ FC7 ignition" in user_message
        assert "[1]" in user_message


@pytest.mark.parametrize("confidence", ["high", "medium", "low"])
def test_valid_confidences_pass_through(confidence: str) -> None:
    """The three the prompt asks for are preserved."""
    response = assemble(
        "q_1", payload(answer="A [1].", confidence=confidence), [chunk("c1")], "DGH", 1
    )
    assert response.confidence == confidence
