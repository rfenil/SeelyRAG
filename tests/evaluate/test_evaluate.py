"""Stage 8 evaluation scoring."""

from __future__ import annotations

from pathlib import Path

from seeley_rag.evaluate import (
    EvalQuestion,
    QueryRecord,
    answer_matches_constraints,
    evaluate_records,
    label_is_citable,
    page_accuracy,
    source_matches,
    write_html_report,
)


def question(**overrides: object) -> EvalQuestion:
    """Build an SME question."""
    base = {
        "id": "dgh-001",
        "question": "TQ heater showing FC7, what do I check?",
        "category": "fault_diagnosis",
        "product_family": "DGH",
        "expected_source": "644066-M MANUAL SERVICE TQ SERIES.pdf",
        "expected_page": 42,
        "must_mention": ["flame", "ignition"],
        "must_not_say": ["replace the gas valve first"],
    }
    base.update(overrides)
    return EvalQuestion.model_validate(base)


def record(**overrides: object) -> QueryRecord:
    """Build a query log row."""
    base = {
        "query_id": "q_1",
        "query": "TQ heater showing FC7, what do I check?",
        "chunk_ids": ["c1", "c2"],
        "answer": "Check flame sensing during ignition [1].",
        "citations": [1],
        "confidence": "high",
        "latency_ms": 1200,
    }
    base.update(overrides)
    return QueryRecord.model_validate(base)


def chunk(**overrides: object) -> dict[str, object]:
    """Build a chunk row."""
    base: dict[str, object] = {
        "chunk_id": "c1",
        "doc_id": "doc1",
        "title": "TQ Service Guide Gas Ducted Heater 644066 M",
        "page_label": "42",
        "label_source": "embedded",
        "page_image": "images/doc1/0041.png",
    }
    base.update(overrides)
    return base


def test_guessed_labels_are_not_citable() -> None:
    """The Stage 2 label ceiling must not become a retrieval failure."""
    assert label_is_citable(chunk(label_source="embedded"))
    assert not label_is_citable(chunk(label_source="index"))
    assert not label_is_citable(chunk(label_source="none", page_label=None))


def test_page_accuracy_skips_when_only_guessed_labels_are_cited() -> None:
    """A guessed printed page is parsing uncertainty, not page-accuracy evidence."""
    hit, reason = page_accuracy(42, [chunk(label_source="index", page_label="42")])

    assert hit is None
    assert reason


def test_page_accuracy_accepts_plus_or_minus_one_on_citable_labels() -> None:
    """Section 10 allows expected_page +/- 1."""
    hit, reason = page_accuracy(42, [chunk(page_label="41", label_source="text")])

    assert hit is True
    assert reason is None


def test_expected_source_can_match_manifest_filename() -> None:
    """Chunk titles are article titles, so filenames come from the manifest."""
    assert source_matches(
        "644066-M MANUAL SERVICE TQ SERIES.pdf",
        chunk(title="TQ Service Guide Gas Ducted Heater"),
        {"doc1": ["644066-M MANUAL SERVICE TQ SERIES.pdf"]},
    )


def test_answer_constraints_check_mentions_and_unsafe_phrases() -> None:
    """The deterministic part of answer grading uses SME phrase lists."""
    assert answer_matches_constraints("Check flame then ignition.", ["flame"], ["replace"])
    assert not answer_matches_constraints("Replace the gas valve first.", [], ["gas valve first"])


def test_evaluate_records_scores_and_skips_page_denominator() -> None:
    """Guessed pages are counted as skipped, not failed."""
    questions = [
        question(),
        question(
            id="dgh-002",
            question="Another page question",
            must_mention=[],
            must_not_say=[],
        ),
    ]
    records = [
        record(),
        record(query_id="q_2", query="Another page question", citations=[1], chunk_ids=["c2"]),
    ]
    chunks = {
        "c1": chunk(),
        "c2": chunk(chunk_id="c2", label_source="index", page_label="42"),
    }

    report = evaluate_records(
        questions,
        records,
        chunks,
        doc_filenames={"doc1": ["644066-M MANUAL SERVICE TQ SERIES.pdf"]},
    )

    assert report.metrics["retrieval_recall_at_8"].value == 1.0
    assert report.metrics["page_accuracy"].total == 1
    assert report.metrics["page_accuracy"].skipped == 1
    assert report.metrics["answer_correctness"].value == 1.0


def test_unanswerable_refusal_is_scored_separately() -> None:
    """Unanswerables should decline instead of answer confidently."""
    report = evaluate_records(
        [question(category="unanswerable", expected_source=None, expected_page=None)],
        [
            record(
                answer="Nothing in the Seeley help centre matched that question.",
                citations=[],
                confidence="low",
            )
        ],
    )

    assert report.metrics["refusal_on_unanswerables"].value == 1.0


def test_html_report_mentions_the_citable_label_rule(tmp_path: Path) -> None:
    """The report should make the skipped-page denominator visible."""
    report = evaluate_records([question()], [record()], {"c1": chunk()})
    path = write_html_report(report, tmp_path / "eval.html")

    html = path.read_text(encoding="utf-8")
    assert "label_source == index" in html
    assert "page_accuracy" in html
