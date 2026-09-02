"""Stage 8 -- evaluation.

build-plan.md section 10.

The evaluator joins the SME question set to answer logs, then scores the pieces
that can be checked deterministically. The page-label rule is deliberately
conservative: labels with ``label_source == "index"`` are guesses, so expected
pages backed only by those rows are excluded from page-accuracy denominators
rather than counted as retrieval failures.
"""

from __future__ import annotations

import html
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from seeley_rag.exceptions import SeeleyRagError

QuestionCategory = Literal[
    "fault_diagnosis", "installation", "spec_lookup", "diagram", "unanswerable"
]

GUESS_LABEL_SOURCES = {"", "none", "index"}
GATES = {
    "retrieval_recall_at_8": 0.85,
    "page_accuracy": 0.70,
    "citation_validity": 0.95,
    "answer_correctness": 0.80,
    "refusal_on_unanswerables": 0.90,
    "p95_latency_ms": 6000,
}

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


class EvalError(SeeleyRagError):
    """The evaluation inputs could not be loaded or scored."""


class EvalQuestion(BaseModel):
    """One SME-authored evaluation question."""

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    id: str
    question: str
    category: QuestionCategory
    product_family: str
    model: str | None = None
    expected_source: str | None = None
    expected_page: int | str | None = None
    must_mention: list[str] = Field(default_factory=list)
    must_not_say: list[str] = Field(default_factory=list)
    notes: str | None = None


class QueryRecord(BaseModel):
    """One row from ``data/reports/queries.jsonl``."""

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    query_id: str
    query: str
    chunk_ids: list[str] = Field(default_factory=list)
    answer: str = ""
    citations: list[int] = Field(default_factory=list)
    confidence: str = "unknown"
    product_family: str | None = None
    latency_ms: int = 0
    eval_id: str | None = None


class FeedbackRecord(BaseModel):
    """One row from ``data/reports/feedback.jsonl``."""

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    query_id: str
    rating: str
    comment: str | None = None


@dataclass
class Metric:
    """Aggregate for one gate."""

    name: str
    passed: int = 0
    total: int = 0
    skipped: int = 0
    gate: float | int | None = None
    lower_is_better: bool = False

    @property
    def value(self) -> float | None:
        """Return the metric value, or ``None`` when no cases were scorable."""
        if not self.total:
            return None
        if self.lower_is_better:
            return float(self.passed)
        return self.passed / self.total

    @property
    def ok(self) -> bool | None:
        """Whether this metric meets its gate."""
        value = self.value
        if value is None or self.gate is None:
            return None
        if self.lower_is_better:
            return value < self.gate
        return value >= self.gate


@dataclass
class CaseResult:
    """Scores for one SME question."""

    question: EvalQuestion
    query_id: str | None = None
    retrieval_hit: bool | None = None
    page_hit: bool | None = None
    page_skipped_reason: str | None = None
    citation_valid: bool | None = None
    answer_correct: bool | None = None
    refused: bool | None = None
    latency_ms: int | None = None
    feedback_rating: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class EvalReport:
    """Full evaluation result."""

    cases: list[CaseResult]
    metrics: dict[str, Metric]
    p95_latency_ms: int | None
    output_path: Path | None = None


def load_questions(path: Path) -> list[EvalQuestion]:
    """Load SME questions from the YAML template format."""
    if not path.exists():
        raise EvalError(f"No SME question set at {path}.")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw_questions = data.get("questions") if isinstance(data, dict) else data
    if not isinstance(raw_questions, list):
        raise EvalError(f"{path} must contain a top-level questions list.")
    return [EvalQuestion.model_validate(row) for row in raw_questions if isinstance(row, dict)]


def load_query_records(path: Path) -> list[QueryRecord]:
    """Load query log rows. Missing logs are treated as empty."""
    return [QueryRecord.model_validate(row) for row in _read_jsonl(path)]


def load_feedback(path: Path) -> dict[str, FeedbackRecord]:
    """Load latest feedback keyed by query id."""
    feedback: dict[str, FeedbackRecord] = {}
    for row in _read_jsonl(path):
        record = FeedbackRecord.model_validate(row)
        feedback[record.query_id] = record
    return feedback


def load_chunk_index(path: Path) -> dict[str, dict[str, Any]]:
    """Load chunks keyed by id. Missing chunks leave source/page checks skipped."""
    return {
        str(row["chunk_id"]): row
        for row in _read_jsonl(path)
        if isinstance(row, dict) and row.get("chunk_id")
    }


def load_doc_filenames(path: Path) -> dict[str, list[str]]:
    """Return ``doc_id -> filenames`` from the manifest, if available."""
    filenames: dict[str, list[str]] = {}
    for article in _read_jsonl(path):
        for attachment in article.get("attachments") or []:
            if not isinstance(attachment, dict):
                continue
            sha256 = attachment.get("sha256")
            filename = attachment.get("filename")
            if not sha256 or not filename:
                continue
            bucket = filenames.setdefault(str(sha256), [])
            if str(filename) not in bucket:
                bucket.append(str(filename))
    return filenames


def evaluate_records(
    questions: Iterable[EvalQuestion],
    records: Iterable[QueryRecord],
    chunks: dict[str, dict[str, Any]] | None = None,
    feedback: dict[str, FeedbackRecord] | None = None,
    doc_filenames: dict[str, list[str]] | None = None,
) -> EvalReport:
    """Score query logs against SME questions."""
    chunks = chunks or {}
    feedback = feedback or {}
    doc_filenames = doc_filenames or {}
    by_question = _match_records(records)

    cases: list[CaseResult] = []
    for question in questions:
        record = by_question.get(question.id) or by_question.get(_normalise_text(question.question))
        case = CaseResult(question=question)
        if record is None:
            case.notes.append("No matching query log row.")
            cases.append(case)
            continue

        case.query_id = record.query_id
        case.latency_ms = record.latency_ms
        if record.query_id in feedback:
            case.feedback_rating = feedback[record.query_id].rating

        retrieved = [chunks[cid] for cid in record.chunk_ids if cid in chunks]
        cited = _cited_rows(record, chunks)
        if question.expected_source:
            if not chunks:
                case.notes.append("Chunk index unavailable; source recall cannot be verified.")
            else:
                case.retrieval_hit = any(
                    source_matches(question.expected_source, row, doc_filenames)
                    for row in retrieved[:8]
                )

        expected_page = _expected_page_int(question.expected_page)
        if expected_page is not None:
            case.page_hit, case.page_skipped_reason = page_accuracy(expected_page, cited)

        if record.citations and chunks:
            case.citation_valid = citation_resolves(record, chunks)
        elif record.citations:
            case.notes.append("Chunk index unavailable; citation resolution cannot be verified.")

        if question.category == "unanswerable":
            case.refused = is_refusal(record)
        elif question.must_mention or question.must_not_say:
            case.answer_correct = answer_matches_constraints(
                record.answer, question.must_mention, question.must_not_say
            )

        cases.append(case)

    return summarise(cases)


def summarise(cases: list[CaseResult]) -> EvalReport:
    """Aggregate case-level results into gate metrics."""
    metrics = {
        "retrieval_recall_at_8": _boolean_metric(
            "retrieval_recall_at_8",
            (c.retrieval_hit for c in cases),
            GATES["retrieval_recall_at_8"],
        ),
        "page_accuracy": _boolean_metric(
            "page_accuracy",
            (c.page_hit for c in cases),
            GATES["page_accuracy"],
            skipped=sum(1 for c in cases if c.page_skipped_reason),
        ),
        "citation_validity": _boolean_metric(
            "citation_validity",
            (c.citation_valid for c in cases),
            GATES["citation_validity"],
        ),
        "answer_correctness": _boolean_metric(
            "answer_correctness",
            (c.answer_correct for c in cases),
            GATES["answer_correctness"],
        ),
        "refusal_on_unanswerables": _boolean_metric(
            "refusal_on_unanswerables",
            (c.refused for c in cases),
            GATES["refusal_on_unanswerables"],
        ),
    }
    latencies = [c.latency_ms for c in cases if c.latency_ms is not None]
    p95 = percentile(latencies, 95) if latencies else None
    latency = Metric(
        "p95_latency_ms",
        passed=1 if p95 is not None and p95 < GATES["p95_latency_ms"] else 0,
        total=1 if p95 is not None else 0,
        gate=GATES["p95_latency_ms"],
        lower_is_better=True,
    )
    if p95 is not None:
        latency.passed = p95
    metrics["p95_latency_ms"] = latency
    return EvalReport(cases=cases, metrics=metrics, p95_latency_ms=p95)


def page_accuracy(
    expected_page: int, cited_rows: list[dict[str, Any]]
) -> tuple[bool | None, str | None]:
    """Check expected page against cited labels, excluding guessed labels."""
    trusted = [row for row in cited_rows if label_is_citable(row)]
    if not trusted:
        return None, "Only guessed or missing page labels were cited."
    for row in trusted:
        for page in page_numbers(row.get("page_range") or row.get("page_label")):
            if abs(page - expected_page) <= 1:
                return True, None
    return False, None


def label_is_citable(row: dict[str, Any]) -> bool:
    """Whether a chunk's page label can be scored."""
    source = str(row.get("label_source") or "").lower()
    return source not in GUESS_LABEL_SOURCES and bool(
        row.get("page_label") or row.get("page_range")
    )


def page_numbers(label: Any) -> list[int]:
    """Extract numeric page labels from a single label or range."""
    if label is None:
        return []
    text = str(label)
    match = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", text)
    if match:
        start, end = int(match.group(1)), int(match.group(2))
        if start <= end and end - start <= 20:
            return list(range(start, end + 1))
    return [int(n) for n in re.findall(r"\d+", text)]


def source_matches(
    expected_source: str,
    row: dict[str, Any],
    doc_filenames: dict[str, list[str]] | None = None,
) -> bool:
    """Check whether a retrieved row comes from the expected document/article."""
    doc_filenames = doc_filenames or {}
    candidates = [
        row.get("title"),
        row.get("source_url"),
        row.get("article_url"),
        row.get("doc_id"),
        *(doc_filenames.get(str(row.get("doc_id") or ""), [])),
    ]
    expected = _normalise_source(expected_source)
    for candidate in candidates:
        actual = _normalise_source(str(candidate or ""))
        if expected and (expected in actual or actual in expected):
            return True
    return False


def citation_resolves(record: QueryRecord, chunks: dict[str, dict[str, Any]]) -> bool:
    """Whether every cited marker resolves to a retrieved chunk row."""
    if not record.citations:
        return False
    for number in record.citations:
        index = number - 1
        if index < 0 or index >= len(record.chunk_ids):
            return False
        if record.chunk_ids[index] not in chunks:
            return False
    return True


def answer_matches_constraints(
    answer: str, must_mention: Iterable[str], must_not_say: Iterable[str]
) -> bool:
    """Check deterministic answer constraints from the SME set."""
    lowered = answer.casefold()
    return all(term.casefold() in lowered for term in must_mention) and not any(
        term.casefold() in lowered for term in must_not_say
    )


def is_refusal(record: QueryRecord) -> bool:
    """Heuristic for unanswerable cases: low/no-citation decline language."""
    text = record.answer.casefold()
    refusal_terms = (
        "not covered",
        "do not cover",
        "does not cover",
        "cannot answer",
        "can't answer",
        "nothing in",
        "no matching",
        "do not have",
        "don't have",
        "not in the",
    )
    return not record.citations and (
        record.confidence == "low" or any(term in text for term in refusal_terms)
    )


def percentile(values: list[int], pct: int) -> int:
    """Nearest-rank percentile."""
    if not values:
        raise ValueError("percentile() needs at least one value")
    ordered = sorted(values)
    rank = math.ceil((pct / 100) * len(ordered))
    return ordered[min(max(rank - 1, 0), len(ordered) - 1)]


def write_html_report(report: EvalReport, path: Path) -> Path:
    """Write a compact HTML evaluation report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = "\n".join(_case_row(case) for case in report.cases)
    metric_rows = "\n".join(_metric_row(metric) for metric in report.metrics.values())
    note = (
        "Page accuracy excludes citations whose labels are guessed "
        "(label_source == index), matching the 61.3% citable-label ceiling."
    )
    path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Seeley RAG Evaluation</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f0f4f8; }}
    .pass {{ color: #0b6b3a; font-weight: 700; }}
    .fail {{ color: #a61b1b; font-weight: 700; }}
    .skip {{ color: #7b8794; }}
    .note {{ max-width: 900px; color: #52606d; }}
  </style>
</head>
<body>
  <h1>Seeley RAG Evaluation</h1>
  <p class="note">{html.escape(note)}</p>
  <h2>Gates</h2>
  <table>
    <thead><tr><th>Metric</th><th>Value</th><th>Gate</th><th>Cases</th><th>Status</th></tr></thead>
    <tbody>{metric_rows}</tbody>
  </table>
  <h2>Cases</h2>
  <table>
    <thead><tr><th>ID</th><th>Category</th><th>Query ID</th>
    <th>Retrieval</th><th>Page</th><th>Answer</th>
    <th>Refusal</th><th>Notes</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</body>
</html>
""",
        encoding="utf-8",
    )
    report.output_path = path
    return path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvalError(f"{path}:{number} is not valid JSON: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _match_records(records: Iterable[QueryRecord]) -> dict[str, QueryRecord]:
    matched: dict[str, QueryRecord] = {}
    for record in records:
        for key in (record.eval_id, _normalise_text(record.query)):
            if key:
                matched[key] = record
    return matched


def _cited_rows(record: QueryRecord, chunks: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number in record.citations:
        index = number - 1
        if 0 <= index < len(record.chunk_ids):
            row = chunks.get(record.chunk_ids[index])
            if row:
                rows.append(row)
    return rows


def _expected_page_int(value: int | str | None) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, int):
        return value
    match = re.search(r"\d+", str(value))
    return int(match.group(0)) if match else None


def _boolean_metric(
    name: str, values: Iterable[bool | None], gate: float, skipped: int = 0
) -> Metric:
    seen = [v for v in values if v is not None]
    return Metric(
        name=name,
        passed=sum(1 for v in seen if v),
        total=len(seen),
        skipped=skipped,
        gate=gate,
    )


def _normalise_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _normalise_source(value: str) -> str:
    return _NON_ALNUM_RE.sub("", value.casefold())


def _metric_row(metric: Metric) -> str:
    if metric.name == "p95_latency_ms":
        value = "n/a" if not metric.total else f"{metric.passed} ms"
        status = _status_class(metric.ok)
        gate = f"&lt; {metric.gate} ms"
        cases = "1" if metric.total else "0"
    else:
        value = "n/a" if metric.value is None else f"{metric.value:.1%}"
        status = _status_class(metric.ok)
        gate = f">= {metric.gate:.0%}" if isinstance(metric.gate, float) else str(metric.gate)
        cases = f"{metric.passed}/{metric.total}"
        if metric.skipped:
            cases += f" ({metric.skipped} skipped)"
    return (
        "<tr>"
        f"<td>{html.escape(metric.name)}</td><td>{value}</td><td>{gate}</td>"
        f'<td>{cases}</td><td class="{status}">{status}</td>'
        "</tr>"
    )


def _case_row(case: CaseResult) -> str:
    notes = list(case.notes)
    if case.page_skipped_reason:
        notes.append(case.page_skipped_reason)
    if case.feedback_rating:
        notes.append(f"feedback: {case.feedback_rating}")
    return (
        "<tr>"
        f"<td>{html.escape(case.question.id)}</td>"
        f"<td>{html.escape(case.question.category)}</td>"
        f"<td>{html.escape(case.query_id or '')}</td>"
        f"<td>{_flag(case.retrieval_hit)}</td>"
        f"<td>{_flag(case.page_hit)}</td>"
        f"<td>{_flag(case.answer_correct)}</td>"
        f"<td>{_flag(case.refused)}</td>"
        f"<td>{html.escape('; '.join(notes))}</td>"
        "</tr>"
    )


def _flag(value: bool | None) -> str:
    if value is None:
        return '<span class="skip">n/a</span>'
    if value:
        return '<span class="pass">pass</span>'
    return '<span class="fail">fail</span>'


def _status_class(value: bool | None) -> str:
    if value is None:
        return "skip"
    return "pass" if value else "fail"
