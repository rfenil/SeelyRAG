#!/usr/bin/env python
"""Stage 8 -- evaluate.

Loads the SME question set, joins it to the query/feedback logs, and writes an
HTML report for the gates in build-plan.md section 10. Use ``--run`` to execute
the question set against the current index before reporting.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from seeley_rag.evaluate import (
    EvalQuestion,
    QueryRecord,
    evaluate_records,
    load_chunk_index,
    load_doc_filenames,
    load_feedback,
    load_query_records,
    load_questions,
    write_html_report,
)
from seeley_rag.paths import (
    CHUNKS_PATH,
    FEEDBACK_LOG_PATH,
    MANIFEST_PATH,
    QUERY_LOG_PATH,
    REPORTS_DIR,
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Seeley RAG against the SME set.")
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("_context/04-eval/sme-question-template.yaml"),
        help="YAML SME question set.",
    )
    parser.add_argument("--queries", type=Path, default=QUERY_LOG_PATH, help="queries.jsonl path.")
    parser.add_argument(
        "--feedback", type=Path, default=FEEDBACK_LOG_PATH, help="feedback.jsonl path."
    )
    parser.add_argument("--chunks", type=Path, default=CHUNKS_PATH, help="chunks.jsonl path.")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH, help="manifest.jsonl path.")
    parser.add_argument(
        "--report",
        type=Path,
        default=REPORTS_DIR / "eval.html",
        help="HTML report output path.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run the SME questions against the current index before reporting.",
    )
    return parser.parse_args(argv)


def run_questions(questions: list[EvalQuestion]) -> list[QueryRecord]:
    """Execute the question set against the current index."""
    from seeley_rag.generate.answer import answer
    from seeley_rag.retrieve.hybrid import retrieve

    records: list[QueryRecord] = []
    for question in questions:
        result = retrieve(question.question, top_k=8, product_hint=question.product_family)
        chunks = result["results"]
        response = answer(
            question.question,
            chunks=chunks,
            pinned=result["pinned_codes"],
            product_family=result["understanding"].product_family,
            eval_id=question.id,
        )
        records.append(
            QueryRecord(
                query_id=response.query_id,
                eval_id=question.id,
                query=question.question,
                chunk_ids=[str(row.get("chunk_id") or "") for row in chunks],
                answer=response.answer,
                citations=[citation.n for citation in response.citations],
                confidence=response.confidence,
                product_family=response.product_family,
                latency_ms=response.latency_ms,
            )
        )
    return records


def main(argv: list[str] | None = None) -> int:
    """Run evaluation and write the report."""
    args = parse_args(argv or sys.argv[1:])
    questions = load_questions(args.questions)
    records = run_questions(questions) if args.run else load_query_records(args.queries)
    feedback = load_feedback(args.feedback)
    chunks = load_chunk_index(args.chunks)
    filenames = load_doc_filenames(args.manifest)
    report = evaluate_records(questions, records, chunks, feedback, filenames)
    write_html_report(report, args.report)

    print(f"Evaluated {len(report.cases)} SME questions.")
    for metric in report.metrics.values():
        if metric.name == "p95_latency_ms":
            value = "n/a" if metric.value is None else f"{int(metric.value)} ms"
        else:
            value = "n/a" if metric.value is None else f"{metric.value:.1%}"
        status = "n/a" if metric.ok is None else ("PASS" if metric.ok else "FAIL")
        print(f"  {metric.name}: {value} ({status})")
    print(f"HTML report: {args.report}")
    return 0 if all(metric.ok is not False for metric in report.metrics.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
