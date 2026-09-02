#!/usr/bin/env python
"""Stage 5 -- measure what reranking actually changes.

production-readiness.md B-4.

Both rerank backends are implemented; neither is on. The plan's own condition
for turning one on is "measured rather than assumed -- do not enable it
permanently on the strength of one query". The SME question set that would
settle accuracy does not exist yet (A-3), so this measures the two things that
*can* be measured honestly today:

* **How much reranking moves the list.** If the reranked top-k is the fused
  top-k in the same order, the backend is buying nothing and the latency is
  pure loss. If it moves a lot, that is not proof it is better -- but it is the
  precondition for it being better, and it is the number that says whether the
  accuracy question is worth an SME's time.
* **What it costs.** Added latency per query, on top of the cascade it wraps.
  Money is one extra LLM call per query on the configured model; the plan's
  "roughly doubles per-query cost" was written against a larger one, so the
  model actually used is printed rather than assumed.

⚠ **What this does not measure is accuracy.** There is no ground truth here, so
nothing below says the reranked order is *more correct* -- only that it differs,
by how much, and for what price. Scoring it against an LLM judge drawn from the
same family would be marking its own homework, so this deliberately does not.

**The reranker is the only variable.** The cascade runs once per query and the
fused, boosted candidate list is captured; both backends then rank that same
list. Running the cascade twice would let embedding-cache state and BM25 ties
leak into the comparison.

Exit codes:
    0 -- ran.
    1 -- no index, no queries, or retrieval failed.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from typing import Any, Sequence

from seeley_rag.exceptions import SeeleyRagError
from seeley_rag.llm import active_provider
from seeley_rag.logging_conf import configure_logging
from seeley_rag.paths import QUERY_LOG_PATH, REPORTS_DIR, rerank_ab_report_path
from seeley_rag.retrieve.hybrid import RetrievalError, retrieve
from seeley_rag.retrieve.rerank import identity_rerank, llm_rerank
from seeley_rag.settings import get_settings

#: Fallback questions when no query log exists. Same set as ``06_search.py``'s
#: demo, so the two scripts can be compared line for line.
DEMO_QUERIES = (
    "TQ heater fault code FC7 what do I check",
    "the ducted heater is throwing E:04",
    "Braemar evaporative cooler water pump not priming",
    "VRF outdoor unit E4 high discharge temperature",
    "what is the manifold gas pressure setting for a TQ5",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Compare the identity and LLM rerank backends over real queries.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("query", nargs="*", help="Questions to compare. Default: the query log.")
    parser.add_argument("--top-k", type=int, default=None, help="Chunks compared per query.")
    parser.add_argument("--limit", type=int, default=None, help="Cap the number of queries.")
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Show the queries and candidate counts, then stop. Spends nothing on reranking.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Rerank model. Defaults to retrieve.llm_rerank_model, then the router model.",
    )
    parser.add_argument("--no-report", action="store_true", help="Print only; write no file.")
    parser.add_argument("--verbose", action="store_true", help="Log at DEBUG level.")
    return parser.parse_args()


def log_queries(limit: int | None = None) -> list[str]:
    """Return the distinct questions in the query log, most recent last.

    The log is the closest thing to a real question set that exists before the
    SME set lands: every line is a question someone actually asked this system.

    Args:
        limit: Keep at most this many, from the end.

    Returns:
        Distinct query strings in first-asked order.
    """
    if not QUERY_LOG_PATH.exists():
        return []
    seen: list[str] = []
    with QUERY_LOG_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                query = str(json.loads(line).get("query") or "").strip()
            except json.JSONDecodeError:
                # A row truncated by a kill is not worth failing a report over.
                continue
            if query and query not in seen:
                seen.append(query)
    return seen[-limit:] if limit else seen


def capture_candidates(query: str, top_k: int) -> tuple[list[dict[str, Any]], float]:
    """Run the cascade once and return the fused, boosted candidates.

    ``retrieve`` truncates for whichever reranker it is given, so the reranker
    slot is used to capture the list before truncation rather than to rank it.

    Args:
        query: The installer's question.
        top_k: Chunks the real cascade would return.

    Returns:
        ``(candidates, seconds)`` -- the pre-rerank list and the cascade time.
    """
    captured: list[dict[str, Any]] = []

    def capture(
        text: str, candidates: Sequence[dict[str, Any]], limit: int
    ) -> list[dict[str, Any]]:
        """Stash the candidate list, then rank it the way the cascade does today."""
        captured.extend(candidates)
        return identity_rerank(text, candidates, limit)

    started = time.perf_counter()
    retrieve(query, top_k=top_k, reranker=capture)
    return captured, time.perf_counter() - started


def compare_one(query: str, top_k: int, model: str | None) -> dict[str, Any]:
    """Rank one query's candidates with both backends and diff the results.

    Args:
        query: The installer's question.
        top_k: Chunks compared.
        model: Rerank model override.

    Returns:
        One row of the report.
    """
    candidates, cascade_seconds = capture_candidates(query, top_k)
    base = identity_rerank(query, candidates, top_k)
    base_ids = [str(row.get("chunk_id")) for row in base]

    started = time.perf_counter()
    ranked = llm_rerank(query, candidates, top_k, model=model)
    rerank_seconds = time.perf_counter() - started

    ranked_ids = [str(row.get("chunk_id")) for row in ranked]
    # A fallback inside llm_rerank relabels its rows "identity"; the backend the
    # rows carry is therefore the honest answer to "did this actually rerank?".
    backends = {str(row.get("rerank_backend")) for row in ranked}

    # Where each reranked chunk sat in the fused order. A chunk the fusion never
    # surfaced in its top-k has no rank there, recorded as None rather than 0.
    previous: list[int | None] = [
        base_ids.index(chunk_id) + 1 if chunk_id in base_ids else None for chunk_id in ranked_ids
    ]
    moved = [
        abs(position - was) for position, was in enumerate(previous, start=1) if was is not None
    ]
    return {
        "query": query,
        "candidates": len(candidates),
        "cascade_seconds": cascade_seconds,
        "rerank_seconds": rerank_seconds,
        "backend": "llm" if backends == {"llm"} else "+".join(sorted(backends)),
        "overlap": len(set(base_ids) & set(ranked_ids)),
        "top_k": top_k,
        "top1_changed": bool(base_ids and ranked_ids and base_ids[0] != ranked_ids[0]),
        "top1_was_ranked": previous[0] if previous else None,
        "mean_movement": statistics.mean(moved) if moved else 0.0,
        "identity_top1": _describe(base[0]) if base else "-",
        "llm_top1": _describe(ranked[0]) if ranked else "-",
    }


def _describe(row: dict[str, Any]) -> str:
    """Return a one-line label for a chunk.

    Args:
        row: A ranked chunk.

    Returns:
        ``Title p.42``.
    """
    title = str(row.get("title") or "Untitled")[:60]
    page = row.get("page_range") or row.get("page_label")
    return f"{title} p.{page}" if page else title


def summarise(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the per-query rows.

    Args:
        rows: Comparison rows.

    Returns:
        The aggregate figures.
    """
    top_k = rows[0]["top_k"] if rows else 0
    latencies = [row["rerank_seconds"] for row in rows]
    return {
        "queries": len(rows),
        "top_k": top_k,
        "mean_overlap": statistics.mean(row["overlap"] for row in rows) if rows else 0.0,
        "identical": sum(1 for row in rows if row["overlap"] == top_k and not row["top1_changed"]),
        "top1_changed": sum(1 for row in rows if row["top1_changed"]),
        "mean_movement": statistics.mean(row["mean_movement"] for row in rows) if rows else 0.0,
        "median_seconds": statistics.median(latencies) if latencies else 0.0,
        "max_seconds": max(latencies) if latencies else 0.0,
        "fell_back": sum(1 for row in rows if row["backend"] != "llm"),
    }


def render_report(rows: Sequence[dict[str, Any]], totals: dict[str, Any], model: str) -> str:
    """Render the markdown report.

    Args:
        rows: Comparison rows.
        totals: Aggregate figures.
        model: The rerank model used.

    Returns:
        The report body.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    lines = [
        "# Reranking A/B -- identity vs listwise LLM",
        "",
        f"Generated {stamp}. Model `{model}`, provider `{active_provider()}`.",
        "",
        "Both backends ranked the *same* fused, boosted candidate list, so the",
        "reranker is the only variable. **These numbers say how much the order",
        "changes and what that costs. They do not say the new order is better** --",
        "that needs the SME question set (production-readiness A-3).",
        "",
        "## Summary",
        "",
        f"- Queries: {totals['queries']}, top-{totals['top_k']} compared",
        f"- Mean overlap: {totals['mean_overlap']:.1f} of {totals['top_k']} chunks unchanged",
        f"- Identical result (same chunks, same first): {totals['identical']}"
        f" of {totals['queries']}",
        f"- Top-1 changed: {totals['top1_changed']} of {totals['queries']}",
        f"- Mean rank movement: {totals['mean_movement']:.2f} places",
        f"- Rerank latency: median {totals['median_seconds']:.2f}s,"
        f" max {totals['max_seconds']:.2f}s",
        f"- Fell back to identity (a failed call): {totals['fell_back']}",
        "",
        "## Per query",
        "",
    ]
    for row in rows:
        promoted = row["top1_was_ranked"]
        promotion = f"was #{promoted}" if promoted else "was outside the fused top-k"
        lines.extend(
            [
                f"### {row['query']}",
                "",
                f"- Candidates fused: {row['candidates']}; rerank {row['rerank_seconds']:.2f}s"
                f" on top of a {row['cascade_seconds']:.2f}s cascade",
                f"- Overlap {row['overlap']}/{row['top_k']},"
                f" mean movement {row['mean_movement']:.2f} places",
                f"- Fusion first: {row['identity_top1']}",
                f"- Rerank first: {row['llm_top1']} ({promotion})",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    """Compare the backends and write the report.

    Returns:
        Process exit code.
    """
    args = parse_args()
    configure_logging(level="DEBUG" if args.verbose else None)

    settings = get_settings().retrieve
    top_k = args.top_k or settings.rerank_top_k
    model = args.model or settings.llm_rerank_model or get_settings().generate.router_model

    typed = " ".join(args.query).strip()
    queries = [typed] if typed else log_queries(args.limit) or list(DEMO_QUERIES)[: args.limit]
    if not queries:
        print("No queries. Type one, or run some through scripts/07_ask.py first.")
        return 1

    print(f"Reranking A/B: {len(queries)} queries, top-{top_k}, model={model}")
    print(f"Candidate pool: {settings.llm_rerank_candidates} per query\n")
    if args.plan:
        for query in queries:
            print(f"  - {query}")
        print(f"\n{len(queries)} rerank calls would be made. Nothing was spent.")
        return 0

    rows: list[dict[str, Any]] = []
    try:
        for number, query in enumerate(queries, start=1):
            row = compare_one(query, top_k, model)
            rows.append(row)
            flag = "" if row["backend"] == "llm" else f"  [{row['backend']}]"
            print(
                f"{number:>3}. overlap {row['overlap']}/{top_k}"
                f"  moved {row['mean_movement']:.2f}"
                f"  {'top1 CHANGED' if row['top1_changed'] else 'top1 same  '}"
                f"  {row['rerank_seconds']:.2f}s{flag}  {query[:52]}"
            )
    except RetrievalError as exc:
        print(f"Retrieval failed: {exc}")
        return 1
    except SeeleyRagError as exc:
        print(f"Failed: {exc}")
        return 1

    totals = summarise(rows)
    print(
        f"\nMean overlap {totals['mean_overlap']:.1f}/{top_k}"
        f" | top-1 changed {totals['top1_changed']}/{totals['queries']}"
        f" | identical {totals['identical']}/{totals['queries']}"
        f" | median {totals['median_seconds']:.2f}s"
    )
    if totals["fell_back"]:
        print(
            f"WARNING: {totals['fell_back']} query(ies) fell back to identity. "
            "Those rows measure nothing -- check the log before reading the summary."
        )
    print("This measures movement and cost, NOT accuracy. Accuracy needs the SME set (A-3).")

    if not args.no_report:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        path = rerank_ab_report_path(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
        path.write_text(render_report(rows, totals, model), encoding="utf-8")
        print(f"\nReport: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
