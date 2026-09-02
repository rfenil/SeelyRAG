#!/usr/bin/env python
"""Stage 5 -- run the retrieval cascade against the built index.

build-plan.md section 7.

Reads ``data/03_index/`` and ``data/02_processed/codes.jsonl``, and prints what
each stage of the cascade did: the parsed query, any pinned fault codes, and the
ranked chunks with the boosts that fired.

``--explain`` shows the per-channel ranks behind each fused score, which is the
fastest way to tell a dense-retrieval problem from a BM25 one.

Exit codes:
    0 -- ran.
    1 -- no index, or retrieval failed.
"""

from __future__ import annotations

import argparse
import sys
import time

from seeley_rag.exceptions import SeeleyRagError
from seeley_rag.llm import active_provider
from seeley_rag.logging_conf import configure_logging
from seeley_rag.retrieve.hybrid import RetrievalError, retrieve
from seeley_rag.retrieve.rerank import llm_rerank, rerank_backend
from seeley_rag.settings import get_settings

#: Questions used by ``--demo``: one per product family, mixing exact-code
#: lookups with natural-language symptom descriptions, because the two exercise
#: opposite channels.
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
        description="Query the Seeley index through the Stage 5 cascade.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("query", nargs="*", help="The question to ask.")
    parser.add_argument("--top-k", type=int, default=None, help="Chunks to return.")
    parser.add_argument("--demo", action="store_true", help="Run the built-in demo questions.")
    parser.add_argument("--explain", action="store_true", help="Show per-channel ranks and boosts.")
    # Tri-state on purpose. `store_true` alone yields False when the flag is
    # absent, which FORCES the rewrite off and makes retrieve.use_query_llm
    # unreachable from here -- so the flag silently overrode config rather than
    # defaulting to it. Absent means None: let the config decide.
    rewrite = parser.add_mutually_exclusive_group()
    rewrite.add_argument(
        "--llm",
        dest="llm",
        action="store_true",
        default=None,
        help="Force the LLM query rewrite on.",
    )
    rewrite.add_argument(
        "--no-llm",
        dest="llm",
        action="store_false",
        help="Force the LLM query rewrite off. Default: retrieve.use_query_llm.",
    )
    parser.add_argument(
        "--llm-rerank",
        action="store_true",
        help="Listwise LLM reranking. Off by default: build-plan 7.2 warns on the cost.",
    )
    parser.add_argument("--verbose", action="store_true", help="Log at DEBUG level.")
    return parser.parse_args()


def show(
    query: str,
    top_k: int | None,
    explain: bool,
    use_llm: bool,
    reranker: object | None = None,
) -> None:
    """Run one query and print the cascade's output.

    Args:
        query: The question.
        top_k: Chunks to return.
        explain: Whether to show per-channel ranks.
        use_llm: Whether query understanding may call an LLM.
        reranker: Optional reranking callable, overriding the configured one.
    """
    started = time.monotonic()
    result = retrieve(query, top_k=top_k, use_llm=use_llm, reranker=reranker)
    elapsed_ms = (time.monotonic() - started) * 1000

    parsed = result["understanding"]
    print("=" * 78)
    print(f"Q: {query}")
    print(
        f"   family={parsed.product_family} models={parsed.model_series} "
        f"codes={parsed.fault_codes} intent={parsed.intent} "
        f"diagram={parsed.wants_diagram} ({parsed.source})"
    )
    if parsed.rewritten_query != parsed.query:
        print(f"   rewritten: {parsed.rewritten_query}")

    if result["pinned_codes"]:
        print("\n   PINNED FAULT CODES (exact lookup, ahead of retrieval):")
        for pin in result["pinned_codes"]:
            row = pin.row
            page = f"p.{row.page_label}" if row.page_label else "no page"
            flag = "  (NOT this product family)" if pin.cross_family else ""
            print(f"     {row.code_key} [{row.product_family}]{flag} {row.meaning[:56]}")
            print(f"       {row.title[:60]} ({page})")

    counts = result["counts"]
    print(
        f"\n   dense={counts['dense']} bm25={counts['bm25']} fused={counts['fused']} "
        f"-> {counts['returned']} in {elapsed_ms:.0f}ms"
    )
    print()
    for position, row in enumerate(result["results"], start=1):
        page = f"p.{row['page_label']}" if row.get("page_label") else "no page"
        boosts = ",".join(row.get("boosts") or []) or "-"
        print(
            f"   {position}. [{row.get('product_family')}/{row.get('kind')}] "
            f"{str(row.get('title', ''))[:52]} ({page})"
        )
        print(
            f"      score={row.get('rerank_score', 0.0):.5f} boosts={boosts}"
            + (f" ranks={row.get('ranks')}" if explain else "")
        )
        if explain:
            body = " ".join(str(row.get("text", "")).split())
            print(f"      {body[:150]}...")
    print()


def _rewrite_state(flag: bool | None) -> str:
    """Describe whether the query rewrite will run.

    Args:
        flag: The tri-state ``--llm`` / ``--no-llm`` value.

    Returns:
        ``on``, ``off``, or the config-derived state.
    """
    if flag is None:
        return "on (config)" if get_settings().retrieve.use_query_llm else "off (config)"
    return "on (flag)" if flag else "off (flag)"


def main() -> int:
    """Run the cascade over the requested queries.

    Returns:
        Process exit code.
    """
    args = parse_args()
    configure_logging(level="DEBUG" if args.verbose else None)

    queries = list(DEMO_QUERIES) if args.demo else [" ".join(args.query).strip()]
    if not queries or not queries[0]:
        print("Give a query, or pass --demo.")
        return 1

    settings = get_settings().retrieve
    reranker = llm_rerank if args.llm_rerank else None
    backend = "llm" if args.llm_rerank else rerank_backend()
    print(
        f"Cascade: dense={settings.dense_top_k} + bm25={settings.bm25_top_k} "
        f"-> RRF(k={settings.rrf_k}) -> boosts -> rerank[{backend}] "
        f"-> top {args.top_k or settings.rerank_top_k}"
    )
    rewrite = _rewrite_state(args.llm)
    print(f"Query rewrite: {rewrite} (provider={active_provider()})\n")
    if backend == "identity":
        print(
            "NOTE: nothing is reranking. No COHERE_API_KEY and --llm-rerank not "
            "set, so results are the boosted fusion order. build-plan 7.2 step 6.\n"
        )

    try:
        for query in queries:
            show(query, args.top_k, args.explain, use_llm=args.llm, reranker=reranker)
    except RetrievalError as exc:
        print(f"Retrieval failed: {exc}")
        return 1
    except SeeleyRagError as exc:
        print(f"Failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
