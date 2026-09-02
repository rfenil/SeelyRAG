#!/usr/bin/env python
"""Stage 6 -- ask a question and get a cited answer.

build-plan.md section 8.

Runs the Stage 5 cascade, sends the passages to the configured model, and prints
the answer with its citations resolved to page labels, page images and the
Freshdesk articles an installer can open to verify.

Every question is appended to ``data/reports/queries.jsonl`` with its
``query_id``, the chunk IDs retrieved and the answer given -- build-plan section
9. That log is the input to the eval, and to any conversation about whether the
system is actually working.

Exit codes:
    0 -- answered, or honestly declined to answer.
    1 -- no index, no key, or retrieval failed.
"""

from __future__ import annotations

import argparse
import sys

from seeley_rag.api.schemas import Citation
from seeley_rag.exceptions import SeeleyRagError
from seeley_rag.generate.answer import answer
from seeley_rag.llm import active_provider, is_configured
from seeley_rag.logging_conf import configure_logging
from seeley_rag.paths import QUERY_LOG_PATH
from seeley_rag.retrieve.rerank import llm_rerank, rerank_backend
from seeley_rag.settings import get_settings

#: Questions spanning the corpus's product families and both question shapes --
#: an exact code lookup, a symptom description, a specification, and one the
#: corpus should decline rather than answer.
DEMO_QUESTIONS = (
    "TQ heater showing FC7, what do I check?",
    "the ducted heater is throwing E:04",
    "what is the manifold gas pressure setting for a TQ5",
    "Braemar evaporative cooler water pump not priming",
    "what is the warranty period on a Tesla Powerwall",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Ask the Seeley installer assistant a question.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("question", nargs="*", help="The question to ask.")
    parser.add_argument("--demo", action="store_true", help="Run the built-in demo questions.")
    parser.add_argument("--top-k", type=int, default=None, help="Passages to retrieve.")
    parser.add_argument("--model", default=None, help="Override the generation model.")
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
        "--llm-rerank", action="store_true", help="Listwise LLM reranking before answering."
    )
    parser.add_argument(
        "--snippets", action="store_true", help="Print the supporting snippet for each citation."
    )
    parser.add_argument("--verbose", action="store_true", help="Log at DEBUG level.")
    return parser.parse_args()


def group_sources(citations: list[Citation]) -> list[list[Citation]]:
    """Group citations that resolve to the same document and page.

    Args:
        citations: Citations from one answer, in marker order.

    Returns:
        Groups, each sharing a title and printed page, in first-seen order.
    """
    groups: dict[tuple[str, str | None], list[Citation]] = {}
    for citation in citations:
        groups.setdefault((citation.title, citation.page_label), []).append(citation)
    return list(groups.values())


def ask(question: str, args: argparse.Namespace) -> None:
    """Answer one question and print it.

    Args:
        question: The installer's question.
        args: Parsed command-line arguments.
    """
    response = answer(
        question,
        model=args.model,
        top_k=args.top_k,
        use_llm=args.llm,
        reranker=llm_rerank if args.llm_rerank else None,
    )

    print("=" * 78)
    print(f"Q: {question}")
    print(
        f"   [{response.query_id}] confidence={response.confidence} "
        f"family={response.product_family} {response.latency_ms}ms"
    )
    print()
    for line in response.answer.splitlines():
        print(f"  {line}")

    if response.citations:
        print("\n  SOURCES")
        # Two chunks from the same page are two citations in the API response
        # -- each marker must resolve -- but showing one page twice is noise,
        # so they share a line here. Deduplicating at retrieval instead stays
        # deferred (ADR 0007): capping per-document hits would have dropped
        # the very page carrying the gas-pressure table.
        for group in group_sources(response.citations):
            first = group[0]
            markers = ", ".join(str(c.n) for c in group)
            page = f"p.{first.page_label}" if first.page_label else "no printed page"
            print(f"    [{markers}] {first.title} ({page})")
            if first.article_url:
                print(f"        verify: {first.article_url}")
            if first.page_image:
                print(f"        image:  {first.page_image}")
            if args.snippets and first.snippet:
                print(f'        "{first.snippet[:150]}..."')
    else:
        print("\n  (no citations -- nothing in the corpus supported an answer)")
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
    """Answer the requested questions.

    Returns:
        Process exit code.
    """
    args = parse_args()
    configure_logging(level="DEBUG" if args.verbose else None)

    questions = list(DEMO_QUESTIONS) if args.demo else [" ".join(args.question).strip()]
    if not questions or not questions[0]:
        print("Ask a question, or pass --demo.")
        return 1

    if not is_configured():
        print(
            f"No API key for the configured provider ({active_provider()}). "
            "Fill in .env, or change generate.provider in config/config.yaml."
        )
        return 1

    settings = get_settings().generate
    backend = "llm" if args.llm_rerank else rerank_backend()
    print(
        f"Model: {args.model or settings.model} ({settings.provider}) | "
        f"rerank[{backend}] | rewrite {_rewrite_state(args.llm)}"
    )
    print(f"Query log: {QUERY_LOG_PATH}\n")

    try:
        for question in questions:
            ask(question, args)
    except SeeleyRagError as exc:
        print(f"Failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
