#!/usr/bin/env python
"""Stage 5 -- what the cascade does with the queries installers actually type.

Every question in `data/reports/queries.jsonl` is well-formed, because whoever
wrote them knew the system: *"TQ heater showing FC7, what do I check?"*. The
end users are trade workers on a roof with one hand free. They type `fc7`.

That difference is not cosmetic. Measured before this script existed, a bare
`fc7` returned a confident, `high`-confidence answer about Climate Wizard
three-phase motor voltage and never mentioned the gas-heater ignition failure
the installer was almost certainly standing in front of -- because `UNKNOWN` was
being treated as a matching product family in the code table, pinning the one
row whose meaning is the string "FAULT CODE 7".

So this is a fixture, not a demo. It exercises the register real users write in:

* bare fault codes with no product (`fc7`, `e4`) -- the dangerous shape
* symptom-only, no product (`no hot air`, `pump not working`)
* misspellings (`braemer heater no ignition`, `gas presure tq5`)
* spacing and case variants of the same code (`fc7` / `FC 7`)
* no punctuation, no question mark, two or three words

**Nothing here has a right answer attached.** It reports what resolved, what
came back, and flags the three shapes that are known to go wrong -- a code with
no family, an unresolved family, and a top-k that is really one document. It is
a place to look at a change's blast radius, not a score.

Exit codes:
    0 -- ran.
    1 -- no index, or retrieval failed.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

from seeley_rag.exceptions import SeeleyRagError
from seeley_rag.llm import active_provider
from seeley_rag.logging_conf import configure_logging
from seeley_rag.parse.base import UNKNOWN_FAMILY
from seeley_rag.retrieve.hybrid import RetrievalError, retrieve
from seeley_rag.retrieve.query import understand
from seeley_rag.retrieve.rerank import rerank_backend
from seeley_rag.settings import get_settings

#: How a trade worker types, grouped by the failure each group probes.
NOVICE_QUERIES: tuple[tuple[str, str], ...] = (
    # A code and nothing else. The highest-risk shape in the whole corpus: the
    # same code means different things on gas heating and evaporative cooling.
    ("fc7", "bare code"),
    ("FC 7", "bare code, spaced"),
    ("e4", "bare code"),
    ("fault code 2", "bare code, spelled out"),
    # Symptom only. The lexicon resolves nothing; retrieval is on its own.
    ("no hot air", "symptom only"),
    ("heater not heating", "symptom only"),
    ("unit keeps cutting out", "symptom only"),
    ("pump not working", "symptom only"),
    ("water not filling", "symptom only"),
    ("cooler smells", "symptom only"),
    ("flashing light on controller", "symptom only"),
    # Product named, but sloppily.
    ("tq wont light", "no apostrophe"),
    ("braemer heater no ignition", "misspelled brand"),
    ("gas presure tq5", "misspelled, suffixed model"),
    ("evap cooler not cooling", "abbreviation"),
    ("my breezair is blowing warm", "colloquial"),
    # Specification, asked the way it gets asked.
    ("how much gas pressure", "spec, no product"),
    ("what size gas line", "spec, no product"),
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run the cascade over the queries trade workers actually type.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("query", nargs="*", help="Ad-hoc query instead of the fixture set.")
    parser.add_argument("--top-k", type=int, default=5, help="Chunks shown per query.")
    parser.add_argument("--limit", type=int, default=None, help="Cap the number of queries.")
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Force the query rewrite off, whatever retrieve.use_query_llm says.",
    )
    parser.add_argument(
        "--flags-only",
        action="store_true",
        help="Print only the queries that tripped a flag.",
    )
    parser.add_argument("--verbose", action="store_true", help="Log at DEBUG level.")
    return parser.parse_args()


def flags(understanding: Any, result: dict[str, Any]) -> list[str]:
    """Return the known-bad shapes this query hit.

    None of these is necessarily wrong. Each is a shape that has produced a
    wrong answer before, so it is surfaced rather than left to be noticed.

    Args:
        understanding: The parsed query.
        result: The cascade result.

    Returns:
        Flag labels, empty when the query looks ordinary.
    """
    out: list[str] = []
    rows = result["results"]
    pinned = result["pinned_codes"]

    if understanding.fault_codes and understanding.product_family == UNKNOWN_FAMILY:
        # The shape that produced a confident answer about the wrong appliance.
        out.append("CODE-NO-FAMILY")
    if any(pin.ambiguous for pin in pinned):
        out.append(f"ambiguous-pins:{len(pinned)}")
    if understanding.product_family == UNKNOWN_FAMILY:
        out.append("family-unresolved")
    if rows:
        families = {str(row.get("product_family")) for row in rows}
        if len(families) > 2:
            # Retrieval could not decide which product line this is about.
            out.append(f"mixed-families:{len(families)}")
        docs = {str(row.get("doc_id") or row.get("title")) for row in rows}
        if len(docs) == 1:
            out.append("single-document")
    else:
        out.append("NO-RESULTS")
    return out


def show(query: str, label: str, top_k: int, use_llm: bool | None) -> list[str]:
    """Run one query and print what happened.

    Args:
        query: The question as typed.
        label: What shape this query probes.
        top_k: Chunks to show.
        use_llm: Whether the rewrite may run.

    Returns:
        The flags raised.
    """
    parsed = understand(query, use_llm=use_llm)
    started = time.perf_counter()
    result = retrieve(query, top_k=top_k, understanding=parsed)
    elapsed = time.perf_counter() - started
    raised = flags(parsed, result)

    print(f"\n{'=' * 78}\nQ: {query!r}  [{label}]  {elapsed:.1f}s")
    print(
        f"   family={parsed.product_family} models={parsed.model_series} "
        f"codes={parsed.fault_codes} intent={parsed.intent} pinned={len(result['pinned_codes'])}"
    )
    if parsed.rewritten_query and parsed.rewritten_query != query:
        print(f"   rewrite: {parsed.rewritten_query[:100]}")
    if raised:
        print(f"   FLAGS: {', '.join(raised)}")
    for pin in result["pinned_codes"]:
        marker = "?" if pin.ambiguous else ("!" if pin.cross_family else "=")
        print(f"   {marker} pin [{pin.row.product_family}] {str(pin.row.meaning)[:58]}")
    for number, row in enumerate(result["results"], start=1):
        page = row.get("page_range") or row.get("page_label") or "-"
        print(
            f"   {number}. [{row.get('product_family')}] " f"{str(row.get('title'))[:58]} p.{page}"
        )
    return raised


def main() -> int:
    """Run the novice-query fixture.

    Returns:
        Process exit code.
    """
    args = parse_args()
    configure_logging(level="DEBUG" if args.verbose else "ERROR")

    typed = " ".join(args.query).strip()
    cases = [(typed, "ad hoc")] if typed else list(NOVICE_QUERIES)[: args.limit]
    use_llm = False if args.no_llm else None
    settings = get_settings().retrieve
    rewrite = "off (forced)" if args.no_llm else ("on" if settings.use_query_llm else "off")

    print(
        f"Novice queries: {len(cases)} | rewrite {rewrite} | rerank[{rerank_backend()}] "
        f"| provider={active_provider()}"
    )
    print(
        "Nothing here is scored. FLAGS mark shapes known to go wrong: a code with no\n"
        "product family, an unresolved family, a top-k that is really one document."
    )

    tally: dict[str, int] = {}
    try:
        for query, label in cases:
            raised = show(query, label, args.top_k, use_llm)
            if args.flags_only and not raised:
                continue
            for flag in raised:
                tally[flag.split(":")[0]] = tally.get(flag.split(":")[0], 0) + 1
    except RetrievalError as exc:
        print(f"Retrieval failed: {exc}")
        return 1
    except SeeleyRagError as exc:
        print(f"Failed: {exc}")
        return 1

    print(f"\n{'=' * 78}\nFlags over {len(cases)} queries:")
    for flag, count in sorted(tally.items(), key=lambda kv: -kv[1]):
        print(f"  {count:>3}  {flag}")
    if not tally:
        print("  none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
