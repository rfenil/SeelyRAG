#!/usr/bin/env python
"""Stage 0 -- PDF corpus triage.

build-plan.md section 4.1. Twenty minutes that set the cost and time budget for
the next two days.

Run it before a scraper exists: hand-download six representative manuals from a
browser, spanning the date range (a 2023 guide, a 2015 one, a 2005 one), and
point this at them. After a crawl, point it at ``data/00_raw/pdf/``.

Reports **all three fractions** -- scanned, diagram-heavy, plain text. Modelling
only the scanned fraction under-budgets vision by 2-3x.

Exit codes:
    0 -- triage completed.
    1 -- no readable PDFs, or every document failed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from seeley_rag.exceptions import ParseError
from seeley_rag.logging_conf import configure_logging
from seeley_rag.parse.triage import triage_corpus, write_report
from seeley_rag.paths import RAW_PDF_DIR


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Triage PDFs into plain-text, diagram-heavy and scanned tiers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "pdfs",
        nargs="*",
        help="PDF files or directories. Defaults to data/00_raw/pdf/.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Triage at most this many PDFs.")
    parser.add_argument(
        "--log-format", choices=("json", "console"), default="console", help="Log output format."
    )
    return parser.parse_args()


def collect_pdfs(inputs: list[str], limit: int | None) -> list[Path]:
    """Expand the command-line arguments into a list of PDF paths.

    Args:
        inputs: Files or directories. Empty means ``data/00_raw/pdf/``.
        limit: Maximum number of PDFs to return.

    Returns:
        Sorted PDF paths.
    """
    roots = [Path(item) for item in inputs] if inputs else [RAW_PDF_DIR]
    found: list[Path] = []
    for root in roots:
        if root.is_dir():
            found.extend(sorted(root.rglob("*.pdf")))
        elif root.suffix.lower() == ".pdf" and root.exists():
            found.append(root)
    # A path may arrive twice via both a directory and an explicit file.
    unique = sorted({p.resolve() for p in found})
    return unique[:limit] if limit else unique


def main() -> int:
    """Run the triage and write a report.

    Returns:
        A process exit code.
    """
    args = parse_args()
    configure_logging(fmt=args.log_format)

    pdfs = collect_pdfs(args.pdfs, args.limit)
    if not pdfs:
        where = ", ".join(args.pdfs) if args.pdfs else str(RAW_PDF_DIR)
        print(f"No PDFs found in: {where}")
        print()
        print("Hand-download six representative manuals spanning the date range")
        print("(e.g. a 2023, a 2015 and a 2005 service guide) and pass them here:")
        print("  python scripts/01_triage.py path/to/*.pdf")
        return 1

    print(f"Triaging {len(pdfs)} PDF(s)...")
    try:
        documents, summary = triage_corpus(pdfs)
    except ParseError as exc:
        print(str(exc))
        return 1

    if summary.documents == 0:
        print(f"Every document failed to open ({summary.failed_documents} attempted).")
        return 1

    print()
    print("The three fractions")
    print("-------------------")
    print(
        f"  plain text     {summary.plain_text_pages:>6}  "
        f"{summary.pct_plain_text:>7.1%}   no vision"
    )
    print(
        f"  diagram-heavy  {summary.diagram_heavy_pages:>6}  {summary.pct_diagram_heavy:>7.1%}"
        "   vision: caption"
    )
    print(
        f"  scanned        {summary.scanned_pages:>6}  {summary.pct_scanned:>7.1%}"
        "   vision: full transcription"
    )
    print(f"  {'':<14} {'':>6}  {'':>7}")
    print(f"  total pages    {summary.total_pages:>6}")
    print(f"  needs vision   {'':>6}  {summary.pct_vision:>7.1%}")
    print()

    if summary.pct_vision > 0.5:
        print("BUDGET WARNING: more than half of all pages need a vision call.")
        print("  Build-plan section 13, risk 2 applies. Consider restricting the pilot")
        print("  to post-2013 manuals and documenting the coverage gap, or expect cost")
        print("  and Day 1 machine time to overrun by roughly 3x.")
        print()

    print(f"Page labels: {summary.pages_with_labels}/{summary.total_pages} pages expose one.")
    if summary.label_offsets:
        print(f"  Front-matter offsets observed: {summary.label_offsets}")
        print("  Cite the printed label, never the index (build-plan section 4.5).")
    if summary.failed_documents:
        print(f"Failed to open: {summary.failed_documents} document(s).")
    print()

    report_path = write_report(documents, summary)
    print(f"Report written to {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
