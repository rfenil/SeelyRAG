#!/usr/bin/env python
"""Stage 0 gate -- may we crawl the portal at all?

build-plan.md section 3.0. **The first action of the project.**

With no Freshdesk API key available, the public crawl is the only acquisition
path. If ``robots.txt`` disallows the solution paths, the acquisition stage is
dead and no amount of engineering fixes it -- escalate to a human, who must
obtain an API key, a bulk PDF export, or written permission to crawl.

Exit codes:
    0 -- every required path is crawlable.
    1 -- a required path is disallowed. The project is blocked.
    2 -- the verdict could not be determined (network failure, odd status).
"""

from __future__ import annotations

import argparse
import sys

from seeley_rag.acquire.robots import RobotsGate
from seeley_rag.exceptions import AcquisitionError, RobotsDisallowedError
from seeley_rag.logging_conf import configure_logging
from seeley_rag.settings import get_settings


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Check whether the Seeley help centre permits our crawl.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-url", default=None, help="Portal origin to check.")
    parser.add_argument("--user-agent", default=None, help="User-Agent to evaluate rules against.")
    parser.add_argument("--show-robots", action="store_true", help="Print robots.txt verbatim.")
    parser.add_argument(
        "--log-format", choices=("json", "console"), default="console", help="Log output format."
    )
    return parser.parse_args()


def main() -> int:
    """Run the gate and print a clear verdict.

    Returns:
        A process exit code.
    """
    args = parse_args()
    configure_logging(fmt=args.log_format)
    settings = get_settings()

    gate = RobotsGate(base_url=args.base_url, user_agent=args.user_agent)
    print(f"Checking {gate.robots_url}")
    print(f"User-Agent: {gate.user_agent}")
    print()

    try:
        report = gate.report()
    except AcquisitionError as exc:
        print("VERDICT: UNDETERMINED")
        print()
        print(str(exc))
        print()
        print("Do not crawl on an undetermined verdict. Resolve and re-run.")
        return 2

    if args.show_robots:
        print("--- robots.txt ---")
        print(report.raw.strip() or "(empty -- nothing is disallowed)")
        print("------------------")
        print()

    print("Required paths:")
    for path, allowed in report.results.items():
        print(f"  [{'ALLOWED' if allowed else 'BLOCKED'}] {path}")
    print()

    if report.crawl_delay is not None:
        print(f"robots.txt requests a Crawl-delay of {report.crawl_delay}s.")
        configured = settings.crawl.delay_seconds
        if report.crawl_delay > configured:
            print(
                f"  Our configured delay is {configured}s. Slow the crawl to at least "
                f"{report.crawl_delay}s: set crawl.rps to {1.0 / report.crawl_delay:.3f} "
                "in config/config.yaml before running the acquire script."
            )
        print()

    try:
        gate.assert_crawlable()
    except RobotsDisallowedError as exc:
        print("VERDICT: BLOCKED -- DO NOT CRAWL")
        print()
        print(str(exc))
        return 1

    print("VERDICT: ALLOWED")
    print()
    print("The crawl may proceed. Next: python scripts/02_acquire.py --limit 3 --dry-run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
