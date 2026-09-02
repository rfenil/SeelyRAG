#!/usr/bin/env python
"""Stage 1 -- crawl the portal and build the manifest.

build-plan.md section 3.

Walks the selected categories' folders, fetches every article, downloads and
deduplicates the attached manuals, and writes ``data/00_raw/manifest.jsonl``.

The crawl is polite by construction: 1 req/sec, single-threaded, honest
User-Agent, every fetch cached to disk, and an immediate stop on 429 or 403. It
calls the robots gate before its first fetch and refuses to run if the gate
fails.

Re-running is cheap and safe. ``data/00_raw`` is write-once and content-
addressed, so a second run serves pages from the HTML cache and skips PDFs it
already holds.

Exit codes:
    0 -- crawl completed (validation warnings may still be printed).
    1 -- the crawl failed or was blocked.
    2 -- the robots gate refused the crawl.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

from seeley_rag.acquire.attachments import AttachmentDownloader
from seeley_rag.acquire.manifest import (
    CrawlProgress,
    ManifestWriter,
    compact,
    load_progress,
    summarise,
    validate,
)
from seeley_rag.acquire.portal import PortalScraper
from seeley_rag.acquire.robots import RobotsGate
from seeley_rag.exceptions import (
    AcquisitionError,
    ManifestError,
    RateLimitedError,
    RobotsDisallowedError,
)
from seeley_rag.logging_conf import configure_logging, get_logger
from seeley_rag.paths import MANIFEST_PATH, REPORTS_DIR, crawl_report_path, ensure_dirs
from seeley_rag.settings import get_settings

log = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed arguments.
    """
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Crawl the Seeley help centre and build the acquisition manifest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        default=settings.pilot_categories,
        help="Case-insensitive substrings matched against category names. "
        "Pass --categories with no values to crawl everything.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Stop after N articles.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the folders and articles that would be crawled, then stop. "
        "Downloads nothing and writes no manifest.",
    )
    parser.add_argument(
        "--rps",
        type=float,
        default=None,
        help="Requests per second. Raising this is not a supported way to go faster.",
    )
    parser.add_argument(
        "--no-attachments", action="store_true", help="Skip PDF downloads; metadata only."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Start over: truncate the manifest and re-acquire everything. "
        "Without this, a run RESUMES from the existing manifest.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Append without skipping work an earlier run completed. Rarely "
        "wanted; it re-fetches pages and PDFs you already hold.",
    )
    parser.add_argument(
        "--skip-robots",
        action="store_true",
        help="Skip the robots gate. Only for offline tests against a cache.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print a running total every N articles.",
    )
    parser.add_argument(
        "--log-format", choices=("json", "console"), default="console", help="Log output format."
    )
    return parser.parse_args()


def run_dry(scraper: PortalScraper, categories: list[str], limit: int | None) -> int:
    """Print the crawl plan without fetching articles or attachments.

    Listing folders and article links still reads the portal (or, more usually,
    the HTML cache), but nothing is downloaded and no manifest is written.

    Args:
        scraper: The configured scraper.
        categories: Category substrings to select.
        limit: Article cap.

    Returns:
        A process exit code.
    """
    folders = scraper.select_folders(categories)
    print(f"Selected {len(folders)} folder(s) for categories: {categories or 'ALL'}")
    print()

    total = 0
    planned = 0
    for folder in folders:
        links = scraper.list_articles(folder["id"])
        total += len(links)
        remaining = None if limit is None else max(0, limit - planned)
        take = len(links) if remaining is None else min(len(links), remaining)
        planned += take
        print(f"  [{folder['category']}]")
        print(f"    {folder['name']} ({folder['id']}): {len(links)} article(s), would fetch {take}")
        for link in links[:take][:3]:
            print(f"      - {link['id']}  {link['title'][:70]}")
        if take > 3:
            print(f"      ... and {take - 3} more")

    scraped = scraper.stats()
    print()
    print("Dry run summary")
    print("---------------")
    print(f"  folders                {len(folders)}")
    print(f"  articles found         {total}")
    print(f"  articles to fetch      {planned}")
    print(f"  listing requests sent  {scraped['requests']} (cache hits: {scraped['cache_hits']})")
    print(f"  listing HTML pulled    {human_bytes(scraped['html_bytes_fetched'])}")
    print(f"  listing HTML cached    {human_bytes(scraped['html_bytes_from_cache'])}")
    print()
    print("A real run would additionally fetch one page per article and download")
    print(f"its attachments -- roughly {planned} more requests at 1 req/sec, so about")
    print(f"{planned / 60:.0f} minutes for the article pages alone.")
    print()
    print("Nothing was downloaded and no manifest was written.")
    print("Re-run without --dry-run to acquire.")
    return 0


def check_gate(skip: bool) -> int:
    """Run the robots gate before the first fetch.

    Args:
        skip: Whether the caller asked to bypass the gate.

    Returns:
        0 to proceed, 2 to abort.
    """
    if skip:
        return 0
    try:
        RobotsGate().assert_crawlable()
    except RobotsDisallowedError as exc:
        print("BLOCKED: robots.txt forbids this crawl.")
        print()
        print(str(exc))
        return 2
    except AcquisitionError as exc:
        print(f"Could not determine the robots verdict: {exc}")
        print("Refusing to crawl on an undetermined verdict.")
        return 2
    return 0


def human_bytes(count: int) -> str:
    """Format a byte count for a human reading a terminal.

    Args:
        count: Number of bytes.

    Returns:
        e.g. ``2.0 MB``.
    """
    size = float(count)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} GB"


def progress_line(
    written: int, scraper: PortalScraper, downloader: AttachmentDownloader | None
) -> str:
    """Build the one-line running total shown during the crawl.

    Args:
        written: Articles written to the manifest so far.
        scraper: The scraper, for HTML volume and request counts.
        downloader: The downloader, for PDF volume. ``None`` when skipping PDFs.

    Returns:
        A single status line.
    """
    scraped = scraper.stats()
    parts = [
        f"{written:>4} articles",
        f"{scraped['requests']:>4} req ({scraped['cache_hits']} cached)",
        f"html {human_bytes(scraped['html_bytes_total']):>9}",
    ]
    if downloader is not None:
        pdf = downloader.summary()
        parts.append(f"pdf {human_bytes(pdf['bytes_stored']):>9}")
        parts.append(f"{pdf['unique_documents']:>3} docs ({pdf['deduplicated']} dup)")
    return "  " + " | ".join(parts)


def run_crawl(
    scraper: PortalScraper,
    args: argparse.Namespace,
    categories: list[str],
    progress: CrawlProgress,
) -> tuple[int, AttachmentDownloader | None]:
    """Walk the selected folders, download attachments, and write the manifest.

    Prints a running total every few articles so a 25-minute unattended crawl is
    observable rather than silent.

    Args:
        scraper: The configured scraper.
        args: Parsed command-line arguments.
        categories: Category substrings to select.
        progress: What an earlier run already completed.

    Returns:
        The number of articles written, and the downloader (for the final
        report) or ``None`` if attachments were skipped.

    Raises:
        RateLimitedError: If the portal blocks us.
        AcquisitionError: On any other fetch failure.
    """
    # One limiter for the whole run. Page fetches and PDF downloads both go
    # through it, so 1 rps is true of the run rather than of each component.
    downloader = None if args.no_attachments else AttachmentDownloader(limiter=scraper.limiter)
    written = 0
    try:
        print("Crawling. Running totals:")
        with ManifestWriter(overwrite=args.overwrite) as manifest:
            for article in scraper.iter_articles(
                categories or None,
                limit=args.limit,
                skip_article_ids=progress.article_ids,
            ):
                if downloader is not None and article.attachments:
                    article.attachments = downloader.download_all(article.attachments)
                manifest.write(article)
                written += 1
                if written % args.progress_every == 0:
                    print(progress_line(written, scraper, downloader), flush=True)
        if written % args.progress_every != 0:
            print(progress_line(written, scraper, downloader), flush=True)
    finally:
        if downloader is not None:
            downloader.close()
    return written, downloader


def print_volume_report(
    written: int, scraper: PortalScraper, downloader: AttachmentDownloader | None
) -> None:
    """Print how much data the crawl actually pulled.

    Separates network bytes from cache bytes, and fetched bytes from stored
    bytes. Both splits answer questions you will actually ask: "did the re-run
    hit the network at all?" and "how much did deduplication save?".

    Args:
        written: Articles written to the manifest.
        scraper: The scraper.
        downloader: The downloader, or ``None``.
    """
    scraped = scraper.stats()
    print()
    print("Data scraped this run")
    print("---------------------")
    print(f"  articles parsed          {written}")
    if scraped["articles_skipped"]:
        print(f"  articles skipped (resume) {scraped['articles_skipped']}")
    print(f"  folders listed           {scraped['folders_listed']}")
    print(f"  HTTP requests sent       {scraped['requests']}")
    print(f"  served from cache        {scraped['cache_hits']}")
    print(f"  HTML over the network    {human_bytes(scraped['html_bytes_fetched'])}")
    print(f"  HTML from cache          {human_bytes(scraped['html_bytes_from_cache'])}")
    print(f"  HTML total               {human_bytes(scraped['html_bytes_total'])}")

    if downloader is None:
        print("  PDFs                     skipped (--no-attachments)")
        return

    pdf = downloader.summary()
    print(f"  PDFs downloaded          {pdf['downloaded']}")
    print(f"  PDFs deduplicated        {pdf['deduplicated']}")
    print(f"  unique documents         {pdf['unique_documents']}")
    print(f"  PDF bytes over the wire  {human_bytes(pdf['bytes_fetched'])}")
    print(f"  PDF bytes stored on disk {human_bytes(pdf['bytes_stored'])}")
    print(f"  saved by deduplication   {human_bytes(pdf['bytes_saved_by_dedupe'])}")
    if pdf["skipped_resumed"]:
        print(f"  PDFs skipped (resume)    {pdf['skipped_resumed']}")
        print(f"  saved by resume          {human_bytes(pdf['bytes_skipped_by_resume'])}")
    if pdf["failed"]:
        print(f"  FAILED downloads         {pdf['failed']}")
    total = scraped["html_bytes_fetched"] + pdf["bytes_fetched"]
    print(f"  TOTAL pulled from portal {human_bytes(total)}")


def write_crawl_report(
    written: int, scraper: PortalScraper, downloader: AttachmentDownloader | None
) -> Path:
    """Write a durable record of what this run pulled.

    The terminal output scrolls away; this does not. It is the audit trail for
    "how much of Seeley's site did we actually take, and when".

    Args:
        written: Articles written to the manifest.
        scraper: The scraper.
        downloader: The downloader, or ``None``.

    Returns:
        The report path.
    """
    timestamp = dt.datetime.now(tz=dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    scraped = scraper.stats()
    pdf = downloader.summary() if downloader else {}

    lines = [
        "# Crawl report",
        "",
        f"Run: {timestamp} (UTC)",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Articles parsed | {written} |",
        f"| Articles skipped (resume) | {scraped['articles_skipped']} |",
        f"| Folders listed | {scraped['folders_listed']} |",
        f"| HTTP requests sent | {scraped['requests']} |",
        f"| Served from cache | {scraped['cache_hits']} |",
        f"| HTML over the network | {human_bytes(scraped['html_bytes_fetched'])} |",
        f"| HTML from cache | {human_bytes(scraped['html_bytes_from_cache'])} |",
    ]
    if pdf:
        lines += [
            f"| PDFs downloaded | {pdf['downloaded']} |",
            f"| PDFs deduplicated | {pdf['deduplicated']} |",
            f"| Unique documents | {pdf['unique_documents']} |",
            f"| PDF bytes over the wire | {human_bytes(pdf['bytes_fetched'])} |",
            f"| PDF bytes stored | {human_bytes(pdf['bytes_stored'])} |",
            f"| Saved by deduplication | {human_bytes(pdf['bytes_saved_by_dedupe'])} |",
            f"| PDFs skipped (resume) | {pdf['skipped_resumed']} |",
            f"| Saved by resume | {human_bytes(pdf['bytes_skipped_by_resume'])} |",
            f"| Failed downloads | {pdf['failed']} |",
        ]
    lines += ["", "Generated by `scripts/02_acquire.py`.", ""]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = crawl_report_path(timestamp)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def report_manifest() -> int:
    """Print the manifest summary and validation result.

    Returns:
        A process exit code.
    """
    try:
        print(summarise().render())
    except ManifestError as exc:
        print(f"Could not summarise the manifest: {exc}")
        return 1

    print()
    problems = validate()
    if problems:
        print(f"Manifest validation found {len(problems)} problem(s):")
        for problem in problems[:20]:
            print(f"  - {problem}")
        if len(problems) > 20:
            print(f"  ... and {len(problems) - 20} more")
    else:
        print("Manifest validation: clean.")

    print()
    print(f"Manifest: {MANIFEST_PATH}")
    print("Next: python scripts/01_triage.py   (triage the PDFs you just acquired)")
    return 0


def resolve_progress(args: argparse.Namespace) -> CrawlProgress:
    """Work out what an earlier run already finished, and say so.

    Resuming is the default. A crawl of this corpus is a ~35-minute unattended
    run against someone else's server; making the safe behaviour opt-in would
    mean the common case is the wasteful one.

    Args:
        args: Parsed command-line arguments.

    Returns:
        The progress to resume from. Empty when starting fresh.
    """
    if args.overwrite:
        print("--overwrite: starting fresh, the existing manifest will be replaced.")
        print()
        return CrawlProgress()

    if args.no_resume:
        print("--no-resume: appending without skipping anything already acquired.")
        print()
        return CrawlProgress()

    progress = load_progress()
    if progress.is_empty:
        return progress

    if progress.needs_compaction:
        # Almost always a half-written final line from a killed process.
        dropped = compact()
        print(f"Repaired the manifest: dropped {dropped} unreadable row(s).")
        progress = load_progress()

    print("Resuming from the existing manifest.")
    print(f"  articles already acquired    {len(progress.article_ids)}")
    print(f"  attachments already on disk  {len(progress.attachments)}")
    print("  (these are skipped without re-fetching; pass --overwrite to start over)")
    print()
    return progress


def main() -> int:
    """Run the acquisition crawl.

    Returns:
        A process exit code.
    """
    args = parse_args()
    configure_logging(fmt=args.log_format)
    ensure_dirs()

    # The gate comes before the first fetch, always.
    gate = check_gate(args.skip_robots)
    if gate != 0:
        return gate

    categories = list(args.categories) if args.categories else []
    progress = resolve_progress(args)
    scraper = PortalScraper(rps=args.rps)

    try:
        if args.dry_run:
            return run_dry(scraper, categories, args.limit)
        written, downloader = run_crawl(scraper, args, categories, progress)
        print_volume_report(written, scraper, downloader)
        report_path = write_crawl_report(written, scraper, downloader)
        print(f"  crawl report             {report_path}")
        print()
    except RateLimitedError as exc:
        print("BLOCKED BY THE PORTAL -- the crawl stopped immediately.")
        print()
        print(str(exc))
        print()
        print("Everything acquired so far is already on disk. Re-run the same")
        print("command once the block clears and it resumes from where it stopped.")
        return 1
    except KeyboardInterrupt:
        print()
        print("Interrupted. Every article written so far is in the manifest;")
        print("re-run the same command to resume from where it stopped.")
        return 1
    except AcquisitionError as exc:
        print(f"Acquisition failed: {exc}")
        return 1
    finally:
        scraper.close()

    return report_manifest()


if __name__ == "__main__":
    sys.exit(main())
