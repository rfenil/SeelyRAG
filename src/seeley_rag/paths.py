"""Single source of truth for every path in the project.

No other module may contain a directory string literal. If you need a path, add
it here and import it. This is what makes a later move to object storage a
one-file change (ADR 0003).

DATA LAYOUT CONTRACT
--------------------
``data/00_raw/`` is **write-once**. Nothing in this codebase may modify or
delete a file underneath it once created. Every later stage reads from it and
writes elsewhere. Two things enforce that:

* No write helper for ``00_raw`` is exported outside :mod:`seeley_rag.acquire`.
* :func:`clean_derived` -- the only deletion helper in the project -- refuses to
  touch it.

Re-running the crawl is therefore always safe and always cheap: cached HTML and
content-addressed PDFs are simply found already present.

PDFs are content-addressed as ``data/00_raw/pdf/{sha256}.pdf``. The mapping from
Freshdesk attachment ID to hash lives in the manifest, not in the filename, so
deduplication is free and re-downloads are idempotent.

Cached HTML is ``data/00_raw/html/{sha1(url)}.html``, checked before every fetch.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from seeley_rag.settings import get_settings

# --------------------------------------------------------------------------
# Roots
# --------------------------------------------------------------------------
DATA_ROOT: Path = get_settings().resolved_data_root

# Stage 00 -- immutable, content-addressed. WRITE-ONCE. See the contract above.
RAW_DIR: Path = DATA_ROOT / "00_raw"
RAW_HTML_DIR: Path = RAW_DIR / "html"
RAW_PDF_DIR: Path = RAW_DIR / "pdf"
MANIFEST_PATH: Path = RAW_DIR / "manifest.jsonl"

# Stage 01 -- parsed pages and rendered page images. Stage 2, not yet built.
INTERIM_DIR: Path = DATA_ROOT / "01_interim"
PAGES_PATH: Path = INTERIM_DIR / "pages.jsonl"
PAGE_IMAGES_DIR: Path = INTERIM_DIR / "page_images"

# Stage 02 -- chunks and the fault-code table. Stage 3, not yet built.
PROCESSED_DIR: Path = DATA_ROOT / "02_processed"
CHUNKS_PATH: Path = PROCESSED_DIR / "chunks.jsonl"
CODES_PATH: Path = PROCESSED_DIR / "codes.jsonl"

# Stage 03 -- vector store. Stage 4, not yet built.
INDEX_DIR: Path = DATA_ROOT / "03_index"

# Caches -- expensive-call memoisation. Stages 2b and 4, not yet built.
CACHE_DIR: Path = DATA_ROOT / "cache"
LLM_CACHE_DIR: Path = CACHE_DIR / "llm"
EMBEDDING_CACHE_DIR: Path = CACHE_DIR / "embeddings"

# Reports -- triage and crawl summaries, written for humans.
REPORTS_DIR: Path = DATA_ROOT / "reports"

# Every answered question, appended as JSONL. build-plan section 9: the first
# week of real queries is worth more than any synthetic eval, so this is written
# from the first answer rather than added when someone thinks to.
QUERY_LOG_PATH: Path = REPORTS_DIR / "queries.jsonl"

# Feedback on answers, keyed by the same query_id. Separate from the query log
# so a rating never rewrites the record of what was actually retrieved and said.
FEEDBACK_LOG_PATH: Path = REPORTS_DIR / "feedback.jsonl"

#: Every directory ``ensure_dirs`` creates. Order is irrelevant; parents are made.
ALL_DIRS: tuple[Path, ...] = (
    RAW_DIR,
    RAW_HTML_DIR,
    RAW_PDF_DIR,
    INTERIM_DIR,
    PAGE_IMAGES_DIR,
    PROCESSED_DIR,
    INDEX_DIR,
    CACHE_DIR,
    LLM_CACHE_DIR,
    EMBEDDING_CACHE_DIR,
    REPORTS_DIR,
)

#: Stages ``clean`` may remove. ``RAW_DIR`` is deliberately absent.
DERIVED_DIRS: tuple[Path, ...] = (
    INTERIM_DIR,
    PROCESSED_DIR,
    INDEX_DIR,
    CACHE_DIR,
)


def ensure_dirs() -> None:
    """Create every data directory. Idempotent; called by ``make init``.

    Existing files are never touched.
    """
    for directory in ALL_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


def clean_derived() -> None:
    """Remove every derived stage, leaving ``data/00_raw`` untouched.

    This is the only deletion helper in the project, and it is deliberately
    incapable of removing raw data: re-acquiring costs a 25-minute polite crawl
    that we would rather not repeat, and the raw tree is the provenance root for
    every citation the system will ever emit.
    """
    for directory in DERIVED_DIRS:
        if directory == RAW_DIR or RAW_DIR in directory.parents:
            # Unreachable given DERIVED_DIRS, but the contract is worth asserting
            # in code rather than only in a comment.
            continue
        if directory.exists():
            shutil.rmtree(directory)
    ensure_dirs()


def html_cache_path(url: str) -> Path:
    """Return the disk-cache path for a fetched page.

    Args:
        url: Absolute URL that was, or will be, fetched.

    Returns:
        ``data/00_raw/html/{sha1(url)}.html``.
    """
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()
    return RAW_HTML_DIR / f"{digest}.html"


def raw_blob_path(sha256: str, suffix: str = ".pdf") -> Path:
    """Return the content-addressed storage path for an attachment.

    The suffix reflects what the bytes actually are, determined by sniffing the
    file's magic number rather than by trusting the portal. The corpus is
    overwhelmingly PDFs, but a handful of attachments are images or Office
    documents, and storing those as ``.pdf`` makes every later stage try to
    parse them as PDFs.

    Args:
        sha256: Hex SHA-256 of the file's bytes.
        suffix: File extension, leading dot included.

    Returns:
        ``data/00_raw/pdf/{sha256}{suffix}``.
    """
    return RAW_PDF_DIR / f"{sha256}{suffix}"


def pdf_path(sha256: str) -> Path:
    """Return the content-addressed storage path for a PDF.

    Args:
        sha256: Hex SHA-256 of the file's bytes.

    Returns:
        ``data/00_raw/pdf/{sha256}.pdf``.
    """
    return raw_blob_path(sha256, ".pdf")


def page_image_path(doc_id: str, page_index: int) -> Path:
    """Return the rendered-page image path.

    Sharded by document so a directory never holds 12,000 files.

    Args:
        doc_id: The document's SHA-256.
        page_index: 0-based page index.

    Returns:
        ``data/01_interim/page_images/{doc_id}/{page_index:04d}.png``.
    """
    return PAGE_IMAGES_DIR / doc_id / f"{page_index:04d}.png"


def relative_to_root(path: Path) -> str:
    """Render a path relative to the repository root, with forward slashes.

    Manifest rows store portable relative paths -- ``data/00_raw/pdf/a3f1....pdf``
    -- so a manifest stays readable when the checkout moves between machines or
    between Windows and Linux.

    Args:
        path: Absolute or relative path.

    Returns:
        A POSIX-style relative path string, or the POSIX form of the input if it
        lies outside the repository root.
    """
    from seeley_rag.settings import REPO_ROOT

    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def triage_report_path(timestamp: str) -> Path:
    """Return the path for a triage report.

    Args:
        timestamp: Compact UTC stamp, e.g. ``20260820T091422Z``.

    Returns:
        ``data/reports/triage_{timestamp}.md``.
    """
    return REPORTS_DIR / f"triage_{timestamp}.md"


def rerank_ab_report_path(timestamp: str) -> Path:
    """Return the path for a reranking A/B report.

    Args:
        timestamp: Compact UTC stamp, e.g. ``20260820T091422Z``.

    Returns:
        ``data/reports/rerank_ab_{timestamp}.md``.
    """
    return REPORTS_DIR / f"rerank_ab_{timestamp}.md"


def crawl_report_path(timestamp: str) -> Path:
    """Return the path for a crawl summary report.

    Args:
        timestamp: Compact UTC stamp, e.g. ``20260820T091422Z``.

    Returns:
        ``data/reports/crawl_{timestamp}.md``.
    """
    return REPORTS_DIR / f"crawl_{timestamp}.md"
