"""Stage 2a -- PDF parsing.

build-plan.md sections 4.2 (three-tier parsing) and 4.3 (page images).

Per page: text, tables, a 150 DPI PNG, and the printed page label.

Two constraints shape the implementation, and both are about not wasting hours:

* **``find_tables()`` is edge-detection-heavy at 0.5-2s per page.** Across the
  12,526 pages in this corpus that is 1.7 to 7 hours on its own. It is gated
  behind :func:`has_table_signal`, a cheap text check, so it only runs on pages
  that plausibly contain a table.
* **Every page gets a PNG, regardless of tier.** That is what makes "show me the
  wiring diagram" work for free: the diagram is on the page being cited. It is
  also the input the vision tiers will need, so rendering now means adding
  vision later costs no re-parse.

Scanned and diagram-heavy pages are recorded with ``needs_vision=True`` and
whatever text they do have. Nothing is silently dropped; the work is queued.

Output: ``data/01_interim/pages.jsonl`` and ``data/01_interim/page_images/``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import fitz

from seeley_rag.acquire.manifest import Document
from seeley_rag.exceptions import ParseError
from seeley_rag.logging_conf import get_logger
from seeley_rag.parse.base import (
    Page,
    Table,
    resolve_doc_type,
    resolve_model_series,
    resolve_product_family,
)
from seeley_rag.parse.pagelabels import read_embedded_label, resolve_label_with_source
from seeley_rag.parse.triage import classify_page
from seeley_rag.paths import page_image_path, relative_to_root
from seeley_rag.settings import REPO_ROOT, Settings, get_settings

log = get_logger(__name__)

#: Two or more consecutive spaces: the signature of column alignment in
#: extracted text.
_COLUMN_GAP = re.compile(r"\S {2,}\S")

#: Fault-code shapes from config/models.yaml. A page mentioning one is very
#: likely to carry the fault-code table, which is the content we least want to
#: miss.
_CODE_HINT = re.compile(
    r"\b(?:[EFH][\s:.\-]?\d{1,2}|FC\s?\d{1,2}|fault\s+code|error\s+code)\b",
    re.IGNORECASE,
)

#: Minimum column-aligned lines before a page is worth the table-detection cost.
MIN_TABLE_LINES = 3


def has_table_signal(text: str) -> bool:
    """Cheap pre-check gating the expensive ``find_tables()`` call.

    Two independent signals, either sufficient: at least
    :data:`MIN_TABLE_LINES` lines showing column alignment, or a fault-code
    mention. The second exists because fault-code tables are the highest-value
    tables in the corpus and are sometimes laid out without wide column gaps --
    missing one would be far more costly than the extra detection pass.

    Args:
        text: The page's extracted text.

    Returns:
        True if the page plausibly contains a table.
    """
    if not text:
        return False
    if _CODE_HINT.search(text):
        return True
    aligned = sum(1 for line in text.splitlines() if _COLUMN_GAP.search(line))
    return aligned >= MIN_TABLE_LINES


def _looks_like_header(row: list[str]) -> bool:
    """Whether a table row reads like a header rather than data.

    Args:
        row: The candidate row's cells.

    Returns:
        True if the row is non-empty and contains no bare numbers.
    """
    cells = [c.strip() for c in row if c and c.strip()]
    if not cells:
        return False
    numeric = sum(1 for c in cells if c.replace(".", "", 1).isdigit())
    return numeric == 0


def extract_tables(page: fitz.Page) -> list[Table]:
    """Detect tables on a page.

    Callers should gate this behind :func:`has_table_signal`.

    Args:
        page: The page to inspect.

    Returns:
        Detected tables. Empty on any detection failure -- a malformed table
        must not cost the page's text.
    """
    tables: list[Table] = []
    try:
        finder = page.find_tables()
    except Exception as exc:  # noqa: BLE001 - PyMuPDF raises bare Exception subclasses
        log.warning("table detection failed", extra={"error": str(exc)})
        return tables

    for found in getattr(finder, "tables", []):
        try:
            rows = [[(cell or "") for cell in row] for row in found.extract()]
        except Exception as exc:  # noqa: BLE001
            log.warning("table extraction failed", extra={"error": str(exc)})
            continue
        if not rows:
            continue
        tables.append(
            Table(
                rows=rows,
                has_header=_looks_like_header(rows[0]),
                bbox=tuple(round(v, 2) for v in found.bbox) if found.bbox else None,
                n_columns=max(len(r) for r in rows),
            )
        )
    return tables


def render_page_png(page: fitz.Page, doc_id: str, page_index: int, dpi: int = 150) -> str | None:
    """Render one page to a PNG.

    Args:
        page: The page to render.
        doc_id: The document's SHA-256, used to shard the output directory.
        page_index: 0-based page index.
        dpi: Render resolution. 150 gives ~150-250 KB per page.

    Returns:
        The repo-relative image path, or ``None`` if rendering failed. A failed
        render is not fatal: the page's text is still worth having.
    """
    target = page_image_path(doc_id, page_index)
    if target.exists():
        return relative_to_root(target)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        pixmap = page.get_pixmap(dpi=dpi)
        pixmap.save(target)
        return relative_to_root(target)
    except Exception as exc:  # noqa: BLE001 - PyMuPDF raises bare Exception subclasses
        log.warning(
            "page render failed",
            extra={"doc_id": doc_id, "page_index": page_index, "error": str(exc)},
        )
        return None


def parse_pdf(
    document: Document,
    render_images: bool = True,
    settings: Settings | None = None,
) -> list[Page]:
    """Parse one document into per-page records.

    Args:
        document: A deduplicated document from
            :func:`seeley_rag.acquire.manifest.unique_documents`. Parsing the
            document rather than the article is what stops a shared manual being
            parsed once per linking article.
        render_images: Whether to render a PNG per page.
        settings: Settings override, for tests.

    Returns:
        One :class:`Page` per page, in document order.

    Raises:
        ParseError: If the file is missing, or is not a PDF.
    """
    resolved = settings or get_settings()
    path = REPO_ROOT / document.stored_path
    if not path.exists():
        path = Path(document.stored_path)
    if not path.exists():
        raise ParseError(f"Document file missing: {document.stored_path}")

    with path.open("rb") as handle:
        if not handle.read(5).startswith(b"%PDF"):
            raise ParseError(
                f"{document.stored_path} is not a PDF. Acquisition stores whatever the "
                "portal served; check the file type before parsing."
            )

    title = document.titles[0] if document.titles else document.primary_filename
    category = document.categories[0] if document.categories else ""
    folder = document.folders[0] if document.folders else ""
    family = resolve_product_family(category, folder, f"{title} {document.primary_filename}")
    doc_type = resolve_doc_type(folder, document.primary_filename)
    model_series = resolve_model_series(title, document.primary_filename)

    pages: list[Page] = []
    try:
        pdf = fitz.open(path)
    except Exception as exc:  # noqa: BLE001
        raise ParseError(f"Could not open {document.stored_path}: {exc}") from exc

    try:
        for index, page in enumerate(pdf):
            text = page.get_text("text")
            stripped = text.strip()
            images = page.get_images()
            _, _, tier = classify_page(len(stripped), len(images), resolved)

            tables = extract_tables(page) if has_table_signal(text) else []
            label, label_source = resolve_label_with_source(text, index, read_embedded_label(page))
            image_path = (
                render_page_png(page, document.sha256, index, resolved.parse.render_dpi)
                if render_images
                else None
            )

            pages.append(
                Page(
                    doc_id=document.sha256,
                    page_index=index,
                    page_label=label,
                    label_source=label_source,
                    text=stripped,
                    tables=tables,
                    tier=tier,
                    needs_vision=tier in ("scanned", "diagram_heavy"),
                    image_path=image_path,
                    source_article_ids=list(document.article_ids),
                    product_family=family,
                    doc_type=doc_type,
                    model_series=model_series,
                    title=title,
                    source_url=(
                        f"/helpdesk/attachments/{document.attachment_ids[0]}"
                        if document.attachment_ids
                        else ""
                    ),
                    category=category,
                    folder=folder,
                    content_stream="pdf",
                )
            )
    finally:
        pdf.close()

    log.info(
        "parsed document",
        extra={
            "doc_id": document.sha256[:16],
            "pages": len(pages),
            "needs_vision": sum(1 for p in pages if p.needs_vision),
            "tables": sum(len(p.tables) for p in pages),
        },
    )
    return pages


def parse_document_to_dicts(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse one document, taking and returning plain dicts.

    A module-level function with picklable arguments, so it can be the target of
    a ``ProcessPoolExecutor``. Pydantic models survive pickling, but dicts keep
    the worker boundary boring.

    Args:
        payload: ``{"document": <Document as dict>, "render_images": bool}``.

    Returns:
        Page records as dicts, or an empty list if the document could not be
        parsed. Failures are logged, never raised: one bad manual must not end a
        parse over 544 documents.
    """
    document = Document(**payload["document"])
    try:
        pages = parse_pdf(document, render_images=payload.get("render_images", True))
    except ParseError as exc:
        log.error(
            "document parse failed",
            extra={"doc_id": document.sha256[:16], "error": str(exc)},
        )
        return []
    return [p.model_dump(mode="json") for p in pages]
