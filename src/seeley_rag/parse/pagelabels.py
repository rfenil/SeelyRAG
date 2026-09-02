"""Printed page-label reconciliation.

build-plan.md section 4.5.

``enumerate(doc)`` is 0-based. Citations render "p.42". SMEs write
``expected_page: 42`` by reading the **printed** number off the page. Service
manuals have front matter, so printed and index routinely differ.

This is not hypothetical for this corpus. Triage over 12,526 pages found 3,608
pages carrying an embedded label, with observed offsets of -4, -3, -2, -1, 0 and
+1 -- most commonly **-2**. A ``+/-1`` eval tolerance hides an off-by-one but not
a front-matter offset, so page accuracy would fail corpus-wide for a reason that
looks exactly like a retrieval bug.

Resolution order, most to least trustworthy:

1. ``page.get_label()`` -- the PDF's own page-label tree.
2. A footer or header regex over the page text.
3. ``page_index + 1``, recorded as a fallback rather than presented as truth.
"""

from __future__ import annotations

import re
from collections import Counter

import fitz

from seeley_rag.logging_conf import get_logger

log = get_logger(__name__)

#: Patterns for a printed page number, applied to the last and first few lines.
#: Ordered most to least specific -- "Page 42 of 118" is far stronger evidence
#: than a bare number on its own line.
FOOTER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bpage\s+(\d{1,4})\s+of\s+\d{1,4}\b", re.IGNORECASE),
    re.compile(r"\bpage\s+(\d{1,4})\b", re.IGNORECASE),
    re.compile(r"^\s*[-–—]\s*(\d{1,4})\s*[-–—]\s*$"),
    re.compile(r"^\s*(\d{1,4})\s*$"),
)

#: How many lines from each end of the page to inspect. Page numbers live in
#: running headers and footers, never in the middle of body text.
EDGE_LINES = 3

#: Beyond this, a "page number" is almost certainly a part number or a year.
MAX_PLAUSIBLE_LABEL = 2000


def read_embedded_label(page: fitz.Page) -> str | None:
    """Return the PDF's own page label, if it declares one.

    ``page.get_label()`` raises a bare ``AssertionError`` from inside PyMuPDF
    when the document cannot be viewed as a PDF, and other internal errors on a
    malformed label tree. Neither should end a parse over hundreds of manuals.

    Args:
        page: The page to read.

    Returns:
        The declared label, or ``None``.
    """
    try:
        label = page.get_label()
    except (AssertionError, RuntimeError, ValueError):
        return None
    return label or None


def label_from_text(text: str) -> str | None:
    """Find a printed page number in a page's header or footer.

    Only the outer few lines are considered. A number in the middle of body text
    is a measurement, a part number or a fault code -- never the page number.

    Args:
        text: The page's extracted text.

    Returns:
        The printed number as a string, or ``None``.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None

    edges = lines[-EDGE_LINES:] + lines[:EDGE_LINES]
    for pattern in FOOTER_PATTERNS:
        for line in edges:
            match = pattern.search(line)
            if match:
                value = match.group(1)
                if value.isdigit() and 0 < int(value) <= MAX_PLAUSIBLE_LABEL:
                    return str(int(value))
    return None


def resolve_label(text: str, page_index: int, embedded_label: str | None) -> str:
    """Resolve a page's printed label.

    Args:
        text: The page's extracted text, for the footer-regex fallback.
        page_index: 0-based index, the last-resort fallback.
        embedded_label: ``page.get_label()``, if the PDF declares one.

    Returns:
        The printed page label. Never empty -- callers always get something
        citable, and :func:`label_is_inferred` says how much to trust it.
    """
    if embedded_label:
        return embedded_label
    from_text = label_from_text(text)
    if from_text:
        return from_text
    return str(page_index + 1)


def resolve_label_with_source(
    text: str, page_index: int, embedded_label: str | None
) -> tuple[str, str]:
    """Resolve a page's printed label and report where it came from.

    Args:
        text: The page's extracted text.
        page_index: 0-based index, the last-resort fallback.
        embedded_label: ``page.get_label()``, if the PDF declares one.

    Returns:
        ``(label, source)`` where source is ``embedded``, ``text`` or ``index``.
    """
    if embedded_label:
        return embedded_label, "embedded"
    from_text = label_from_text(text)
    if from_text:
        return from_text, "text"
    return str(page_index + 1), "index"


def label_is_inferred(text: str, embedded_label: str | None) -> bool:
    """Whether a label had to be guessed from the index.

    Worth recording: a corpus where most labels are inferred cannot support the
    page-accuracy gate, and it is better to know that before an eval run than to
    conclude retrieval is broken.

    Args:
        text: The page's extracted text.
        embedded_label: ``page.get_label()`` output.

    Returns:
        True if neither the PDF nor the page text supplied a label.
    """
    return not embedded_label and label_from_text(text) is None


def detect_offset(document: fitz.Document) -> int | None:
    """Detect a document's front-matter offset.

    Args:
        document: An open PDF.

    Returns:
        The most common ``label - (index + 1)``, or ``None`` when no page
        carries a numeric label. A non-zero value is front matter, and it is the
        reason citations must use the label rather than the index.
    """
    offsets: Counter[int] = Counter()
    for index, page in enumerate(document):
        label = read_embedded_label(page)
        if label is None:
            label = label_from_text(page.get_text("text"))
        if label and label.isdigit():
            offsets[int(label) - (index + 1)] += 1
    if not offsets:
        return None
    return offsets.most_common(1)[0][0]


def detect_offset_for_path(path: str) -> int | None:
    """Open a PDF and detect its front-matter offset.

    Args:
        path: Path to the PDF.

    Returns:
        The offset, or ``None`` if it cannot be determined.
    """
    try:
        with fitz.open(path) as document:
            return detect_offset(document)
    except Exception as exc:  # noqa: BLE001 - PyMuPDF raises bare Exception subclasses
        log.warning("could not detect page-label offset", extra={"path": path, "error": str(exc)})
        return None
