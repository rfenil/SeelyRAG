"""Stage 0 -- PDF corpus triage.

build-plan.md section 4.1.

Twenty minutes of work that sets the cost and time budget for the next two days.
Every page is classified into one of three tiers:

* **plain_text**  -- a usable text layer, few images. PyMuPDF extraction is
  enough. Cheap.
* **diagram_heavy** -- has text but is picture-dominated. Needs a vision call
  for a caption, or "TQ wiring diagram" will never retrieve it.
* **scanned** -- no usable text layer. Needs a full vision transcription. The
  most expensive tier.

**Report all three fractions, not just the scanned one.** The build plan's v1
modelled only the scanned fraction and under-budgeted vision by 2-3x, because in
illustrated service manuals the diagram-heavy tier is often the larger of the
two. The vision line is the only estimate in the project that can move by 3x,
and it moves as a function of these numbers.

Each page also records ``page.get_label()`` -- the *printed* page number. Service
manuals have front matter, so the printed number routinely differs from the
0-based index by 4-10. Citations must use the label; see build-plan section 4.5,
which is also why the SME question template states which one it is asking for.
"""

from __future__ import annotations

import datetime as dt
from collections import Counter
from pathlib import Path
from typing import Literal

import fitz
from pydantic import BaseModel, Field

from seeley_rag.exceptions import ParseError
from seeley_rag.logging_conf import get_logger
from seeley_rag.paths import REPORTS_DIR, triage_report_path
from seeley_rag.settings import Settings, get_settings

log = get_logger(__name__)

#: Parsing tier a page falls into. Drives the vision budget.
Tier = Literal["plain_text", "diagram_heavy", "scanned"]


class PageTriage(BaseModel):
    """Triage result for a single PDF page.

    Attributes:
        page_index: 0-based index, as ``enumerate(doc)`` gives it. Internal.
        page_label: The printed page number from ``page.get_label()``. This is
            what a citation shows and what an SME writes in ``expected_page``.
        chars: Characters of extractable text, whitespace-stripped.
        n_images: Images embedded on the page.
        has_text_layer: Whether there is enough text to work with.
        diagram_heavy: Has text, but is picture-dominated.
        tier: The resulting parsing tier.
    """

    page_index: int
    page_label: str | None
    chars: int
    n_images: int
    has_text_layer: bool
    diagram_heavy: bool
    tier: Tier


class DocumentTriage(BaseModel):
    """Triage result for one PDF.

    Attributes:
        path: File that was triaged.
        page_count: Pages in the document.
        pages: Per-page results.
        label_offset: ``page_label - (page_index + 1)`` where the label is
            numeric and consistent, else ``None``. A non-zero value is the
            front-matter offset that makes page accuracy fail corpus-wide while
            looking like a retrieval bug.
        error: Why the document could not be read, if it could not.
    """

    path: str
    page_count: int = 0
    pages: list[PageTriage] = Field(default_factory=list)
    label_offset: int | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        """Whether the document was read successfully."""
        return self.error is None


class TriageSummary(BaseModel):
    """Aggregate triage across a set of PDFs.

    Attributes:
        documents: Documents triaged successfully.
        failed_documents: Documents that could not be read.
        total_pages: Pages across all readable documents.
        scanned_pages: Pages with no usable text layer.
        diagram_heavy_pages: Pages with text but picture-dominated.
        plain_text_pages: Pages needing no vision call.
        pages_with_labels: Pages exposing a printed page label.
        label_offsets: Observed ``label - (index + 1)`` offsets, by frequency.
    """

    documents: int = 0
    failed_documents: int = 0
    total_pages: int = 0
    scanned_pages: int = 0
    diagram_heavy_pages: int = 0
    plain_text_pages: int = 0
    pages_with_labels: int = 0
    label_offsets: dict[str, int] = Field(default_factory=dict)

    def _fraction(self, count: int) -> float:
        """Return ``count / total_pages``, or 0.0 when there are no pages."""
        return (count / self.total_pages) if self.total_pages else 0.0

    @property
    def pct_scanned(self) -> float:
        """Fraction of pages needing full vision transcription (Tier B)."""
        return self._fraction(self.scanned_pages)

    @property
    def pct_diagram_heavy(self) -> float:
        """Fraction of pages needing a vision caption (Tier C)."""
        return self._fraction(self.diagram_heavy_pages)

    @property
    def pct_plain_text(self) -> float:
        """Fraction of pages needing no vision call at all (Tier A)."""
        return self._fraction(self.plain_text_pages)

    @property
    def pct_vision(self) -> float:
        """Fraction of pages needing a vision call of either kind.

        This is the number the budget actually hangs off. If it is above ~50%,
        the pilot should be restricted to newer manuals and the gap documented
        (build-plan section 13, risk 2).
        """
        return self.pct_scanned + self.pct_diagram_heavy


def is_pdf(path: Path) -> bool:
    """Whether a file really is a PDF, judged by its magic bytes.

    The extension is not evidence. Acquisition stores content-addressed files,
    and anything the portal served with a 200 lands there -- including, in at
    least one observed case, a login page.

    Args:
        path: File to check.

    Returns:
        True if the file starts with ``%PDF``.
    """
    try:
        with path.open("rb") as handle:
            return handle.read(5).startswith(b"%PDF")
    except OSError:
        return False


def read_page_label(page: fitz.Page) -> str | None:
    """Return a page's printed label, or ``None`` when it has none.

    ``page.get_label()`` raises a bare ``AssertionError`` from inside PyMuPDF
    when the document cannot be viewed as a PDF, and raises other internal
    errors on malformed page-label trees. Neither should end a triage run over
    hundreds of manuals, and neither is recoverable information -- absent a
    label, the caller falls back to ``page_index + 1``.

    Args:
        page: The page to read.

    Returns:
        The printed label, or ``None``.
    """
    try:
        return page.get_label() or None
    except (AssertionError, RuntimeError, ValueError):
        return None


def classify_page(
    chars: int, n_images: int, settings: Settings | None = None
) -> tuple[bool, bool, Tier]:
    """Classify one page into a parsing tier.

    Args:
        chars: Stripped character count of the page's text layer.
        n_images: Number of embedded images.
        settings: Settings override, for tests.

    Returns:
        ``(has_text_layer, diagram_heavy, tier)``.
    """
    config = (settings or get_settings()).triage
    has_text_layer = chars > config.text_layer_min_chars
    if not has_text_layer:
        return False, False, "scanned"
    diagram_heavy = (
        chars < config.diagram_heavy_max_chars and n_images >= config.diagram_heavy_min_images
    )
    return True, diagram_heavy, "diagram_heavy" if diagram_heavy else "plain_text"


def triage_pdf(path: Path, settings: Settings | None = None) -> DocumentTriage:
    """Triage every page of one PDF.

    Args:
        path: PDF to inspect.
        settings: Settings override, for tests.

    Returns:
        A :class:`DocumentTriage`. A document that cannot be opened comes back
        with ``error`` set rather than raising -- one corrupt manual must not
        stop the triage of the other five.
    """
    if not is_pdf(path):
        # PyMuPDF happily opens PNGs and JPEGs as documents, so a non-PDF that
        # slipped into the store would be triaged as a one-page scan rather than
        # reported. Worse, page.get_label() asserts on a non-PDF and takes the
        # whole run down. Check the magic bytes and say so plainly instead.
        log.error("not a pdf", extra={"path": str(path)})
        return DocumentTriage(path=str(path), error="not a PDF (wrong file type in the store)")

    try:
        document = fitz.open(path)
    except Exception as exc:  # noqa: BLE001 - PyMuPDF raises bare Exception subclasses
        log.error("could not open pdf", extra={"path": str(path), "error": str(exc)})
        return DocumentTriage(path=str(path), error=str(exc))

    pages: list[PageTriage] = []
    offsets: Counter[int] = Counter()
    try:
        for index, page in enumerate(document):
            text = page.get_text("text").strip()
            images = page.get_images()
            has_text_layer, diagram_heavy, tier = classify_page(len(text), len(images), settings)

            label = read_page_label(page)
            if label and label.isdigit():
                offsets[int(label) - (index + 1)] += 1

            pages.append(
                PageTriage(
                    page_index=index,
                    page_label=label,
                    chars=len(text),
                    n_images=len(images),
                    has_text_layer=has_text_layer,
                    diagram_heavy=diagram_heavy,
                    tier=tier,
                )
            )
        page_count = document.page_count
    finally:
        document.close()

    label_offset = offsets.most_common(1)[0][0] if offsets else None
    log.info(
        "triaged pdf",
        extra={"path": str(path), "pages": page_count, "label_offset": label_offset},
    )
    return DocumentTriage(
        path=str(path), page_count=page_count, pages=pages, label_offset=label_offset
    )


def triage_corpus(
    paths: list[Path], settings: Settings | None = None
) -> tuple[list[DocumentTriage], TriageSummary]:
    """Triage a set of PDFs and aggregate the three fractions.

    Args:
        paths: PDFs to inspect.
        settings: Settings override, for tests.

    Returns:
        The per-document results and the aggregate summary.

    Raises:
        ParseError: If ``paths`` is empty.
    """
    if not paths:
        raise ParseError(
            "No PDFs to triage. Hand-download six representative manuals -- span "
            "the date range, e.g. a 2023, a 2015 and a 2005 guide -- or point this "
            "at data/00_raw/pdf/ after a crawl."
        )

    documents = [triage_pdf(path, settings) for path in paths]
    summary = TriageSummary()
    offsets: Counter[int] = Counter()

    for document in documents:
        if not document.ok:
            summary.failed_documents += 1
            continue
        summary.documents += 1
        if document.label_offset is not None:
            offsets[document.label_offset] += 1
        for page in document.pages:
            summary.total_pages += 1
            if page.page_label:
                summary.pages_with_labels += 1
            if page.tier == "scanned":
                summary.scanned_pages += 1
            elif page.tier == "diagram_heavy":
                summary.diagram_heavy_pages += 1
            else:
                summary.plain_text_pages += 1

    summary.label_offsets = {str(k): v for k, v in sorted(offsets.items())}
    return documents, summary


def render_report(documents: list[DocumentTriage], summary: TriageSummary, timestamp: str) -> str:
    """Render the triage results as a markdown report.

    Args:
        documents: Per-document results.
        summary: Aggregate summary.
        timestamp: Compact UTC stamp for the heading.

    Returns:
        The report as markdown.
    """
    lines: list[str] = [
        "# PDF triage report",
        "",
        f"Generated: {timestamp} (UTC)  ",
        f"Documents: {summary.documents} readable, {summary.failed_documents} failed  ",
        f"Pages: {summary.total_pages}",
        "",
        "## The three fractions",
        "",
        "These set the vision budget. All three matter -- modelling only the",
        "scanned fraction under-budgets vision by 2-3x, because diagram-heavy",
        "pages need a vision call too.",
        "",
        "| Tier | Pages | Share | Needs vision |",
        "|---|---:|---:|---|",
        f"| A · plain text | {summary.plain_text_pages} | {summary.pct_plain_text:.1%} | no |",
        f"| C · diagram-heavy | {summary.diagram_heavy_pages} | {summary.pct_diagram_heavy:.1%} "
        "| yes, caption |",
        f"| B · scanned | {summary.scanned_pages} | {summary.pct_scanned:.1%} "
        "| yes, full transcription |",
        "",
        f"**Pages needing a vision call: {summary.pct_vision:.1%}**",
        "",
    ]

    if summary.pct_vision > 0.5:
        lines += [
            "> **Budget warning.** More than half of all pages need a vision call.",
            "> Build-plan section 13 risk 2 applies: restrict the pilot to post-2013",
            "> manuals and document the coverage gap explicitly, or expect both cost",
            "> and Day 1 machine time to overrun by around 3x.",
            "",
        ]

    lines += [
        "## Page labels",
        "",
        f"Pages exposing a printed label: {summary.pages_with_labels} / {summary.total_pages}",
        "",
    ]
    if summary.label_offsets:
        lines += ["| Offset (label - index-1) | Documents |", "|---:|---:|"]
        lines += [f"| {k} | {v} |" for k, v in summary.label_offsets.items()]
        lines += [
            "",
            "A non-zero offset is front matter. Cite the **label**, not the index.",
            "An offset left unreconciled makes page accuracy fail corpus-wide while",
            "looking exactly like a retrieval bug (build-plan section 4.5).",
            "",
        ]
    else:
        lines += [
            "No numeric page labels found. Citations will fall back to",
            "`page_index + 1`, so confirm against a printed page before trusting",
            "any `expected_page` in the eval set.",
            "",
        ]

    lines += [
        "## Per document",
        "",
        "| Document | Pages | Plain | Diagram | Scanned | Label offset |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for document in documents:
        if not document.ok:
            lines.append(
                f"| {Path(document.path).name} | — | — | — | — | FAILED: {document.error} |"
            )
            continue
        tiers = Counter(page.tier for page in document.pages)
        lines.append(
            f"| {Path(document.path).name} | {document.page_count} "
            f"| {tiers['plain_text']} | {tiers['diagram_heavy']} | {tiers['scanned']} "
            f"| {document.label_offset if document.label_offset is not None else '—'} |"
        )

    lines += [
        "",
        "---",
        "",
        "Generated by `scripts/01_triage.py`. See build-plan.md section 4.1.",
        "",
    ]
    return "\n".join(lines)


def write_report(documents: list[DocumentTriage], summary: TriageSummary) -> Path:
    """Write the triage report to ``data/reports/``.

    Args:
        documents: Per-document results.
        summary: Aggregate summary.

    Returns:
        The path written.
    """
    timestamp = dt.datetime.now(tz=dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = triage_report_path(timestamp)
    path.write_text(render_report(documents, summary, timestamp), encoding="utf-8")
    log.info("wrote triage report", extra={"path": str(path)})
    return path
