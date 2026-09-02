"""Stage 2 data model and metadata resolution.

build-plan.md section 4, deliverable at the end of section 4.5.

:class:`Page` is the ``pages.jsonl`` schema, and it is shared by both content
streams: PDF pages from :mod:`seeley_rag.parse.pdf`, and synthetic single-page
records for the diagnostic articles from :mod:`seeley_rag.parse.html`. One schema
for both is what lets Stage 3 chunk them with the same code path while keeping
``content_stream`` available for the retrieval boost.

Named ``base.py`` to mirror :mod:`seeley_rag.acquire.base`, which plays the same
role for Stage 1.

Metadata resolution lives here too. Folder and category names give
``product_family`` and ``doc_type`` at near-perfect accuracy for zero LLM cost
(build-plan section 3.4), and getting it right is what prevents a TQ fault code
being answered from a Climate Wizard manual -- the failure that permanently
destroys installer trust.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Literal

from pydantic import BaseModel, ConfigDict, Field

from seeley_rag.exceptions import ParseError
from seeley_rag.logging_conf import get_logger
from seeley_rag.settings import get_models_lexicon

log = get_logger(__name__)

#: Parsing tier. Mirrors :data:`seeley_rag.parse.triage.Tier`.
Tier = Literal["plain_text", "diagram_heavy", "scanned"]

#: Which ingestion path produced a page.
ContentStream = Literal["pdf", "diagnostic_article"]

#: Where a page's printed label came from, most to least trustworthy.
#:
#: ``embedded`` -- the PDF's own page-label tree.
#: ``text``     -- a footer or header regex over the page text.
#: ``index``    -- fell back to ``page_index + 1``; a guess, not a label.
#: ``none``     -- not applicable (diagnostic articles have no printed page).
#:
#: Recorded rather than inferred. A label that merely *equals* ``index + 1`` is
#: not the same as a fallback, and conflating them hides whether the corpus can
#: support the page-accuracy gate at all.
LabelSource = Literal["embedded", "text", "index", "none"]

#: Product family when the lexicon cannot place a document. Deliberately not
#: guessed: a wrong family is worse than an absent one, because retrieval
#: soft-boosts on it.
UNKNOWN_FAMILY = "UNKNOWN"

#: Document type when no folder pattern matches.
UNKNOWN_DOC_TYPE = "unknown"


class Table(BaseModel):
    """One detected table, kept whole.

    A fault-code table is the most valuable content in the corpus and the
    easiest to destroy by chunking, so it travels as a unit with its own
    provenance until Stage 3 decides how to split it.

    Attributes:
        rows: Cell text, row-major. The first row is the header when
            ``has_header`` is set.
        has_header: Whether ``rows[0]`` looks like a header. Used by the
            multi-page merge: a continuation page has no header row.
        bbox: ``(x0, y0, x1, y1)`` on the page, for the column-geometry
            comparison that drives multi-page merging.
        n_columns: Column count, cached for that same comparison.
    """

    model_config = ConfigDict(extra="forbid")

    rows: list[list[str]] = Field(default_factory=list)
    has_header: bool = False
    bbox: tuple[float, float, float, float] | None = None
    n_columns: int = 0

    def to_markdown(self) -> str:
        """Render the table as markdown.

        Returns:
            A markdown table, header row included when present.
        """
        if not self.rows:
            return ""
        width = max(len(row) for row in self.rows)
        padded = [list(row) + [""] * (width - len(row)) for row in self.rows]
        lines = ["| " + " | ".join(c.replace("\n", " ").strip() for c in padded[0]) + " |"]
        lines.append("|" + "|".join(["---"] * width) + "|")
        for row in padded[1:]:
            lines.append("| " + " | ".join(c.replace("\n", " ").strip() for c in row) + " |")
        return "\n".join(lines)


class Page(BaseModel):
    """One page of parsed content. This is the ``pages.jsonl`` row schema.

    Attributes:
        doc_id: The document's SHA-256 for PDF pages, or ``article:{id}`` for a
            diagnostic article.
        page_index: 0-based index within the document. ``None`` for articles.
            Internal only -- never shown to a user.
        page_label: The **printed** page number. This is what a citation
            displays, and what an SME writes in ``expected_page``. Service
            manuals have front matter, so it routinely differs from
            ``page_index + 1`` (build-plan section 4.5).
        label_source: How ``page_label`` was obtained. An ``index`` source means
            the number is a guess and should not be trusted in an eval.
        text: Extracted text, whitespace normalised per line.
        tables: Tables detected on the page.
        tier: Parsing tier, from the same thresholds triage uses.
        needs_vision: Whether this page still requires a vision call. True for
            the scanned and diagram-heavy tiers until ``vision.py`` is
            implemented, so the work is recorded rather than silently skipped.
        image_path: Repo-relative path to the rendered page PNG, if rendered.
        source_article_ids: Every article that links this document. A chunk from
            a shared manual must be able to cite whichever article the installer
            arrived from.
        product_family: Resolved from category and folder via
            ``config/models.yaml``.
        doc_type: Resolved from the folder name.
        model_series: Model codes found in the document or article title.
        title: The linking article's title.
        source_url: The attachment URL for PDFs, the article URL for articles.
        article_url: The article to open for verification.
        category: Solution category, carried through for breadcrumbs.
        folder: Folder name, carried through for breadcrumbs.
        content_stream: Which ingestion path produced this page.
    """

    # protected_namespaces is cleared because `model_series` is a real domain
    # field -- Seeley model codes -- not a pydantic accessor.
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    doc_id: str
    page_index: int | None = None
    page_label: str | None = None
    label_source: LabelSource = "none"
    text: str = ""
    tables: list[Table] = Field(default_factory=list)
    tier: Tier = "plain_text"
    needs_vision: bool = False
    image_path: str | None = None
    source_article_ids: list[str] = Field(default_factory=list)
    product_family: str = UNKNOWN_FAMILY
    doc_type: str = UNKNOWN_DOC_TYPE
    model_series: list[str] = Field(default_factory=list)
    title: str = ""
    source_url: str = ""
    article_url: str | None = None
    category: str = ""
    folder: str = ""
    content_stream: ContentStream = "pdf"

    @property
    def char_count(self) -> int:
        """Length of the extracted text."""
        return len(self.text)

    @property
    def has_content(self) -> bool:
        """Whether this page carries anything worth indexing.

        A scanned page awaiting transcription has no usable text, and indexing
        its empty body would put a citable-but-contentless chunk in the store.
        """
        return bool(self.text.strip()) or bool(self.tables)

    def breadcrumb(self) -> str:
        """Build the breadcrumb prefix Stage 3 prepends to every chunk.

        Putting the product name into the embedded text of every chunk lifts
        retrieval measurably, including for chunks whose body never names the
        product (build-plan section 5.1, rule 5).

        Returns:
            e.g. ``Ducted Gas Heating > Service Guides > TQ Service Guide > p.42``.
        """
        parts = [p for p in (self.category, self.folder, self.title) if p]
        if self.page_label:
            parts.append(f"p.{self.page_label}")
        return " > ".join(parts)


# ---------------------------------------------------------------------------
# Metadata resolution from config/models.yaml
# ---------------------------------------------------------------------------


def resolve_product_family(category: str, folder: str = "", title: str = "") -> str:
    """Resolve a document's product family from its metadata.

    Checks, in order of decreasing reliability: category patterns, aliases in
    the category or folder, then model codes in the title.

    Within each tier the **longest matching pattern wins**, not the first one
    found. Family names in this corpus nest: "VRF REVERSE CYCLE SERVICE AND
    INSTALLATION" contains RC's "Reverse Cycle", so first-match ordering
    labelled all 111 VRF documents as RC. Longest-match resolves it, and it does
    not depend on dictionary ordering in the YAML -- which is exactly the kind of
    accidental dependency that produces a wrong answer nobody can explain.

    Returns :data:`UNKNOWN_FAMILY` rather than guessing. Retrieval soft-boosts
    on this field, so a confident wrong answer routes a TQ question into a
    Climate Wizard manual -- build-plan section 13, risk 3.

    Args:
        category: Solution category name.
        folder: Folder name.
        title: Article or document title.

    Returns:
        A family key from ``config/models.yaml``, or :data:`UNKNOWN_FAMILY`.
    """
    families: dict[str, Any] = get_models_lexicon().get("families", {})
    haystack = f"{category} {folder}".lower()

    for field in ("category_patterns", "aliases"):
        best_key = UNKNOWN_FAMILY
        best_len = 0
        for key, spec in families.items():
            for pattern in spec.get(field, []):
                lowered = pattern.lower()
                if lowered in haystack and len(lowered) > best_len:
                    best_key, best_len = key, len(lowered)
        if best_key != UNKNOWN_FAMILY:
            return best_key

    # Title model codes are the weakest signal -- "TE" appears inside ordinary
    # words -- so they are matched as whole tokens only, and last. Longest code
    # wins here too, so "MCMX" beats a bare "CW" appearing in the same title.
    tokens = {t.strip(".,()/-").upper() for t in title.replace("/", " ").split()}
    best_key = UNKNOWN_FAMILY
    best_len = 0
    for key, spec in families.items():
        for code in spec.get("model_codes", []):
            upper = code.upper()
            if upper in tokens and len(upper) > best_len:
                best_key, best_len = key, len(upper)
    return best_key


def resolve_doc_type(folder: str, title: str = "") -> str:
    """Resolve a document type from the folder name.

    Args:
        folder: Folder name, the primary signal.
        title: Title, used as a fallback.

    Returns:
        A doc-type key from ``config/models.yaml``, or
        :data:`UNKNOWN_DOC_TYPE`.
    """
    doc_types: dict[str, list[str]] = get_models_lexicon().get("doc_types", {})
    for haystack in (folder.lower(), title.lower()):
        if not haystack:
            continue
        for key, patterns in doc_types.items():
            for pattern in patterns:
                if pattern.lower() in haystack:
                    return key
    return UNKNOWN_DOC_TYPE


def resolve_model_series(*texts: str) -> list[str]:
    """Extract known model codes from titles.

    Matched as whole tokens: substring matching turns every "TE" in ordinary
    prose into a model hit, and a wrong model code is a wrong answer.

    Args:
        *texts: Titles or filenames to scan.

    Returns:
        Distinct model codes, in lexicon order.
    """
    families: dict[str, Any] = get_models_lexicon().get("families", {})
    tokens: set[str] = set()
    for text in texts:
        for raw in text.replace("/", " ").replace("-", " ").split():
            tokens.add(raw.strip(".,()").upper())

    found: list[str] = []
    for spec in families.values():
        for code in spec.get("model_codes", []):
            if code.upper() in tokens and code not in found:
                found.append(code)
    return found


# ---------------------------------------------------------------------------
# pages.jsonl I/O
# ---------------------------------------------------------------------------


class PagesWriter:
    """Append-safe JSONL writer for parsed pages.

    Flushed per row, like the manifest writer: a full parse is a long unattended
    run, and buffered rows lost to a kill would have to be re-parsed.

    Args:
        path: Destination. Defaults to ``data/01_interim/pages.jsonl``.
        overwrite: Truncate instead of appending.

    Attributes:
        count: Rows written.
    """

    def __init__(self, path: Path | None = None, overwrite: bool = False) -> None:
        from seeley_rag.paths import PAGES_PATH

        self.path = path or PAGES_PATH
        self.overwrite = overwrite
        self.count = 0
        self._handle: Any = None

    def __enter__(self) -> PagesWriter:
        """Open the file for writing."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open(
            "w" if self.overwrite else "a", encoding="utf-8", newline="\n"
        )
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the file."""
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def write(self, page: Page) -> None:
        """Append one page.

        Args:
            page: The page to record.

        Raises:
            ParseError: If the writer is not open.
        """
        if self._handle is None:
            raise ParseError("PagesWriter must be used as a context manager.")
        self._handle.write(json.dumps(page.model_dump(mode="json"), ensure_ascii=False) + "\n")
        self._handle.flush()
        self.count += 1

    def write_all(self, pages: list[Page]) -> int:
        """Append several pages.

        Args:
            pages: Pages to record.

        Returns:
            How many were written.
        """
        for page in pages:
            self.write(page)
        return len(pages)


def read_pages(path: Path | None = None) -> Iterator[Page]:
    """Stream ``pages.jsonl``.

    Args:
        path: Source. Defaults to ``data/01_interim/pages.jsonl``.

    Yields:
        Each page in file order.

    Raises:
        ParseError: If the file is missing or a row is invalid.
    """
    from seeley_rag.paths import PAGES_PATH

    resolved = path or PAGES_PATH
    if not resolved.exists():
        raise ParseError(f"No parsed pages at {resolved}. Run `python scripts/03_parse.py` first.")
    with resolved.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield Page.model_validate_json(stripped)
            except Exception as exc:  # noqa: BLE001 - pydantic and json both raise
                raise ParseError(f"{resolved}:{number} is not a valid page row: {exc}") from exc


def parsed_doc_ids(path: Path | None = None) -> set[str]:
    """Return the document IDs already present in ``pages.jsonl``.

    Lets a re-run skip documents it has already parsed, the same way the crawl
    resumes from the manifest.

    Args:
        path: Source. Defaults to ``data/01_interim/pages.jsonl``.

    Returns:
        Document IDs already parsed. Empty when the file does not exist.
    """
    from seeley_rag.paths import PAGES_PATH

    resolved = path or PAGES_PATH
    if not resolved.exists():
        return set()
    seen: set[str] = set()
    with resolved.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                seen.add(json.loads(stripped)["doc_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return seen
