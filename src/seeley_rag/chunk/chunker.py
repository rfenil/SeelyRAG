"""Stage 3 -- page-anchored chunking.

build-plan.md section 5.1.

Rules the implementation holds to:

1. **Never cross a page boundary** (except the multi-page table merge). Page
   provenance is a requirement, so violating it is made structurally
   impossible: :func:`chunk_page` only ever sees one page.
2. Target ~800 tokens, hard max ~1,200, ~120-token overlap within a page.
3. A detected table is one atomic chunk, capped at ~6,000 tokens -- see
   :mod:`seeley_rag.chunk.tables`.
4. Merge multi-page tables before chunking.
5. Breadcrumb prefix on every chunk:
   ``Ducted Gas Heating > Service Guides > TQ Service Guide 644066-M > p.42``

Rule 5 alone measurably lifts retrieval: it puts the product name into the
embedded text of every chunk, including chunks whose body never names it.

Output: ``data/02_processed/chunks.jsonl``.

Splitting is hierarchical -- paragraphs, then lines, then sentences, then a hard
token cut. Each level is tried only when the level above leaves a piece too big,
so ordinary prose splits on blank lines and only pathological input reaches the
blunt instrument. Manual text is full of numbered procedure steps and bare
newlines, which is why lines sit above sentences: splitting "1. Isolate power"
from "2. Remove the panel" at a line break preserves the step; splitting on
full stops does not.
"""

from __future__ import annotations

import re
from typing import Iterable, Iterator

from seeley_rag.chunk.base import Chunk, make_chunk_id
from seeley_rag.chunk.tables import (
    MergedTable,
    merge_multipage_tables,
    render_table,
    split_oversized_table,
    table_caption,
)
from seeley_rag.chunk.tokens import count_tokens, truncate_to_tokens
from seeley_rag.logging_conf import get_logger
from seeley_rag.parse.base import Page
from seeley_rag.settings import get_settings

log = get_logger(__name__)

#: Paragraph break: a blank line, however much trailing whitespace it carries.
_PARAGRAPH_RE = re.compile(r"\n\s*\n")

#: Sentence end followed by whitespace. Kept deliberately simple -- it splits
#: "5.5 kPa. Check" correctly enough, and a mis-split inside a chunk costs far
#: less than the regex complexity needed to be exhaustive about abbreviations.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def build_breadcrumb(category: str, folder: str, title: str, page_label: str | None) -> str:
    """Build the breadcrumb prefix prepended to every chunk.

    Args:
        category: Solution category.
        folder: Folder name.
        title: Document or article title.
        page_label: Printed page label, if any.

    Returns:
        e.g. ``Ducted Gas Heating > Service Guides > TQ Service Guide > p.42``.
    """
    parts = [part for part in (category, folder, title) if part]
    if page_label:
        parts.append(f"p.{page_label}")
    return " > ".join(parts)


def _chunk_from_page(
    page: Page,
    ordinal: int,
    body: str,
    kind: str,
    page_range: str | None = None,
    page_span: int = 1,
) -> Chunk:
    """Build one chunk record carrying a page's full provenance.

    Args:
        page: Source page.
        ordinal: Position within the page.
        body: Chunk body, breadcrumb not yet attached.
        kind: ``"prose"`` or ``"table"``.
        page_range: Printed page range for a merged table, else ``None``.
        page_span: Pages the content covers; 1 for everything but a merge.

    Returns:
        A finalised chunk.
    """
    breadcrumb = build_breadcrumb(page.category, page.folder, page.title, page.page_label)
    text = f"{breadcrumb}\n\n{body}" if breadcrumb else body
    chunk = Chunk(
        chunk_id=make_chunk_id(page.doc_id, page.page_index, ordinal),
        doc_id=page.doc_id,
        text=text,
        kind=kind,  # type: ignore[arg-type]
        page_index=page.page_index,
        page_label=page.page_label,
        label_source=page.label_source,
        page_range=page_range,
        page_span=page_span,
        page_image=page.image_path,
        source_article_ids=list(page.source_article_ids),
        product_family=page.product_family,
        model_series=list(page.model_series),
        doc_type=page.doc_type,
        title=page.title,
        source_url=page.source_url,
        article_url=page.article_url,
        category=page.category,
        folder=page.folder,
        tier=page.tier,
        content_stream=page.content_stream,
        needs_vision=page.needs_vision,
    )
    return chunk.finalise()


def split_text(
    text: str,
    target_tokens: int = 800,
    max_tokens: int = 1200,
    overlap_tokens: int = 120,
) -> list[str]:
    """Split prose into overlapping pieces without crossing a page boundary.

    Pieces accumulate to ``target_tokens`` and are flushed; ``max_tokens`` is
    the ceiling a single piece may never exceed. The overlap is taken from the
    tail of the previous piece so a procedure split across two chunks still
    reads from either one.

    Args:
        text: Page text.
        target_tokens: Size to aim for.
        max_tokens: Hard ceiling per piece.
        overlap_tokens: Tokens of the previous piece to repeat.

    Returns:
        The pieces, in order. Empty when ``text`` has no content.
    """
    stripped = text.strip()
    if not stripped:
        return []
    if count_tokens(stripped) <= max_tokens:
        return [stripped]

    units = _split_to_units(stripped, max_tokens)
    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for unit in units:
        unit_tokens = count_tokens(unit)
        if current and current_tokens + unit_tokens > target_tokens:
            pieces.append("\n\n".join(current))
            carry = _tail_overlap(pieces[-1], overlap_tokens)
            # The overlap yields to the ceiling, never the other way round. A
            # unit may itself be as large as max_tokens, so prepending a carry
            # to it can breach the hard cap -- and that cap is an API limit
            # while the overlap is only a convenience. Drop the carry rather
            # than the content.
            if carry and count_tokens(carry) + unit_tokens <= max_tokens:
                current, current_tokens = [carry], count_tokens(carry)
            else:
                current, current_tokens = [], 0
        current.append(unit)
        current_tokens += unit_tokens

    if current:
        pieces.append("\n\n".join(current))

    # Last line of defence. _split_to_units guarantees every unit fits and the
    # carry rule above keeps the accumulator inside the cap, but a truncation
    # here is far cheaper than a 400 from the embedding API.
    return [
        piece if count_tokens(piece) <= max_tokens else truncate_to_tokens(piece, max_tokens)
        for piece in pieces
        if piece.strip()
    ]


def _split_to_units(text: str, max_tokens: int) -> list[str]:
    """Break text into units no larger than ``max_tokens``.

    Descends paragraphs -> lines -> sentences -> hard token cut, moving to the
    next level only for the pieces the previous level left too large.

    Args:
        text: Text to break up.
        max_tokens: Ceiling for a single unit.

    Returns:
        Units, in document order.
    """
    units: list[str] = []
    for paragraph in _PARAGRAPH_RE.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if count_tokens(paragraph) <= max_tokens:
            units.append(paragraph)
            continue
        units.extend(_split_stubborn(paragraph, max_tokens))
    return units


def _split_stubborn(text: str, max_tokens: int) -> list[str]:
    """Split a paragraph that is itself too large.

    Args:
        text: The oversized paragraph.
        max_tokens: Ceiling for a single unit.

    Returns:
        Units within the ceiling.
    """
    out: list[str] = []
    for line in _regroup(text.split("\n"), max_tokens):
        if count_tokens(line) <= max_tokens:
            out.append(line)
            continue
        for sentence in _regroup(_SENTENCE_RE.split(line), max_tokens):
            if count_tokens(sentence) <= max_tokens:
                out.append(sentence)
            else:
                out.extend(_hard_split(sentence, max_tokens))
    return [u for u in out if u.strip()]


def _regroup(fragments: Iterable[str], max_tokens: int) -> list[str]:
    """Recombine small fragments up to ``max_tokens``.

    Splitting a paragraph into single lines and embedding each one destroys the
    context that made the paragraph meaningful, so fragments are packed back
    together as far as the ceiling allows.

    Args:
        fragments: Fragments in order.
        max_tokens: Ceiling per group.

    Returns:
        Regrouped text blocks.
    """
    groups: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for fragment in fragments:
        fragment = fragment.strip()
        if not fragment:
            continue
        tokens = count_tokens(fragment)
        if current and current_tokens + tokens > max_tokens:
            groups.append("\n".join(current))
            current, current_tokens = [], 0
        current.append(fragment)
        current_tokens += tokens
    if current:
        groups.append("\n".join(current))
    return groups


def _hard_split(text: str, max_tokens: int) -> list[str]:
    """Cut text at token boundaries, ignoring meaning.

    The last resort, for input with no usable break at all -- a run-on table
    dumped as prose, or an unbroken string of part numbers.

    Args:
        text: Text to cut.
        max_tokens: Ceiling per piece.

    Returns:
        Pieces within the ceiling.
    """
    pieces: list[str] = []
    remaining = text
    while remaining:
        head = truncate_to_tokens(remaining, max_tokens)
        if not head:
            break
        pieces.append(head)
        remaining = remaining[len(head) :].strip()
    return pieces


def _tail_overlap(text: str, overlap_tokens: int) -> str:
    """Return the last ``overlap_tokens`` of ``text``, on a word boundary.

    Args:
        text: The piece being closed.
        overlap_tokens: How much to carry forward.

    Returns:
        The overlapping tail, or ``""`` when overlap is disabled.
    """
    if overlap_tokens <= 0 or not text:
        return ""
    words = text.split()
    if not words:
        return ""
    # Walk back from the end until the tail reaches the overlap budget. Cheaper
    # than tokenising the whole piece and slicing, and the result lands on a
    # word boundary rather than mid-token.
    tail: list[str] = []
    for word in reversed(words):
        tail.insert(0, word)
        if count_tokens(" ".join(tail)) >= overlap_tokens:
            break
    return " ".join(tail)


def chunk_page(page: Page, merged_tables: list[MergedTable] | None = None) -> list[Chunk]:
    """Chunk a single page, never crossing its boundary.

    Tables are emitted first and atomically; the page's prose follows. Table
    chunks lead because a page whose value is its fault-code table should have
    that table as its lowest-numbered, most stable chunk id.

    Args:
        page: One parsed page.
        merged_tables: Tables already merged across pages and anchored to this
            one. When ``None``, the page's own tables are used unmerged -- the
            single-page path, used by tests and by callers with one page in hand.

    Returns:
        Chunk records for that page, in order.
    """
    settings = get_settings().chunk
    chunks: list[Chunk] = []
    ordinal = 0

    tables = (
        merged_tables
        if merged_tables is not None
        else [
            MergedTable(table=t, page=page, page_range=None, ordinal=i)
            for i, t in enumerate(page.tables)
        ]
    )

    for entry in tables:
        caption = table_caption(entry.page)
        rendered = render_table(entry.table, caption)
        parts = (
            [rendered]
            if count_tokens(rendered) <= settings.table_max_tokens
            else split_oversized_table(entry.table, settings.table_max_tokens, caption)
        )
        for part in parts:
            if len(part.strip()) < settings.min_chunk_chars:
                continue
            chunks.append(
                _chunk_from_page(page, ordinal, part, "table", entry.page_range, entry.page_span)
            )
            ordinal += 1

    for piece in split_text(
        page.text,
        target_tokens=settings.target_tokens,
        max_tokens=settings.max_tokens,
        overlap_tokens=settings.overlap_tokens,
    ):
        if len(piece.strip()) < settings.min_chunk_chars:
            continue
        chunks.append(_chunk_from_page(page, ordinal, piece, "prose", None))
        ordinal += 1

    return chunks


def chunk_document(pages: Iterable[Page]) -> list[Chunk]:
    """Chunk every page of one document, merging tables across pages first.

    Args:
        pages: Every parsed page of a single document.

    Returns:
        Chunks for the whole document, in page order.
    """
    ordered = sorted(pages, key=lambda p: (p.page_index if p.page_index is not None else -1))
    merged = merge_multipage_tables(ordered)

    by_page: dict[int | None, list[MergedTable]] = {}
    for entry in merged:
        by_page.setdefault(entry.page.page_index, []).append(entry)

    chunks: list[Chunk] = []
    for page in ordered:
        chunks.extend(chunk_page(page, by_page.get(page.page_index, [])))
    return chunks


def chunk_corpus(pages: Iterable[Page]) -> Iterator[Chunk]:
    """Chunk a whole corpus, grouping pages into documents as they stream.

    ``pages.jsonl`` is written document by document, so pages of one document
    arrive contiguously and a group can be closed as soon as the ``doc_id``
    changes. That keeps memory to one document rather than the whole 13,156-page
    corpus.

    Args:
        pages: Pages in file order.

    Yields:
        Chunks, document by document.
    """
    current_doc: str | None = None
    buffer: list[Page] = []

    for page in pages:
        if current_doc is not None and page.doc_id != current_doc:
            yield from chunk_document(buffer)
            buffer = []
        current_doc = page.doc_id
        buffer.append(page)

    if buffer:
        yield from chunk_document(buffer)
