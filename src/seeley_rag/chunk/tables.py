"""Stage 3 -- table chunking and multi-page merge.

build-plan.md section 5.1, rules 3 and 4.

Two failure modes this module exists to prevent, both of which destroy exactly
the most valuable content in the corpus:

* **Uncapped table chunks.** "One chunk whatever its size" 400s against
  ``text-embedding-3-large`` (hard cap 8,191 tokens) and truncates in Cohere
  rerank (near 4k). Cap at ~6,000 tokens, split on row boundaries, and repeat
  the header in each part.
* **Shredded multi-page tables.** Fault-code tables run 2-4 pages with the
  header only on the first, so per-page detection splits the code away from its
  meaning and neither half retrieves. Where consecutive pages have matching
  column geometry and the later has no header row, merge into one chunk anchored
  to the first page, with a ``page_range`` field.

The merge is deliberately conservative. A wrongly merged table welds two
unrelated fault tables together and cites the wrong page for half its rows,
which is worse than the shredding it was meant to fix -- so every one of the
:func:`is_continuation` conditions must hold, not most of them.
"""

from __future__ import annotations

from typing import Any, Iterable, NamedTuple

from seeley_rag.chunk.tokens import count_tokens, truncate_to_tokens
from seeley_rag.parse.base import Page, Table

#: Maximum horizontal drift, in points, between two tables still considered the
#: same column geometry.
#:
#: Measured over the corpus rather than guessed. Of the 36 adjacent page pairs
#: whose column counts match, the edge deltas fall into two clearly separated
#: groups: 24 pairs at 0-11pt (detector jitter on what is visibly one table),
#: two at 12.2-12.3pt (a numbered procedure table continuing across a page),
#: then nothing until 70.9pt and above -- which inspection confirms are
#: genuinely different tables with different column layouts. 24pt sits in the
#: empty gap, so it captures the real continuations without reaching the first
#: false one.
BBOX_TOLERANCE_PT = 24.0


class MergedTable(NamedTuple):
    """A table after the multi-page merge.

    Attributes:
        table: The table itself, rows concatenated across pages.
        page: The page the table **starts** on. Citations anchor here.
        page_range: ``"42-44"`` when the table spans pages and the labels
            support saying so, else ``None``. Kept separate from ``page_label``
            so a citation can show the span while still linking to a single
            page image.
        ordinal: Position of this table among its start page's tables.
        page_span: How many pages the table covers. Always accurate, even when
            ``page_range`` is ``None`` because the labels were guesses -- so a
            merge stays countable, and the eval can tell that a chunk spanning
            three pages cannot be scored against a single expected page.
    """

    table: Table
    page: Page
    page_range: str | None
    ordinal: int
    page_span: int = 1


def _columns_match(left: Table, right: Table) -> bool:
    """Whether two tables have the same column count and horizontal extent.

    Args:
        left: The earlier table.
        right: The candidate continuation.

    Returns:
        True when both the column count and the left/right edges agree within
        :data:`BBOX_TOLERANCE_PT`.
    """
    if left.n_columns != right.n_columns or left.n_columns == 0:
        return False
    if left.bbox is None or right.bbox is None:
        # Without geometry the column count alone is too weak a signal to weld
        # two tables together, so refuse rather than guess.
        return False
    return (
        abs(left.bbox[0] - right.bbox[0]) <= BBOX_TOLERANCE_PT
        and abs(left.bbox[2] - right.bbox[2]) <= BBOX_TOLERANCE_PT
    )


def is_continuation(previous: Table, candidate: Table) -> bool:
    """Whether ``candidate`` continues ``previous`` from the preceding page.

    Requires all of: matching column geometry, no header row on the candidate,
    and a non-empty candidate. A header on the candidate means a new table
    began, which is the single most reliable signal available.

    Args:
        previous: Last table on the earlier page.
        candidate: First table on the following page.

    Returns:
        True only when every condition holds.
    """
    if candidate.has_header or not candidate.rows:
        return False
    return _columns_match(previous, candidate)


def merge_multipage_tables(pages: Iterable[Page]) -> list[MergedTable]:
    """Merge tables that continue across consecutive pages of one document.

    Only the **last** table on a page may continue onto the **first** table of
    the next, and only when those pages are adjacent by ``page_index``. A gap in
    page indices -- which a resumed or partial parse can produce -- breaks the
    chain rather than silently welding pages 12 and 40 together.

    Args:
        pages: Pages of a single document, in any order; sorted here by
            ``page_index``.

    Returns:
        Every table in the document, continuations folded into their opener and
        anchored to the page the table started on.
    """
    ordered = sorted(pages, key=lambda p: (p.page_index if p.page_index is not None else -1))
    merged: list[MergedTable] = []
    #: Slot in ``merged`` still eligible for continuation, and the page index it
    #: currently ends on. ``None`` when the previous page closed the chain.
    open_slot: int | None = None
    open_page_index: int | None = None

    for page in ordered:
        if not page.tables:
            open_slot, open_page_index = None, None
            continue

        start = 0
        adjacent = (
            open_slot is not None
            and open_page_index is not None
            and page.page_index is not None
            and page.page_index == open_page_index + 1
        )
        if adjacent and open_slot is not None:
            existing = merged[open_slot]
            if is_continuation(existing.table, page.tables[0]):
                existing.table.rows.extend(page.tables[0].rows)
                merged[open_slot] = existing._replace(
                    page_range=_format_range(existing.page.page_label, page.page_label),
                    page_span=existing.page_span + 1,
                )
                start = 1

        for ordinal, table in enumerate(page.tables[start:], start=start):
            merged.append(MergedTable(table=table, page=page, page_range=None, ordinal=ordinal))

        # A page that only continued the open table leaves that same table open.
        # A page that also opened new ones passes the baton to its last.
        if len(page.tables) > start:
            open_slot = len(merged) - 1
        open_page_index = page.page_index

    return merged


def _format_range(first: str | None, last: str | None) -> str | None:
    """Format a printed page range, or refuse to.

    Returns ``None`` unless both labels are numeric and ascending. 38.7% of the
    corpus's page labels are guesses (build-plan section 4.5), and merges over
    guessed labels produce pairs like ``p9 -> p8`` or ``p3 -> p9``. A citation
    reading "pages 9-8" is worse than one naming only the anchor page, so a
    range that cannot be true is not shown.

    Args:
        first: Label of the page the table starts on.
        last: Label of the page it ends on.

    Returns:
        ``"42-44"``, or ``None`` when the range would be nonsense.
    """
    if not first or not last or first == last:
        return None
    if not (first.isdigit() and last.isdigit()):
        return None
    if int(last) <= int(first):
        return None
    return f"{first}-{last}"


def render_table(table: Table, caption: str = "") -> str:
    """Render a table as markdown, optionally captioned.

    Markdown rather than raw cells: the pipe layout keeps a fault code adjacent
    to its meaning in the embedded text, which is the association retrieval has
    to preserve.

    Args:
        table: The table to render.
        caption: Optional line placed above the table.

    Returns:
        Markdown text.
    """
    body = table.to_markdown()
    return f"{caption}\n{body}".strip() if caption else body


def split_oversized_table(table: Table, max_tokens: int = 6000, caption: str = "") -> list[str]:
    """Split a table that exceeds the embedding cap, repeating its header.

    Splits on row boundaries so no row is ever cut in half, and repeats the
    header in every part -- a part without its header is a grid of numbers with
    nothing to say what they mean.

    A single row that alone exceeds the cap is truncated rather than dropped.
    That is lossy and deliberate: dropping it loses the row entirely, and the
    alternative of an oversized part fails the embedding call for the whole
    chunk.

    Args:
        table: The table to render and split.
        max_tokens: Hard cap per part.
        caption: Caption repeated above each part.

    Returns:
        One or more markdown strings, each within ``max_tokens``.
    """
    if not table.rows:
        return []

    rendered = render_table(table, caption)
    if count_tokens(rendered) <= max_tokens:
        return [rendered]

    width = max(len(row) for row in table.rows)
    header_rows = table.rows[:1] if table.has_header else []
    body_rows = table.rows[1:] if table.has_header else table.rows

    prefix_lines: list[str] = []
    if caption:
        prefix_lines.append(caption)
    if header_rows:
        prefix_lines.append(_render_row(header_rows[0], width))
        prefix_lines.append("|" + "|".join(["---"] * width) + "|")
    prefix = "\n".join(prefix_lines)
    prefix_tokens = count_tokens(prefix) if prefix else 0

    parts: list[str] = []
    current: list[str] = []
    current_tokens = prefix_tokens

    for row in body_rows:
        line = _render_row(row, width)
        line_tokens = count_tokens(line)

        if prefix_tokens + line_tokens > max_tokens:
            # One row alone busts the cap. Flush what we have, then emit the
            # row truncated to fit rather than losing it.
            if current:
                parts.append(_join_part(prefix, current))
                current, current_tokens = [], prefix_tokens
            parts.append(truncate_to_tokens(_join_part(prefix, [line]), max_tokens))
            continue

        if current and current_tokens + line_tokens > max_tokens:
            parts.append(_join_part(prefix, current))
            current, current_tokens = [], prefix_tokens

        current.append(line)
        current_tokens += line_tokens

    if current:
        parts.append(_join_part(prefix, current))
    return parts


def _render_row(row: list[str], width: int) -> str:
    """Render one table row as a markdown line.

    Args:
        row: Cell text.
        width: Column count to pad to.

    Returns:
        A markdown table row.
    """
    padded = list(row) + [""] * (width - len(row))
    return "| " + " | ".join(cell.replace("\n", " ").strip() for cell in padded) + " |"


def _join_part(prefix: str, lines: list[str]) -> str:
    """Join a header prefix and body lines into one markdown part.

    Args:
        prefix: Caption and header lines, already rendered.
        lines: Body rows.

    Returns:
        The part's markdown.
    """
    return "\n".join(([prefix] if prefix else []) + lines)


def table_caption(page: Any) -> str:
    """Build the caption placed above a table chunk.

    Naming the table's document and page inside the embedded text means a query
    like "TQ fault code table" can match the caption even when the table body is
    nothing but codes and abbreviations.

    Args:
        page: The :class:`~seeley_rag.parse.base.Page` the table came from.

    Returns:
        A one-line caption.
    """
    label = f" (p.{page.page_label})" if page.page_label else ""
    return f"Table from {page.title}{label}:" if page.title else "Table:"
