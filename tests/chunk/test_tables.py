"""Stage 3 table handling: the multi-page merge and the embedding cap.

build-plan section 5.1, rules 3 and 4. Both rules exist to protect fault-code
tables, which are the most valuable content in the corpus and the easiest to
destroy by chunking.
"""

from __future__ import annotations

from seeley_rag.chunk.tables import (
    BBOX_TOLERANCE_PT,
    is_continuation,
    merge_multipage_tables,
    render_table,
    split_oversized_table,
    table_caption,
)
from seeley_rag.chunk.tokens import count_tokens
from seeley_rag.parse.base import Page, Table

HEADER = ["Code", "Meaning", "Action"]


def make_table(
    rows: list[list[str]],
    has_header: bool = True,
    n_columns: int = 3,
    bbox: tuple[float, float, float, float] | None = (40.0, 100.0, 400.0, 500.0),
) -> Table:
    """Build a table with default geometry.

    Args:
        rows: Cell text, row-major.
        has_header: Whether ``rows[0]`` is a header.
        n_columns: Column count.
        bbox: Page geometry.

    Returns:
        The table.
    """
    return Table(rows=rows, has_header=has_header, n_columns=n_columns, bbox=bbox)


def make_page(index: int, label: str, tables: list[Table]) -> Page:
    """Build a page carrying tables.

    Args:
        index: 0-based page index.
        label: Printed page label.
        tables: Tables on the page.

    Returns:
        The page.
    """
    return Page(
        doc_id="d" * 64,
        page_index=index,
        page_label=label,
        label_source="embedded",
        text="",
        tables=tables,
        title="TQ Service Guide",
        category="Ducted Gas Heating",
        folder="Service Guides",
    )


class TestContinuationDetection:
    """A continuation must satisfy every condition, not most of them."""

    def test_headerless_matching_geometry_continues(self) -> None:
        """The intended case: same columns, no header on the follower."""
        first = make_table([HEADER, ["E:04", "Flame fault", "Check sensor"]])
        second = make_table([["E:05", "Ignition fault", "Check igniter"]], has_header=False)
        assert is_continuation(first, second)

    def test_a_header_row_means_a_new_table(self) -> None:
        """The single most reliable signal that a different table began."""
        first = make_table([HEADER, ["E:04", "Flame fault", "Check"]])
        second = make_table([HEADER, ["E:05", "Ignition fault", "Check"]])
        assert not is_continuation(first, second)

    def test_different_column_count_is_not_a_continuation(self) -> None:
        """Different geometry means a different table."""
        first = make_table([HEADER, ["E:04", "Flame", "Check"]])
        second = make_table([["a", "b"]], has_header=False, n_columns=2)
        assert not is_continuation(first, second)

    def test_missing_geometry_refuses_rather_than_guesses(self) -> None:
        """Column count alone is too weak to weld two tables together.

        A wrong merge cites the wrong page for half its rows, which is worse
        than the shredding the merge exists to fix.
        """
        first = make_table([HEADER, ["E:04", "Flame", "Check"]], bbox=None)
        second = make_table([["E:05", "Ignition", "Check"]], has_header=False, bbox=None)
        assert not is_continuation(first, second)

    def test_drift_within_tolerance_is_accepted(self) -> None:
        """Detector jitter of a few points must not break a real table."""
        first = make_table([HEADER, ["E:04", "Flame", "Check"]])
        drifted = make_table(
            [["E:05", "Ignition", "Check"]],
            has_header=False,
            bbox=(40.0 + BBOX_TOLERANCE_PT - 1, 100.0, 400.0, 500.0),
        )
        assert is_continuation(first, drifted)

    def test_drift_beyond_tolerance_is_rejected(self) -> None:
        """A genuinely different layout must not merge."""
        first = make_table([HEADER, ["E:04", "Flame", "Check"]])
        far = make_table(
            [["Capacity", "900", "960"]],
            has_header=False,
            bbox=(40.0 + BBOX_TOLERANCE_PT + 50, 100.0, 400.0, 500.0),
        )
        assert not is_continuation(first, far)

    def test_empty_candidate_is_not_a_continuation(self) -> None:
        """An empty table carries nothing to merge."""
        first = make_table([HEADER, ["E:04", "Flame", "Check"]])
        assert not is_continuation(first, make_table([], has_header=False))


class TestMultiPageMerge:
    """Rule 4: fault tables run 2-4 pages with the header only on the first."""

    def test_three_page_table_becomes_one_chunk_anchored_to_the_first(self) -> None:
        """The TQ fault table's real shape: header on p42, rows through p44."""
        pages = [
            make_page(41, "42", [make_table([HEADER, ["E:04", "Flame fault", "Check"]])]),
            make_page(42, "43", [make_table([["E:05", "Ignition", "Check"]], has_header=False)]),
            make_page(43, "44", [make_table([["E:06", "Fan", "Check"]], has_header=False)]),
        ]
        merged = merge_multipage_tables(pages)
        assert len(merged) == 1
        entry = merged[0]
        assert entry.page.page_index == 41, "must anchor to the page it started on"
        assert entry.page_span == 3
        assert entry.page_range == "42-44"
        flat = [cell for row in entry.table.rows for cell in row]
        assert "E:04" in flat and "E:05" in flat and "E:06" in flat

    def test_non_adjacent_pages_never_merge(self) -> None:
        """A gap in page indices breaks the chain rather than welding pages."""
        pages = [
            make_page(11, "12", [make_table([HEADER, ["E:04", "Flame", "Check"]])]),
            make_page(39, "40", [make_table([["E:05", "Ignition", "Check"]], has_header=False)]),
        ]
        merged = merge_multipage_tables(pages)
        assert len(merged) == 2
        assert all(entry.page_span == 1 for entry in merged)

    def test_a_page_without_tables_breaks_the_chain(self) -> None:
        """Content between two tables means they are not one table."""
        pages = [
            make_page(0, "1", [make_table([HEADER, ["E:04", "Flame", "Check"]])]),
            make_page(1, "2", []),
            make_page(2, "3", [make_table([["E:05", "Ignition", "Check"]], has_header=False)]),
        ]
        assert len(merge_multipage_tables(pages)) == 2

    def test_only_the_last_table_on_a_page_can_continue(self) -> None:
        """A table followed by another on the same page is closed."""
        pages = [
            make_page(
                0,
                "1",
                [
                    make_table([HEADER, ["E:04", "Flame", "Check"]]),
                    make_table([HEADER, ["Part", "Number", "Qty"]]),
                ],
            ),
            make_page(1, "2", [make_table([["611518", "80C", "1"]], has_header=False)]),
        ]
        merged = merge_multipage_tables(pages)
        assert len(merged) == 2
        # The continuation joined the second table, not the fault table.
        assert merged[0].page_span == 1
        assert merged[1].page_span == 2

    def test_descending_labels_yield_no_range(self) -> None:
        """38.7% of labels are guesses, and "pages 9-8" is worse than silence.

        The merge still happens -- page_index adjacency is trustworthy -- but
        the citation refuses to print a range that cannot be true.
        """
        pages = [
            make_page(8, "9", [make_table([HEADER, ["E:04", "Flame", "Check"]])]),
            make_page(9, "8", [make_table([["E:05", "Ignition", "Check"]], has_header=False)]),
        ]
        merged = merge_multipage_tables(pages)
        assert len(merged) == 1
        assert merged[0].page_span == 2
        assert merged[0].page_range is None

    def test_non_numeric_labels_yield_no_range(self) -> None:
        """Roman or prefixed labels cannot be ordered, so no range is claimed."""
        pages = [
            make_page(0, "P4", [make_table([HEADER, ["E:04", "Flame", "Check"]])]),
            make_page(1, "P5", [make_table([["E:05", "Ignition", "Check"]], has_header=False)]),
        ]
        merged = merge_multipage_tables(pages)
        assert merged[0].page_span == 2
        assert merged[0].page_range is None

    def test_empty_input(self) -> None:
        """No pages, no tables, no error."""
        assert merge_multipage_tables([]) == []


class TestOversizedTables:
    """Rule 3: cap at ~6,000 tokens, split on rows, repeat the header."""

    def test_small_table_is_not_split(self) -> None:
        """Atomicity is the default; splitting is the exception."""
        table = make_table([HEADER, ["E:04", "Flame fault", "Check sensor"]])
        assert len(split_oversized_table(table, max_tokens=6000)) == 1

    def test_oversized_table_splits_and_repeats_the_header(self) -> None:
        """A part without its header is a grid of numbers with no meaning."""
        rows = [HEADER] + [
            [f"E:{i:02d}", f"Fault number {i}", "Check the sensor"] for i in range(400)
        ]
        parts = split_oversized_table(make_table(rows), max_tokens=300)
        assert len(parts) > 1
        for part in parts:
            assert "Code" in part and "Meaning" in part, "header missing from a part"
            assert count_tokens(part) <= 300

    def test_split_never_cuts_a_row_in_half(self) -> None:
        """Row boundaries are the split points, so no cell is orphaned."""
        rows = [HEADER] + [[f"E:{i:02d}", f"Fault {i}", "Action"] for i in range(200)]
        parts = split_oversized_table(make_table(rows), max_tokens=200)
        rejoined = "\n".join(parts)
        for i in range(200):
            assert f"E:{i:02d}" in rejoined, f"row {i} lost in the split"

    def test_a_single_giant_row_is_truncated_not_dropped(self) -> None:
        """Lossy on purpose: dropping loses the row, oversize fails the call."""
        rows = [HEADER, ["E:04", "word " * 5000, "Check"]]
        parts = split_oversized_table(make_table(rows), max_tokens=200)
        assert parts, "the row must survive in some form"
        for part in parts:
            assert count_tokens(part) <= 200

    def test_empty_table_yields_nothing(self) -> None:
        """An empty table is not a chunk."""
        assert split_oversized_table(make_table([]), max_tokens=100) == []


class TestRendering:
    """The rendered form is what gets embedded, so it must keep pairs adjacent."""

    def test_markdown_keeps_a_code_beside_its_meaning(self) -> None:
        """The association retrieval has to preserve."""
        rendered = render_table(make_table([HEADER, ["E:04", "Flame sensing fault", "Check"]]))
        code_line = next(line for line in rendered.split("\n") if "E:04" in line)
        assert "Flame sensing fault" in code_line

    def test_caption_names_the_document_and_page(self) -> None:
        """Lets "TQ fault code table" match a table body full of bare codes."""
        page = make_page(41, "42", [])
        caption = table_caption(page)
        assert "TQ Service Guide" in caption
        assert "p.42" in caption
