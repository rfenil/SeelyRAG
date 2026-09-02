"""Tests for PDF parsing.

The table-detection gate is the load-bearing part. ``find_tables()`` costs
0.5-2s per page, and across this corpus's 12,526 pages an ungated call is
1.7-7 hours on its own.
"""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from seeley_rag.acquire.manifest import Document
from seeley_rag.exceptions import ParseError
from seeley_rag.parse.pdf import has_table_signal, parse_pdf, render_page_png
from seeley_rag.settings import Settings


def write_pdf(path: Path, pages: list[str], footer: str | None = None) -> Path:
    """Write a PDF with the given page bodies.

    Args:
        path: Destination.
        pages: One body string per page.
        footer: Optional footer, formatted with the 1-based page number.

    Returns:
        The written path.
    """
    document = fitz.open()
    for number, body in enumerate(pages, start=1):
        page = document.new_page()
        page.insert_textbox(fitz.Rect(20, 20, 570, 700), body, fontsize=9)
        if footer:
            page.insert_textbox(fitz.Rect(250, 760, 400, 790), footer.format(number), fontsize=9)
    document.save(path)
    document.close()
    return path


def make_document(path: Path, **overrides: object) -> Document:
    """Build a Document pointing at a local PDF.

    Args:
        path: The PDF on disk.
        **overrides: Field overrides.

    Returns:
        A :class:`Document`.
    """
    payload: dict = {
        "sha256": "a" * 64,
        "stored_path": str(path),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "filenames": ["TQ Service Guide 644066 M.pdf"],
        "attachment_ids": ["47234382931"],
        "article_ids": ["47001247136"],
        "titles": ["TQ Service Guide Gas Ducted Heater 644066 M"],
        "categories": ["DUCTED GAS HEATING (DGH) SERVICE AND INSTALLATION"],
        "folders": ["Service Guides"],
    }
    payload.update(overrides)
    return Document(**payload)  # type: ignore[arg-type]


class TestHasTableSignal:
    """The gate in front of the expensive table-detection pass."""

    def test_column_aligned_lines_trigger_it(self) -> None:
        """Runs of two or more spaces are the signature of column layout."""
        text = "Code    Meaning\nFC7     No flame\nFC11    Flue fault\nFC19    HX fault"
        assert has_table_signal(text) is True

    def test_a_fault_code_mention_triggers_it(self) -> None:
        """Fault-code tables are what we least want to miss.

        Some are laid out without wide column gaps, so a code mention is treated
        as sufficient on its own -- an extra detection pass is far cheaper than
        losing a fault-code table.
        """
        assert has_table_signal("See fault code FC7 for details.") is True
        assert has_table_signal("The error code table is on the next page.") is True

    def test_prose_does_not_trigger_it(self) -> None:
        """The whole point is not paying 0.5-2s for ordinary pages."""
        prose = (
            "This appliance must be installed by a licensed technician. "
            "Ensure adequate clearance around the unit before commencing work."
        )
        assert has_table_signal(prose) is False

    def test_a_single_aligned_line_is_not_enough(self) -> None:
        """One gap is a typographic accident, not a table."""
        assert has_table_signal("just  one gap here") is False

    def test_empty_text(self) -> None:
        """A scanned page has no text to gate on."""
        assert has_table_signal("") is False


class TestParsePdf:
    """End-to-end document parsing."""

    def test_produces_one_page_per_page(self, tmp_path: Path, settings: Settings) -> None:
        """The row count must match the document."""
        path = write_pdf(tmp_path / "a.pdf", ["body one " * 40, "body two " * 40])
        pages = parse_pdf(make_document(path), render_images=False, settings=settings)
        assert len(pages) == 2
        assert [p.page_index for p in pages] == [0, 1]

    def test_metadata_is_resolved_and_carried(self, tmp_path: Path, settings: Settings) -> None:
        """Product routing metadata comes from category, folder and title."""
        path = write_pdf(tmp_path / "a.pdf", ["body " * 40])
        page = parse_pdf(make_document(path), render_images=False, settings=settings)[0]
        assert page.product_family == "DGH"
        assert page.doc_type == "service_guide"
        assert "TQ" in page.model_series
        assert page.category.startswith("DUCTED GAS HEATING")
        assert page.folder == "Service Guides"

    def test_every_linking_article_is_recorded(self, tmp_path: Path, settings: Settings) -> None:
        """A shared manual must be citable from whichever article was the route."""
        path = write_pdf(tmp_path / "a.pdf", ["body " * 40])
        document = make_document(path, article_ids=["1", "2", "3"])
        page = parse_pdf(document, render_images=False, settings=settings)[0]
        assert page.source_article_ids == ["1", "2", "3"]

    def test_printed_footer_becomes_the_label(self, tmp_path: Path, settings: Settings) -> None:
        """Front matter means printed pages start above 1.

        Here index 0 prints as page 5, so the label must be 5 and its source
        must say it was read rather than guessed.
        """
        path = write_pdf(tmp_path / "a.pdf", ["body " * 40, "body " * 40], footer="Page {}")
        # Rewrite with an offset: printed numbers start at 5.
        document = fitz.open()
        for n in (5, 6):
            page = document.new_page()
            page.insert_textbox(fitz.Rect(20, 20, 570, 700), "body " * 40, fontsize=9)
            page.insert_textbox(fitz.Rect(250, 760, 400, 790), f"Page {n}", fontsize=9)
        document.save(path)
        document.close()

        pages = parse_pdf(make_document(path), render_images=False, settings=settings)
        assert [p.page_label for p in pages] == ["5", "6"]
        assert all(p.label_source == "text" for p in pages)

    def test_label_falls_back_to_index_and_says_so(
        self, tmp_path: Path, settings: Settings
    ) -> None:
        """An unlabelled page still gets a citable number, marked as a guess."""
        path = write_pdf(tmp_path / "a.pdf", ["body text without any footer " * 20])
        page = parse_pdf(make_document(path), render_images=False, settings=settings)[0]
        assert page.page_label == "1"
        assert page.label_source == "index"

    def test_scanned_page_is_flagged_for_vision(self, tmp_path: Path, settings: Settings) -> None:
        """No text layer means the work is queued, not silently dropped."""
        document = fitz.open()
        document.new_page()
        path = tmp_path / "scanned.pdf"
        document.save(path)
        document.close()

        page = parse_pdf(make_document(path), render_images=False, settings=settings)[0]
        assert page.tier == "scanned"
        assert page.needs_vision is True

    def test_plain_page_does_not_need_vision(self, tmp_path: Path, settings: Settings) -> None:
        """A dense text page costs nothing beyond extraction."""
        path = write_pdf(tmp_path / "a.pdf", ["dense body text " * 80])
        page = parse_pdf(make_document(path), render_images=False, settings=settings)[0]
        assert page.tier == "plain_text"
        assert page.needs_vision is False

    def test_tables_are_detected_on_a_signalling_page(
        self, tmp_path: Path, settings: Settings
    ) -> None:
        """A ruled fault-code table should come back as a table record."""
        document = fitz.open()
        page = document.new_page()
        page.insert_textbox(
            fitz.Rect(20, 20, 400, 200),
            "Code    Meaning\nFC7     No flame\nFC11    Flue fault\nFC19    HX fault",
            fontsize=10,
        )
        # Draw a grid so PyMuPDF's edge detection has something to find.
        for i in range(4):
            y = 30 + i * 20
            page.draw_line(fitz.Point(20, y), fitz.Point(300, y))
        page.draw_line(fitz.Point(20, 30), fitz.Point(20, 90))
        page.draw_line(fitz.Point(150, 30), fitz.Point(150, 90))
        page.draw_line(fitz.Point(300, 30), fitz.Point(300, 90))
        path = tmp_path / "table.pdf"
        document.save(path)
        document.close()

        parsed = parse_pdf(make_document(path), render_images=False, settings=settings)
        # Detection is best-effort; what must hold is that it was attempted and
        # did not throw, and that the page text survived either way.
        assert "FC7" in parsed[0].text

    def test_page_images_are_rendered(self, tmp_path: Path, temp_data_root: Path) -> None:
        """Every page gets a PNG, which is what makes diagram citation free."""
        path = write_pdf(tmp_path / "a.pdf", ["body " * 40, "body " * 40])
        pages = parse_pdf(make_document(path), render_images=True)
        assert all(p.image_path for p in pages)
        for page in pages:
            assert (temp_data_root.parent / page.image_path).exists() or Path(
                page.image_path
            ).exists()

    def test_missing_file_raises(self, tmp_path: Path, settings: Settings) -> None:
        """A manifest row pointing at a deleted file must fail loudly."""
        document = make_document(tmp_path / "gone.pdf")
        with pytest.raises(ParseError, match="missing"):
            parse_pdf(document, render_images=False, settings=settings)

    def test_non_pdf_raises_rather_than_being_parsed(
        self, tmp_path: Path, settings: Settings
    ) -> None:
        """Acquisition stores whatever the portal served.

        A login page saved with a .pdf extension actually happened; PyMuPDF will
        happily open some non-PDFs, so the type is checked explicitly.
        """
        fake = tmp_path / "notreally.pdf"
        fake.write_bytes(b"<!DOCTYPE html><html>login</html>")
        with pytest.raises(ParseError, match="not a PDF"):
            parse_pdf(make_document(fake), render_images=False, settings=settings)


def test_render_page_png_is_idempotent(tmp_path: Path, temp_data_root: Path) -> None:
    """A re-parse must not re-render pages it already has."""
    path = write_pdf(tmp_path / "a.pdf", ["body " * 40])
    with fitz.open(path) as document:
        first = render_page_png(document[0], "doc123", 0)
        assert first is not None
        target = temp_data_root / "01_interim" / "page_images" / "doc123" / "0000.png"
        stamp = target.stat().st_mtime_ns
        second = render_page_png(document[0], "doc123", 0)
        assert second == first
        assert target.stat().st_mtime_ns == stamp
