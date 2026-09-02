"""Tests for printed page-label reconciliation.

build-plan.md section 4.5 calls this out as a failure mode that "looks exactly
like a retrieval bug", and triage confirmed it in this corpus: 3,608 of 12,526
pages carry an embedded label, with a most-common front-matter offset of -2.
"""

from __future__ import annotations

import fitz
import pytest

from seeley_rag.parse.pagelabels import (
    detect_offset,
    label_from_text,
    label_is_inferred,
    read_embedded_label,
    resolve_label,
    resolve_label_with_source,
)


class TestLabelFromText:
    """Reading a printed number out of a page's header or footer."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("body text here\n\nPage 42 of 118", "42"),
            ("body\nPage 7", "7"),
            ("body\n- 15 -", "15"),
            ("body\n– 15 –", "15"),
            ("23\nheader then body", "23"),
            ("body\n\n9\n", "9"),
        ],
    )
    def test_recognised_footers(self, text: str, expected: str) -> None:
        """Each footer style seen in these manuals resolves."""
        assert label_from_text(text) == expected

    def test_page_of_total_wins_over_a_bare_number(self) -> None:
        """The more specific pattern is stronger evidence and is tried first."""
        assert label_from_text("7\nsome body\nPage 42 of 118") == "42"

    def test_numbers_in_body_text_are_ignored(self) -> None:
        """A gas pressure or a fault code is not a page number.

        Only the outer few lines are considered, which is what keeps a
        measurement in the middle of a procedure from being cited as "p.250".
        """
        text = "\n".join(["header"] + ["set the manifold to 250 Pa"] * 10 + ["footer text"])
        assert label_from_text(text) is None

    def test_implausibly_large_numbers_are_rejected(self) -> None:
        """A part number or a year is not a page number."""
        assert label_from_text("body\n644066") is None

    def test_empty_text(self) -> None:
        """A scanned page has no text to read a label from."""
        assert label_from_text("") is None
        assert label_from_text("   \n  \n") is None

    def test_leading_zeros_are_normalised(self) -> None:
        """ "007" and "7" are the same page."""
        assert label_from_text("body\nPage 007") == "7"


class TestResolveLabel:
    """The resolution order and its provenance."""

    def test_embedded_label_wins(self) -> None:
        """The PDF's own label tree is the most trustworthy source."""
        label, source = resolve_label_with_source("body\nPage 9", 0, "iv")
        assert (label, source) == ("iv", "embedded")

    def test_footer_is_the_second_choice(self) -> None:
        """Absent an embedded label, the printed footer is still real evidence."""
        label, source = resolve_label_with_source("body\nPage 42 of 118", 10, None)
        assert (label, source) == ("42", "text")

    def test_index_fallback_is_marked_as_a_guess(self) -> None:
        """A guess must be recorded as one.

        Presenting index+1 as a printed label is what makes page accuracy fail
        corpus-wide while looking like a retrieval problem.
        """
        label, source = resolve_label_with_source("no footer at all", 41, None)
        assert (label, source) == ("42", "index")

    def test_resolve_label_never_returns_empty(self) -> None:
        """Callers always get something citable."""
        assert resolve_label("", 0, None) == "1"

    def test_label_is_inferred_flags_the_fallback(self) -> None:
        """A corpus of mostly-inferred labels cannot support the eval gate."""
        assert label_is_inferred("no footer", None) is True
        assert label_is_inferred("body\nPage 3", None) is False
        assert label_is_inferred("no footer", "iv") is False


class TestDetectOffset:
    """Front-matter offset detection."""

    @staticmethod
    def _pdf_with_footers(tmp_path, first_printed: int, pages: int = 6):
        """Write a PDF whose printed numbers start at ``first_printed``."""
        document = fitz.open()
        for n in range(pages):
            page = document.new_page()
            page.insert_textbox(fitz.Rect(20, 20, 570, 700), "body text " * 40, fontsize=9)
            page.insert_textbox(
                fitz.Rect(250, 780, 350, 800), f"Page {first_printed + n}", fontsize=9
            )
        path = tmp_path / "offset.pdf"
        document.save(path)
        document.close()
        return path

    def test_front_matter_offset_is_detected(self, tmp_path) -> None:
        """Printed page 3 at index 0 means an offset of +2.

        The real corpus shows the mirror case (-2): front matter that is counted
        by the viewer but not printed.
        """
        path = self._pdf_with_footers(tmp_path, first_printed=3)
        with fitz.open(path) as document:
            assert detect_offset(document) == 2

    def test_no_offset_when_printed_matches_index(self, tmp_path) -> None:
        """A document with no front matter has offset zero."""
        path = self._pdf_with_footers(tmp_path, first_printed=1)
        with fitz.open(path) as document:
            assert detect_offset(document) == 0

    def test_none_when_no_page_carries_a_label(self, tmp_path) -> None:
        """Absent any evidence, the answer is "unknown", not zero."""
        document = fitz.open()
        document.new_page()
        path = tmp_path / "bare.pdf"
        document.save(path)
        document.close()
        with fitz.open(path) as opened:
            assert detect_offset(opened) is None


def test_read_embedded_label_survives_a_non_pdf(tmp_path) -> None:
    """``get_label()`` asserts inside PyMuPDF on a non-PDF document.

    This actually happened: a login page stored with a .pdf extension took down
    a triage run over 544 documents with a bare AssertionError.
    """
    image = tmp_path / "not-a-pdf.png"
    pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 8, 8))
    pixmap.save(image)

    with fitz.open(image) as document:
        page = document[0]
        assert read_embedded_label(page) is None
