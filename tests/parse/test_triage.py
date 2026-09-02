"""Tests for the Stage 0 PDF triage.

Test PDFs are synthesised with PyMuPDF rather than committed as binaries, so the
tier boundaries are driven by explicit character and image counts that a reader
can see rather than by opaque fixture files.
"""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from seeley_rag.exceptions import ParseError
from seeley_rag.parse.triage import (
    classify_page,
    render_report,
    triage_corpus,
    triage_pdf,
    write_report,
)
from seeley_rag.settings import Settings


def make_pdf(path: Path, pages: list[tuple[str, bool]]) -> Path:
    """Write a PDF with the requested pages.

    Args:
        path: Destination.
        pages: One ``(text, with_image)`` pair per page.

    Returns:
        The written path.
    """
    document = fitz.open()
    for text, with_image in pages:
        page = document.new_page()
        if text:
            page.insert_textbox(fitz.Rect(20, 20, 570, 800), text, fontsize=8)
        if with_image:
            pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 40, 40))
            pixmap.set_rect(pixmap.irect, (128, 128, 128))
            page.insert_image(fitz.Rect(300, 500, 340, 540), pixmap=pixmap)
    document.save(path)
    document.close()
    return path


class TestClassifyPage:
    """The tier boundaries."""

    def test_no_text_is_scanned(self, settings: Settings) -> None:
        """No usable text layer means a full vision transcription (Tier B)."""
        has_text, diagram, tier = classify_page(chars=10, n_images=1, settings=settings)
        assert has_text is False
        assert tier == "scanned"

    def test_text_with_image_is_diagram_heavy(self, settings: Settings) -> None:
        """Has text but picture-dominated: a vision caption is still needed (Tier C).

        This is the tier the build plan's v1 forgot, and forgetting it
        under-budgets vision by 2-3x.
        """
        has_text, diagram, tier = classify_page(chars=300, n_images=2, settings=settings)
        assert has_text is True
        assert diagram is True
        assert tier == "diagram_heavy"

    def test_lots_of_text_is_plain(self, settings: Settings) -> None:
        """A dense text page needs no vision call at all (Tier A)."""
        has_text, diagram, tier = classify_page(chars=2000, n_images=1, settings=settings)
        assert tier == "plain_text"

    def test_sparse_text_without_images_is_plain_not_diagram(self, settings: Settings) -> None:
        """A short page with no pictures has nothing for vision to caption."""
        _, diagram, tier = classify_page(chars=300, n_images=0, settings=settings)
        assert diagram is False
        assert tier == "plain_text"

    @pytest.mark.parametrize(
        ("chars", "expected"),
        [(100, "scanned"), (101, "diagram_heavy")],
    )
    def test_text_layer_threshold(self, settings: Settings, chars: int, expected: str) -> None:
        """The text-layer threshold is exclusive at 100 characters."""
        _, _, tier = classify_page(chars=chars, n_images=1, settings=settings)
        assert tier == expected


class TestTriagePdf:
    """Per-document triage."""

    def test_counts_pages_and_text(self, tmp_path: Path, settings: Settings) -> None:
        """Every page is inspected and its character count recorded."""
        pdf = make_pdf(tmp_path / "a.pdf", [("Hello " * 200, False), ("More " * 200, False)])
        result = triage_pdf(pdf, settings)
        assert result.ok
        assert result.page_count == 2
        assert len(result.pages) == 2
        assert all(p.chars > 100 for p in result.pages)
        assert all(p.tier == "plain_text" for p in result.pages)

    def test_blank_page_is_scanned(self, tmp_path: Path, settings: Settings) -> None:
        """A page with no text layer is exactly what Tier B is for."""
        pdf = make_pdf(tmp_path / "b.pdf", [("", False)])
        result = triage_pdf(pdf, settings)
        assert result.pages[0].tier == "scanned"
        assert result.pages[0].has_text_layer is False

    def test_page_index_is_zero_based(self, tmp_path: Path, settings: Settings) -> None:
        """Internal indices are 0-based; citations use the printed label instead."""
        pdf = make_pdf(tmp_path / "c.pdf", [("a" * 500, False), ("b" * 500, False)])
        result = triage_pdf(pdf, settings)
        assert [p.page_index for p in result.pages] == [0, 1]

    def test_unreadable_file_is_reported_not_raised(
        self, tmp_path: Path, settings: Settings
    ) -> None:
        """One corrupt manual must not stop the triage of the other five."""
        broken = tmp_path / "broken.pdf"
        broken.write_bytes(b"this is not a pdf")
        result = triage_pdf(broken, settings)
        assert result.ok is False
        assert result.error


class TestTriageCorpus:
    """Aggregation across documents."""

    def test_reports_all_three_fractions(self, tmp_path: Path, settings: Settings) -> None:
        """All three fractions, not just the scanned one.

        The vision budget is a function of scanned AND diagram-heavy; modelling
        only the former is what under-budgeted the build plan's v1 by 2-3x.
        """
        make_pdf(tmp_path / "a.pdf", [("x " * 600, False), ("", False)])
        make_pdf(tmp_path / "b.pdf", [("short text here " * 12, True)])

        _, summary = triage_corpus(sorted(tmp_path.glob("*.pdf")), settings)

        assert summary.total_pages == 3
        assert summary.plain_text_pages == 1
        assert summary.scanned_pages == 1
        assert summary.diagram_heavy_pages == 1
        assert summary.pct_scanned == pytest.approx(1 / 3)
        assert summary.pct_diagram_heavy == pytest.approx(1 / 3)
        assert summary.pct_plain_text == pytest.approx(1 / 3)

    def test_pct_vision_is_the_budget_number(self, tmp_path: Path, settings: Settings) -> None:
        """Both vision tiers together are what the cost estimate hangs off."""
        make_pdf(tmp_path / "a.pdf", [("x " * 600, False), ("", False)])
        _, summary = triage_corpus([tmp_path / "a.pdf"], settings)
        assert summary.pct_vision == pytest.approx(summary.pct_scanned + summary.pct_diagram_heavy)

    def test_failed_documents_are_counted_separately(
        self, tmp_path: Path, settings: Settings
    ) -> None:
        """A broken file is visible in the summary rather than silently absent."""
        make_pdf(tmp_path / "good.pdf", [("x " * 600, False)])
        (tmp_path / "bad.pdf").write_bytes(b"nope")
        _, summary = triage_corpus(sorted(tmp_path.glob("*.pdf")), settings)
        assert summary.documents == 1
        assert summary.failed_documents == 1

    def test_empty_input_raises_actionable_error(self, settings: Settings) -> None:
        """The message tells you to hand-download six manuals spanning the dates."""
        with pytest.raises(ParseError, match="hand-download|Hand-download"):
            triage_corpus([], settings)

    def test_no_pages_does_not_divide_by_zero(self, tmp_path: Path, settings: Settings) -> None:
        """An all-failed run still reports rather than crashing."""
        (tmp_path / "bad.pdf").write_bytes(b"nope")
        _, summary = triage_corpus([tmp_path / "bad.pdf"], settings)
        assert summary.pct_scanned == 0.0


class TestReport:
    """The markdown report."""

    def test_report_states_all_three_fractions(self, tmp_path: Path, settings: Settings) -> None:
        """The report is the artefact the budget decision is made from."""
        make_pdf(tmp_path / "a.pdf", [("x " * 600, False), ("", False)])
        documents, summary = triage_corpus([tmp_path / "a.pdf"], settings)
        report = render_report(documents, summary, "20260820T091422Z")
        assert "plain text" in report
        assert "diagram-heavy" in report
        assert "scanned" in report
        assert "Pages needing a vision call" in report

    def test_report_warns_when_vision_exceeds_half(
        self, tmp_path: Path, settings: Settings
    ) -> None:
        """Above ~50% the pilot needs rescoping; the report must say so."""
        make_pdf(tmp_path / "a.pdf", [("", False), ("", False)])
        documents, summary = triage_corpus([tmp_path / "a.pdf"], settings)
        report = render_report(documents, summary, "t")
        assert "Budget warning" in report

    def test_report_mentions_label_fallback_when_none_found(
        self, tmp_path: Path, settings: Settings
    ) -> None:
        """Absent labels mean citations fall back to the index; flag it explicitly."""
        make_pdf(tmp_path / "a.pdf", [("x " * 600, False)])
        documents, summary = triage_corpus([tmp_path / "a.pdf"], settings)
        report = render_report(documents, summary, "t")
        assert "page_index + 1" in report

    def test_write_report_creates_a_file(
        self, tmp_path: Path, temp_data_root: Path, settings: Settings
    ) -> None:
        """Reports land in data/reports/ with a UTC timestamp in the name."""
        make_pdf(tmp_path / "a.pdf", [("x " * 600, False)])
        documents, summary = triage_corpus([tmp_path / "a.pdf"], settings)
        path = write_report(documents, summary)
        assert path.exists()
        assert path.name.startswith("triage_")
        assert "# PDF triage report" in path.read_text(encoding="utf-8")
