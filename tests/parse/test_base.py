"""Tests for the Stage 2 page schema and metadata resolution.

Metadata resolution is the mechanism that stops a TQ fault code being answered
from a Climate Wizard manual -- build-plan section 13, risk 3, described there as
the failure that permanently destroys installer trust.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from seeley_rag.exceptions import ParseError
from seeley_rag.parse.base import (
    UNKNOWN_DOC_TYPE,
    UNKNOWN_FAMILY,
    Page,
    PagesWriter,
    Table,
    parsed_doc_ids,
    read_pages,
    resolve_doc_type,
    resolve_model_series,
    resolve_product_family,
)


class TestTable:
    """The table record."""

    def test_markdown_includes_a_header_separator(self) -> None:
        """Rendered tables must survive as tables in the embedded chunk text."""
        table = Table(rows=[["Code", "Meaning"], ["FC7", "Ignition failure"]], has_header=True)
        rendered = table.to_markdown()
        assert "| Code | Meaning |" in rendered
        assert "|---|---|" in rendered
        assert "| FC7 | Ignition failure |" in rendered

    def test_ragged_rows_are_padded(self) -> None:
        """PyMuPDF returns ragged rows; markdown needs rectangular ones."""
        table = Table(rows=[["a", "b", "c"], ["d"]])
        assert table.to_markdown().splitlines()[-1] == "| d |  |  |"

    def test_newlines_inside_cells_are_flattened(self) -> None:
        """A newline inside a cell would break the markdown row."""
        table = Table(rows=[["multi\nline"]])
        assert "\n" not in table.to_markdown().splitlines()[0]

    def test_empty_table_renders_empty(self) -> None:
        """Nothing detected means nothing rendered."""
        assert Table().to_markdown() == ""


class TestPage:
    """The pages.jsonl row schema."""

    def test_breadcrumb_names_the_product_and_page(self) -> None:
        """Rule 5 of build-plan section 5.1.

        The breadcrumb puts the product name into the embedded text of every
        chunk, including chunks whose body never names it.
        """
        page = Page(
            doc_id="abc",
            page_label="42",
            category="Ducted Gas Heating",
            folder="Service Guides",
            title="TQ Service Guide 644066-M",
        )
        assert page.breadcrumb() == (
            "Ducted Gas Heating > Service Guides > TQ Service Guide 644066-M > p.42"
        )

    def test_breadcrumb_omits_the_page_for_an_article(self) -> None:
        """A diagnostic article has no printed page to cite."""
        page = Page(doc_id="article:1", category="DGH", title="FC7 guide")
        assert page.breadcrumb() == "DGH > FC7 guide"

    def test_scanned_page_awaiting_vision_has_no_content(self) -> None:
        """Indexing it would create a citable chunk with nothing in it."""
        page = Page(doc_id="a", tier="scanned", needs_vision=True, text="")
        assert page.has_content is False

    def test_a_page_with_only_a_table_has_content(self) -> None:
        """A fault-code table with no prose is the most valuable page there is."""
        page = Page(doc_id="a", text="", tables=[Table(rows=[["FC7", "no flame"]])])
        assert page.has_content is True

    def test_unknown_fields_are_rejected(self) -> None:
        """The schema is closed so a typo fails loudly."""
        with pytest.raises(Exception):
            Page(doc_id="a", pge_label="42")  # type: ignore[call-arg]

    def test_model_series_is_a_real_field(self) -> None:
        """``model_`` collides with a pydantic protected namespace by default."""
        page = Page(doc_id="a", model_series=["TQ", "TQM"])
        assert page.model_series == ["TQ", "TQM"]


class TestResolveProductFamily:
    """Family resolution from category, folder and title."""

    def test_category_pattern_wins(self) -> None:
        """Category is the most reliable signal the portal gives us."""
        assert resolve_product_family("DUCTED GAS HEATING (DGH) SERVICE AND INSTALLATION") == "DGH"

    def test_reverse_cycle_category(self) -> None:
        """The second pilot category resolves too."""
        assert resolve_product_family("REVERSE CYCLE SERVICE AND INSTALLATION") == "RC"

    def test_alias_in_the_folder_name(self) -> None:
        """Folder names carry aliases when the category is generic."""
        assert resolve_product_family("Expert Advice", "Climate Wizard tips") == "EVAP"

    def test_model_code_in_the_title_as_a_last_resort(self) -> None:
        """Titles carry model codes; they are the weakest signal, so they are last."""
        assert resolve_product_family("Uncategorised", "", "CQ4 service manual") == "RC"

    def test_unknown_rather_than_a_guess(self) -> None:
        """A wrong family is worse than an absent one.

        Retrieval soft-boosts on this field, so a confident wrong answer routes
        a TQ question into the wrong product's manual.
        """
        assert resolve_product_family("Something Else Entirely") == UNKNOWN_FAMILY

    def test_longest_pattern_wins_over_a_nested_one(self) -> None:
        """VRF must not resolve to RC.

        "VRF REVERSE CYCLE SERVICE AND INSTALLATION" contains RC's "Reverse
        Cycle" pattern. With first-match ordering, all 111 VRF documents in this
        corpus were labelled RC -- exactly the cross-product contamination the
        lexicon exists to prevent, and invisible without a check like this.
        """
        assert resolve_product_family("VRF REVERSE CYCLE SERVICE AND INSTALLATION") == "VRF"
        assert resolve_product_family("REVERSE CYCLE SERVICE AND INSTALLATION") == "RC"

    def test_resolution_does_not_depend_on_yaml_ordering(self) -> None:
        """Longest-match makes the result independent of dict order.

        Relying on the order families happen to appear in the YAML is the kind of
        accidental dependency that produces a wrong answer nobody can explain.
        """
        vrf = resolve_product_family("VRF Reverse Cycle", "Service Manuals")
        assert vrf == "VRF"

    def test_vrf_model_codes_resolve_from_a_generic_category(self) -> None:
        """Expert-advice articles have no product category of their own.

        Their model codes are the only routing signal available.
        """
        family = resolve_product_family(
            "EXPERT ADVICE: MONTHLY HIGHLIGHTS", "", "E9 water overflow SDHV KDHV KDHA"
        )
        assert family == "VRF"

    def test_model_codes_match_whole_tokens_only(self) -> None:
        """ "TE" appears inside ordinary words; substring matching would misroute.

        "TEMPERATURE" must not resolve to the DGH "TE" series.
        """
        assert resolve_product_family("Unknown", "", "TEMPERATURE SETTINGS") == UNKNOWN_FAMILY


class TestResolveDocType:
    """Document-type resolution from the folder name."""

    @pytest.mark.parametrize(
        ("folder", "expected"),
        [
            ("Service Guides", "service_guide"),
            ("Installation Manuals", "installation"),
            ("Owners Manuals", "owner_manual"),
            ("Diagnostics and Specific Fault Finding", "fault_finding"),
            ("Spare Parts", "spare_parts"),
        ],
    )
    def test_real_folder_names(self, folder: str, expected: str) -> None:
        """Every folder name is taken from the live portal."""
        assert resolve_doc_type(folder) == expected

    def test_unknown_folder(self) -> None:
        """An unmatched folder is reported, not guessed."""
        assert resolve_doc_type("YouTube video tutorials") == UNKNOWN_DOC_TYPE

    def test_title_is_a_fallback(self) -> None:
        """Some folders are generic while the title is specific."""
        assert resolve_doc_type("Misc", "TQ Service Guide 644066") == "service_guide"


class TestResolveModelSeries:
    """Model-code extraction."""

    def test_extracts_known_codes(self) -> None:
        """Codes drive the hard filter when a user names a model explicitly."""
        assert "TQ" in resolve_model_series("TQ Service Guide Gas Ducted Heater 644066 M")

    def test_splits_on_slashes_and_hyphens(self) -> None:
        """Titles list several models as "TA5/TE5/TE4"."""
        found = resolve_model_series("TA5/TE5/TE4 SPI control system")
        assert {"TA5", "TE5", "TE4"} <= set(found)

    def test_no_substring_matches(self) -> None:
        """Whole tokens only. A wrong model code is a wrong answer."""
        assert resolve_model_series("TEMPERATURE and THERMOSTAT notes") == []

    def test_returns_no_duplicates(self) -> None:
        """A title repeating a code yields it once."""
        assert resolve_model_series("TQ TQ TQ").count("TQ") == 1


class TestPagesIO:
    """Reading and writing pages.jsonl."""

    def test_round_trip(self, temp_data_root: Path) -> None:
        """What the writer wrote, the reader reads back identically."""
        page = Page(doc_id="abc", page_index=0, page_label="3", text="hello", title="T")
        with PagesWriter() as writer:
            writer.write(page)

        restored = list(read_pages())
        assert len(restored) == 1
        assert restored[0].doc_id == "abc"
        assert restored[0].page_label == "3"

    def test_rows_are_flushed_immediately(self, temp_data_root: Path) -> None:
        """A long parse killed mid-run must keep what it already produced."""
        from seeley_rag.paths import PAGES_PATH

        with PagesWriter() as writer:
            writer.write(Page(doc_id="a"))
            assert PAGES_PATH.read_text(encoding="utf-8").strip()

    def test_one_json_object_per_line(self, temp_data_root: Path) -> None:
        """JSONL, so a crash leaves a readable file."""
        from seeley_rag.paths import PAGES_PATH

        with PagesWriter() as writer:
            writer.write_all([Page(doc_id="a"), Page(doc_id="b")])
        lines = PAGES_PATH.read_text(encoding="utf-8").strip().split("\n")
        assert [json.loads(line)["doc_id"] for line in lines] == ["a", "b"]

    def test_non_ascii_survives(self, temp_data_root: Path) -> None:
        """Titles carry en-dashes and warning glyphs."""
        with PagesWriter() as writer:
            writer.write(Page(doc_id="a", title="TQ – FC7 ⚠️"))
        assert list(read_pages())[0].title == "TQ – FC7 ⚠️"

    def test_write_outside_context_manager_raises(self, temp_data_root: Path) -> None:
        """Misuse fails loudly rather than dropping rows."""
        with pytest.raises(ParseError):
            PagesWriter().write(Page(doc_id="a"))

    def test_missing_file_names_the_command(self, temp_data_root: Path) -> None:
        """The error tells you what to run."""
        with pytest.raises(ParseError, match="03_parse"):
            list(read_pages())

    def test_parsed_doc_ids_supports_resume(self, temp_data_root: Path) -> None:
        """A re-run skips documents already parsed."""
        with PagesWriter() as writer:
            writer.write(Page(doc_id="doc-a", page_index=0))
            writer.write(Page(doc_id="doc-a", page_index=1))
            writer.write(Page(doc_id="doc-b", page_index=0))
        assert parsed_doc_ids() == {"doc-a", "doc-b"}

    def test_parsed_doc_ids_on_missing_file(self, temp_data_root: Path) -> None:
        """The first run has nothing parsed, which is not an error."""
        assert parsed_doc_ids() == set()

    def test_overwrite_truncates(self, temp_data_root: Path) -> None:
        """--overwrite starts from an empty file."""
        with PagesWriter() as writer:
            writer.write(Page(doc_id="a"))
        with PagesWriter(overwrite=True) as writer:
            writer.write(Page(doc_id="b"))
        assert [p.doc_id for p in read_pages()] == ["b"]
