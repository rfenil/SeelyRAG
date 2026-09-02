"""Stage 3 fault-code extraction.

build-plan section 5.3. Each filter here exists because of specific junk a first
pass over the real corpus produced, so each has a test naming that junk. Delete
a filter and its test tells you exactly what comes back.

Precision is the priority: a missed code still reaches the installer through
hybrid retrieval, while a wrong one is *pinned ahead* of it.
"""

from __future__ import annotations

import pytest

from seeley_rag.chunk.base import Chunk
from seeley_rag.chunk.codes import (
    annotate_chunks,
    build_code_table,
    codes_by_key,
    extract_codes,
    is_code_like,
    is_usable_meaning,
    looks_corrupt,
    normalise_code,
    sweep_text,
)


def make_chunk(text: str, family: str = "DGH", chunk_id: str = "c1") -> Chunk:
    """Build a chunk carrying the given text.

    Args:
        text: Chunk text.
        family: Product family.
        chunk_id: Identifier.

    Returns:
        A finalised chunk.
    """
    return Chunk(
        chunk_id=chunk_id,
        doc_id="d" * 64,
        text=text,
        product_family=family,
        title="TQ Service Guide",
        page_label="42",
    ).finalise()


class TestNormalisation:
    """Every printed spelling of one code must answer the same query."""

    @pytest.mark.parametrize("printed", ["E:04", "E 04", "E-04", "E04", "e.04", "E4"])
    def test_separator_and_case_variants_collapse(self, printed: str) -> None:
        """E:04, E 04, E-04 and E4 are one code."""
        assert normalise_code(printed) == "E04"

    def test_digits_are_zero_padded(self) -> None:
        """Otherwise 'Fault Code 2' and 'Fault Code 02' become two codes."""
        assert normalise_code("2", from_phrase=True) == "FC02"
        assert normalise_code("02", from_phrase=True) == "FC02"

    def test_bare_number_from_a_phrase_gets_the_fc_prefix(self) -> None:
        """The manuals print "Fault Code 08"; installers say "FC8"."""
        assert normalise_code("8", from_phrase=True) == "FC08"

    def test_bare_number_outside_a_phrase_is_rejected(self) -> None:
        """Standing alone a number is a quantity, a page or a dimension."""
        assert normalise_code("8", from_phrase=False) == ""

    def test_flash_codes_cannot_collide_with_letter_codes(self) -> None:
        """FLASH3 and E:03 are different faults and must key differently."""
        assert normalise_code("3 flashes") == "FLASH3"
        assert normalise_code("3 flashes") != normalise_code("E:03")


class TestCandidateFiltering:
    """Filter 1: the phrase patterns capture whatever word follows."""

    @pytest.mark.parametrize("token", ["E4", "FC53", "b5", "08", "H12"])
    def test_code_shaped_tokens_are_accepted(self, token: str) -> None:
        """Real codes must survive the filter."""
        assert is_code_like(token)

    @pytest.mark.parametrize(
        "word",
        ["access", "chart", "column", "definition", "displayed", "does", "history", "Braemar"],
    )
    def test_words_the_first_pass_wrongly_admitted_are_rejected(self, word: str) -> None:
        """Every one of these became a "fault code" in the first corpus pass."""
        assert not is_code_like(word)

    def test_phrase_capture_of_an_ordinary_word_yields_nothing(self) -> None:
        """End to end: the sentence that produced a code named "history"."""
        assert extract_codes("The fault code history can be accessed from settings.") == []


class TestFaultContext:
    """Filter 2: [EFH]\\d{1,2} matches dimensions and part numbers."""

    def test_code_in_fault_context_is_kept(self) -> None:
        """The intended case."""
        assert "E04" in extract_codes("The unit reports fault E:04 and locks out.")

    def test_code_without_fault_vocabulary_is_discarded(self) -> None:
        """ "F 12" in a dimensions table is not a fault code."""
        assert extract_codes("Duct sizes are F 12 and H 10 respectively.") == []

    def test_flash_code_is_recognised(self) -> None:
        """DGH flash codes are a real diagnostic channel."""
        assert "FLASH3" in extract_codes("The LED shows 3 flashes to indicate the fault.")


class TestMeaningExtraction:
    """Filters 3 and 4: the adjacent cell, and no contents-page filler."""

    def test_meaning_comes_from_the_cell_beside_the_code(self) -> None:
        """Wide tables repeat 'code | meaning' across one row.

        Joining every other cell welded four codes' meanings into each one.
        """
        row = "| E0 | Malfunction of ODU | E1 | High pressure protection |"
        hits = {hit.code_key: hit.meaning for hit in sweep_text(row)}
        assert hits["E00"] == "Malfunction of ODU"
        assert hits["E01"] == "High pressure protection"
        assert "High pressure" not in hits["E00"], "neighbouring code's meaning leaked in"

    def test_contents_page_filler_is_rejected(self) -> None:
        """ "FAULT CODE 08 EXAMPLE......." is a table of contents, not a meaning."""
        assert not is_usable_meaning("FAULT CODE 08 EXAMPLE.........................45")

    def test_empty_meaning_is_rejected(self) -> None:
        """A code row with nothing to say has no value for pinning."""
        assert not is_usable_meaning("")
        assert not is_usable_meaning("  ")

    def test_prose_meaning_is_the_containing_sentence(self) -> None:
        """Prose falls back to the sentence, taken verbatim."""
        text = "Fault code 30 means the loom ID does not match the heater model."
        hits = sweep_text(text)
        assert hits
        assert "loom ID does not match" in hits[0].meaning


class TestCorruptionDetection:
    """Filter on the Stage 2 defect that shows through into meanings."""

    def test_interleaved_columns_are_detected(self) -> None:
        """ "Full water protection" woven into itself is not a meaning."""
        assert looks_corrupt("Full wFautlel rw partoetre pcrtiootnection")

    def test_broken_cmap_output_is_detected(self) -> None:
        """A shifted alphabet from a broken ToUnicode CMap."""
        assert looks_corrupt("(QVXUHWKHPRWRUSRZHUFDEOHLVILWWHGFRUUHFWO")

    def test_legitimate_meanings_survive(self) -> None:
        """The filter must not eat real fault descriptions."""
        for good in (
            "High air discharge temperature protection of compressor",
            "Indoor unit full water error",
            "ODU Ambient Temperature sensor failure",
            "PLC - PCBA COMMUNICATION FAILURE Cooler PLC has lost communication",
        ):
            assert not looks_corrupt(good), good

    def test_a_corrupt_sighting_does_not_lose_the_code(self) -> None:
        """Rejecting at the occurrence level lets a clean sighting win.

        This is why the check lives in the meaning filter rather than being
        applied to finished rows.
        """
        chunks = [
            make_chunk("| E9 | Full wFautlel rw partoetre pcrtiootnection |", chunk_id="bad"),
            make_chunk("| E9 | Indoor unit full water error |", chunk_id="good"),
        ]
        rows = build_code_table(chunks)
        assert len(rows) == 1
        assert rows[0].code_key == "E09"
        assert rows[0].meaning == "Indoor unit full water error"


class TestCodeTable:
    """The lookup table itself."""

    def test_codes_are_scoped_by_product_family(self) -> None:
        """E:04 on a gas heater is not E:04 on a VRF unit.

        Collapsing them would answer a DGH question from a VRF manual.
        """
        rows = build_code_table(
            [
                make_chunk("| E4 | Flame sensing fault |", family="DGH", chunk_id="a"),
                make_chunk(
                    "| E4 | High discharge temperature protection of compressor |",
                    family="VRF",
                    chunk_id="b",
                ),
            ]
        )
        assert len(rows) == 2
        assert {r.product_family for r in rows} == {"DGH", "VRF"}

    def test_table_evidence_beats_prose(self) -> None:
        """A table row is far stronger evidence than a passing mention."""
        rows = build_code_table(
            [
                make_chunk("The fault E:04 is described elsewhere in this guide.", chunk_id="p"),
                make_chunk("| E4 | Flame sensing fault, no flame detected |", chunk_id="t"),
            ]
        )
        assert len(rows) == 1
        assert rows[0].in_table
        assert "Flame sensing fault" in rows[0].meaning

    def test_rows_carry_citation_provenance(self) -> None:
        """A pinned code must be citable, or it cannot be shown to an installer."""
        rows = build_code_table([make_chunk("| E4 | Flame sensing fault |", chunk_id="c9")])
        assert rows[0].chunk_id == "c9"
        assert rows[0].page_label == "42"
        assert rows[0].title == "TQ Service Guide"

    def test_grouping_by_key_keeps_every_family(self) -> None:
        """One key can legitimately map to several families."""
        rows = build_code_table(
            [
                make_chunk("| E4 | Flame sensing fault |", family="DGH", chunk_id="a"),
                make_chunk(
                    "| E4 | High discharge temperature protection of compressor |",
                    family="VRF",
                    chunk_id="b",
                ),
            ]
        )
        assert len(codes_by_key(rows)["E04"]) == 2

    def test_empty_corpus_yields_an_empty_table(self) -> None:
        """No chunks, no codes, no error."""
        assert build_code_table([]) == []


class TestAnnotation:
    """Chunks carry their own codes so retrieval can pin on a match."""

    def test_codes_are_stamped_onto_the_chunk(self) -> None:
        """The field has to travel on the chunk, not only in the lookup table."""
        chunks = annotate_chunks([make_chunk("Fault E:04 and fault E:05 are both listed.")])
        assert chunks[0].fault_codes == ["E04", "E05"]

    def test_a_chunk_without_codes_gets_an_empty_list(self) -> None:
        """Absence must be recorded as absence, not left unset."""
        assert annotate_chunks([make_chunk("General installation guidance.")])[0].fault_codes == []
