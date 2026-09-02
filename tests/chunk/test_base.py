"""Stage 3 I/O and the incremental-index primitives.

``chunk_id`` determinism plus ``content_hash`` are what make a re-index cheap.
They are the mechanism by which the 3,459 pages awaiting vision can be folded in
later as an update rather than a rebuild, so they are tested as a contract
rather than as an implementation detail.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from seeley_rag.chunk.base import (
    Chunk,
    FaultCode,
    JsonlWriter,
    chunk_hashes,
    content_hash,
    make_chunk_id,
    read_chunks,
    read_codes,
)
from seeley_rag.exceptions import ParseError


def make_chunk(chunk_id: str = "d:p0:c00", text: str = "Body text.") -> Chunk:
    """Build a finalised chunk.

    Args:
        chunk_id: Identifier.
        text: Chunk text.

    Returns:
        The chunk, with hash and token count stamped.
    """
    return Chunk(chunk_id=chunk_id, doc_id="d" * 64, text=text).finalise()


class TestHashing:
    """The hash is the embedding cache key and the change detector."""

    def test_identical_text_hashes_identically(self) -> None:
        """Cache hits depend on this."""
        assert content_hash("same text") == content_hash("same text")

    def test_different_text_hashes_differently(self) -> None:
        """So does change detection."""
        assert content_hash("one") != content_hash("two")

    def test_finalise_stamps_hash_and_tokens(self) -> None:
        """A chunk reaches the index stage already measured."""
        chunk = make_chunk(text="Flame sensing fault on the TQ series.")
        assert chunk.content_hash == content_hash(chunk.text)
        assert chunk.token_count > 0

    def test_finalise_accepts_a_precomputed_count(self) -> None:
        """Avoids tokenising text the caller has already measured."""
        assert make_chunk().finalise(token_count=99).token_count == 99


class TestChunkIds:
    """Deterministic ids let the store upsert instead of being refilled."""

    def test_pdf_page_id_shape(self) -> None:
        """Document, page and ordinal, all recoverable."""
        assert make_chunk_id("abc", 41, 0) == "abc:p41:c00"

    def test_article_id_marks_the_absent_page(self) -> None:
        """Articles have no printed page, and that must not read as page 0."""
        assert make_chunk_id("article:123", None, 0) == "article:123:pna:c00"

    def test_ordinals_are_zero_padded_so_ids_sort(self) -> None:
        """c02 before c10, which a bare integer would get wrong."""
        ids = [make_chunk_id("d", 0, i) for i in (2, 10)]
        assert sorted(ids) == ids


class TestIncrementalIndexing:
    """chunk_hashes partitions a re-chunk into unchanged, new and gone."""

    def test_missing_file_is_an_empty_map_not_an_error(self, tmp_path: Path) -> None:
        """A first run has nothing to compare against."""
        assert chunk_hashes(tmp_path / "absent.jsonl") == {}

    def test_hashes_round_trip(self, tmp_path: Path) -> None:
        """What was written is what comes back."""
        path = tmp_path / "chunks.jsonl"
        chunks = [make_chunk("a:p0:c00", "First."), make_chunk("a:p1:c00", "Second.")]
        with JsonlWriter(path) as writer:
            writer.write_all(chunks)
        assert chunk_hashes(path) == {c.chunk_id: c.content_hash for c in chunks}

    def test_unchanged_chunks_are_recognised_across_runs(self, tmp_path: Path) -> None:
        """The property the whole incremental design rests on.

        Re-chunking untouched pages must produce byte-identical ids and hashes,
        or every re-index re-embeds the entire corpus.
        """
        path = tmp_path / "chunks.jsonl"
        with JsonlWriter(path) as writer:
            writer.write_all([make_chunk("a:p0:c00", "Unchanged body.")])
        before = chunk_hashes(path)
        rebuilt = make_chunk("a:p0:c00", "Unchanged body.")
        assert before[rebuilt.chunk_id] == rebuilt.content_hash

    def test_edited_text_is_detected_as_changed(self, tmp_path: Path) -> None:
        """Same id, different hash: embed this one, keep the rest."""
        path = tmp_path / "chunks.jsonl"
        with JsonlWriter(path) as writer:
            writer.write_all([make_chunk("a:p0:c00", "Original body.")])
        before = chunk_hashes(path)
        edited = make_chunk("a:p0:c00", "Edited body.")
        assert edited.chunk_id in before
        assert before[edited.chunk_id] != edited.content_hash

    def test_a_truncated_row_is_skipped_not_fatal(self, tmp_path: Path) -> None:
        """A kill mid-write must not make the whole map unreadable."""
        path = tmp_path / "chunks.jsonl"
        with JsonlWriter(path) as writer:
            writer.write_all([make_chunk("a:p0:c00", "Complete row.")])
        with path.open("a", encoding="utf-8") as handle:
            handle.write('{"chunk_id": "b:p0:c00", "content_ha')
        assert list(chunk_hashes(path)) == ["a:p0:c00"]


class TestJsonlIO:
    """Writing and streaming the Stage 3 outputs."""

    def test_writer_requires_a_context_manager(self, tmp_path: Path) -> None:
        """Writing to an unopened handle must fail loudly."""
        with pytest.raises(ParseError, match="context manager"):
            JsonlWriter(tmp_path / "x.jsonl").write(make_chunk())

    def test_chunks_round_trip_with_every_field(self, tmp_path: Path) -> None:
        """Provenance must survive serialisation, or citations break."""
        path = tmp_path / "chunks.jsonl"
        original = Chunk(
            chunk_id="a:p41:c00",
            doc_id="d" * 64,
            text="Table from TQ Service Guide (p.42):",
            kind="table",
            page_index=41,
            page_label="42",
            label_source="embedded",
            page_range="42-44",
            page_span=3,
            fault_codes=["E04"],
            model_series=["TQ"],
            product_family="DGH",
        ).finalise()
        with JsonlWriter(path) as writer:
            writer.write(original)
        restored = list(read_chunks(path))[0]
        assert restored == original
        assert restored.is_table
        assert restored.page_span == 3

    def test_utf8_titles_survive(self, tmp_path: Path) -> None:
        """Windows defaults to cp1252 and would corrupt article titles."""
        path = tmp_path / "chunks.jsonl"
        chunk = Chunk(
            chunk_id="a:p0:c00",
            doc_id="d",
            text="MDHX – VRF Multi Split ⚠️",
            title="Braemar – café",
        ).finalise()
        with JsonlWriter(path) as writer:
            writer.write(chunk)
        assert list(read_chunks(path))[0].title == "Braemar – café"

    def test_codes_round_trip(self, tmp_path: Path) -> None:
        """The lookup table is written and read with the same schema."""
        path = tmp_path / "codes.jsonl"
        row = FaultCode(
            code="E:04",
            code_key="E04",
            meaning="Flame sensing fault",
            product_family="DGH",
            in_table=True,
        )
        with JsonlWriter(path) as writer:
            writer.write(row)
        assert list(read_codes(path))[0] == row

    def test_blank_lines_are_skipped(self, tmp_path: Path) -> None:
        """Trailing newlines must not become a validation error."""
        path = tmp_path / "chunks.jsonl"
        with JsonlWriter(path) as writer:
            writer.write(make_chunk())
        with path.open("a", encoding="utf-8") as handle:
            handle.write("\n\n")
        assert len(list(read_chunks(path))) == 1

    def test_missing_file_names_the_command_to_run(self, tmp_path: Path) -> None:
        """The error has to tell you what to do about it."""
        with pytest.raises(ParseError, match="04_index"):
            list(read_chunks(tmp_path / "absent.jsonl"))

    def test_a_malformed_row_is_fatal_and_located(self, tmp_path: Path) -> None:
        """Silent skipping would hide corruption; the line number is named."""
        path = tmp_path / "chunks.jsonl"
        path.write_text('{"not": "a chunk"}\n', encoding="utf-8")
        with pytest.raises(ParseError, match=":1"):
            list(read_chunks(path))

    def test_overwrite_replaces_and_append_extends(self, tmp_path: Path) -> None:
        """Both modes are used: overwrite for a run, append for a resume."""
        path = tmp_path / "chunks.jsonl"
        with JsonlWriter(path, overwrite=True) as writer:
            writer.write(make_chunk("a:p0:c00"))
        with JsonlWriter(path, overwrite=False) as writer:
            writer.write(make_chunk("a:p1:c00"))
        assert len(list(read_chunks(path))) == 2
        with JsonlWriter(path, overwrite=True) as writer:
            writer.write(make_chunk("a:p2:c00"))
        assert len(list(read_chunks(path))) == 1
