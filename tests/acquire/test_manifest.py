"""Tests for manifest writing, reading, validation and summarising."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from seeley_rag.acquire.base import Article, Attachment
from seeley_rag.acquire.manifest import (
    ManifestWriter,
    load_manifest,
    read_manifest,
    summarise,
    validate,
)
from seeley_rag.exceptions import ManifestError


def article(article_id: str, **overrides: object) -> Article:
    """Build an article for the manifest.

    Args:
        article_id: Freshdesk article ID.
        **overrides: Field overrides.

    Returns:
        An :class:`Article`.
    """
    payload: dict[str, object] = {
        "article_id": article_id,
        "title": f"Article {article_id}",
        "url": f"https://x/support/solutions/articles/{article_id}-slug",
        "category": "DUCTED GAS HEATING (DGH) SERVICE AND INSTALLATION",
        "folder": "Service Guides",
        "folder_id": "47000783696",
        "body_text": "Pdf attached",
    }
    payload.update(overrides)
    return Article(**payload)  # type: ignore[arg-type]


def stored_attachment(tmp: Path, attachment_id: str, content: bytes = b"pdf-bytes") -> Attachment:
    """Write a fake PDF and return an attachment pointing at it.

    Args:
        tmp: Data root.
        attachment_id: Freshdesk attachment ID.
        content: File bytes.

    Returns:
        A downloaded :class:`Attachment`.
    """
    import hashlib

    digest = hashlib.sha256(content).hexdigest()
    target = tmp / "00_raw" / "pdf" / f"{digest}.pdf"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return Attachment(
        attachment_id=attachment_id,
        filename="manual.pdf",
        url=f"https://x/helpdesk/attachments/{attachment_id}",
        sha256=digest,
        size_bytes=len(content),
        stored_path=str(target),
    )


class TestWriteAndRead:
    """The JSONL round trip."""

    def test_round_trip(self, temp_data_root: Path) -> None:
        """What the writer wrote, the reader reads back identically."""
        with ManifestWriter() as writer:
            writer.write(article("1"))
            writer.write(article("2"))

        articles = load_manifest()
        assert [a.article_id for a in articles] == ["1", "2"]
        assert articles[0].title == "Article 1"

    def test_appends_by_default(self, temp_data_root: Path) -> None:
        """An interrupted crawl resumes without discarding what it fetched."""
        with ManifestWriter() as writer:
            writer.write(article("1"))
        with ManifestWriter() as writer:
            writer.write(article("2"))
        assert len(load_manifest()) == 2

    def test_overwrite_truncates(self, temp_data_root: Path) -> None:
        """A clean run starts from an empty manifest."""
        with ManifestWriter() as writer:
            writer.write(article("1"))
        with ManifestWriter(overwrite=True) as writer:
            writer.write(article("2"))
        assert [a.article_id for a in load_manifest()] == ["2"]

    def test_one_json_object_per_line(self, temp_data_root: Path) -> None:
        """JSONL, so a crash leaves a readable file rather than a truncated array."""
        from seeley_rag.paths import MANIFEST_PATH

        with ManifestWriter() as writer:
            writer.write_all([article("1"), article("2")])
        lines = MANIFEST_PATH.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        assert all(json.loads(line)["article_id"] for line in lines)

    def test_non_ascii_survives(self, temp_data_root: Path) -> None:
        """Titles carry en-dashes and warning glyphs; encoding is pinned to UTF-8."""
        with ManifestWriter() as writer:
            writer.write(article("1", title="Braemar TQ – FC7 ⚠️"))
        assert load_manifest()[0].title == "Braemar TQ – FC7 ⚠️"

    def test_write_outside_context_manager_raises(self, temp_data_root: Path) -> None:
        """Misuse fails loudly rather than dropping rows on the floor."""
        writer = ManifestWriter()
        with pytest.raises(ManifestError):
            writer.write(article("1"))

    def test_missing_manifest_raises_actionable_error(self, temp_data_root: Path) -> None:
        """The error names the command that produces the file."""
        with pytest.raises(ManifestError, match="02_acquire"):
            list(read_manifest())

    def test_malformed_line_raises(self, temp_data_root: Path) -> None:
        """A corrupt manifest must not be silently half-read."""
        from seeley_rag.paths import MANIFEST_PATH

        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST_PATH.write_text('{"not": "an article"}\n', encoding="utf-8")
        with pytest.raises(ManifestError):
            load_manifest()

    def test_blank_lines_are_skipped(self, temp_data_root: Path) -> None:
        """A trailing newline is not a parse error."""
        from seeley_rag.paths import MANIFEST_PATH

        with ManifestWriter() as writer:
            writer.write(article("1"))
        with MANIFEST_PATH.open("a", encoding="utf-8") as handle:
            handle.write("\n\n")
        assert len(load_manifest()) == 1


class TestValidate:
    """Manifest validation."""

    def test_clean_manifest_has_no_problems(self, temp_data_root: Path) -> None:
        """A fully downloaded manifest validates clean."""
        with ManifestWriter() as writer:
            writer.write(article("1", attachments=[stored_attachment(temp_data_root, "a1")]))
        assert validate() == []

    def test_missing_pdf_is_reported(self, temp_data_root: Path) -> None:
        """A row pointing at a missing PDF is how a stage indexes 400 of 600 manuals.

        Catching it here is the whole point of validation.
        """
        attachment = stored_attachment(temp_data_root, "a1")
        Path(attachment.stored_path).unlink()
        with ManifestWriter() as writer:
            writer.write(article("1", attachments=[attachment]))

        problems = validate()
        assert len(problems) == 1
        assert "does not exist" in problems[0]

    def test_hash_path_mismatch_is_reported(self, temp_data_root: Path) -> None:
        """Content addressing is broken if the filename is not the hash."""
        attachment = stored_attachment(temp_data_root, "a1")
        wrong = Path(attachment.stored_path).with_name("not-a-hash.pdf")
        Path(attachment.stored_path).rename(wrong)
        attachment = attachment.model_copy(update={"stored_path": str(wrong)})
        with ManifestWriter() as writer:
            writer.write(article("1", attachments=[attachment]))

        problems = validate()
        assert any("content addressing is broken" in p for p in problems)

    def test_undownloaded_attachment_is_reported(self, temp_data_root: Path) -> None:
        """An attachment we never retrieved is a gap, and must be visible."""
        never = Attachment(attachment_id="a1", filename="m.pdf", url="https://x/1")
        with ManifestWriter() as writer:
            writer.write(article("1", attachments=[never]))
        assert any("never downloaded" in p for p in validate())

    def test_duplicate_article_id_is_reported(self, temp_data_root: Path) -> None:
        """Re-running without --overwrite can double up rows."""
        with ManifestWriter() as writer:
            writer.write(article("1"))
            writer.write(article("1"))
        assert any("duplicate article_id" in p for p in validate())

    def test_empty_category_is_reported(self, temp_data_root: Path) -> None:
        """Category drives product routing; an empty one is a latent wrong answer."""
        with ManifestWriter() as writer:
            writer.write(article("1", category=""))
        assert any("empty category" in p for p in validate())


class TestSummarise:
    """The run summary."""

    def test_counts_articles_and_streams(self, temp_data_root: Path) -> None:
        """The stub versus content split is the headline number."""
        with ManifestWriter() as writer:
            writer.write(article("1", attachments=[stored_attachment(temp_data_root, "a1")]))
            writer.write(article("2", body_text="x" * 500))

        summary = summarise()
        assert summary["articles"] == 2
        assert summary["stub_articles"] == 1
        assert summary["content_articles"] == 1

    def test_deduplication_is_counted_and_bytes_not_double_counted(
        self, temp_data_root: Path
    ) -> None:
        """Two references to one document is one document and one document's bytes."""
        shared = b"the same manual"
        first = stored_attachment(temp_data_root, "a1", shared)
        second = stored_attachment(temp_data_root, "a2", shared)

        with ManifestWriter() as writer:
            writer.write(article("1", attachments=[first]))
            writer.write(article("2", attachments=[second]))

        summary = summarise()
        assert summary["attachments"] == 2
        assert summary["unique_documents"] == 1
        assert summary["duplicate_attachments"] == 1
        assert summary["bytes_stored"] == len(shared)
        assert summary["duplication_rate"] == 0.5

    def test_render_is_printable(self, temp_data_root: Path) -> None:
        """The summary is printed at the end of every crawl."""
        with ManifestWriter() as writer:
            writer.write(article("1", attachments=[stored_attachment(temp_data_root, "a1")]))
        rendered = summarise().render()
        assert "Manifest summary" in rendered
        assert "duplication rate" in rendered

    def test_empty_manifest_summarises_without_dividing_by_zero(self, temp_data_root: Path) -> None:
        """A dry or aborted run still summarises."""
        with ManifestWriter():
            pass
        summary = summarise()
        assert summary["articles"] == 0
        assert summary["duplication_rate"] == 0.0
