"""Tests for the deduplicated document view.

``unique_documents()`` is the work-list every stage after acquisition consumes.
Getting it wrong in either direction is expensive: iterate articles instead and
a shared manual is parsed once per article; drop the article mapping and a chunk
from a shared manual can only cite one of the articles that link it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from seeley_rag.acquire.base import Article, Attachment
from seeley_rag.acquire.manifest import ManifestWriter, unique_documents

SHARED = b"%PDF-1.4 April 2005 Braemar Heaters Service Guide"
OTHER = b"%PDF-1.4 a different manual"


def stored(
    tmp: Path, attachment_id: str, content: bytes, filename: str = "manual.pdf"
) -> Attachment:
    """Write a PDF into the raw store and return its attachment record.

    Args:
        tmp: Data root.
        attachment_id: Freshdesk attachment ID.
        content: File bytes.
        filename: Filename the portal served it under.

    Returns:
        A downloaded :class:`Attachment`.
    """
    digest = hashlib.sha256(content).hexdigest()
    target = tmp / "00_raw" / "pdf" / f"{digest}.pdf"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return Attachment(
        attachment_id=attachment_id,
        filename=filename,
        url=f"https://x/helpdesk/attachments/{attachment_id}",
        sha256=digest,
        size_bytes=len(content),
        stored_path=str(target),
    )


def article(
    article_id: str, attachments: list[Attachment], folder: str = "Service Guides"
) -> Article:
    """Build a manifest article.

    Args:
        article_id: Freshdesk article ID.
        attachments: Attachments to record.
        folder: Folder name.

    Returns:
        An :class:`Article`.
    """
    return Article(
        article_id=article_id,
        title=f"Article {article_id}",
        url=f"https://x/support/solutions/articles/{article_id}-slug",
        category="DUCTED GAS HEATING (DGH) SERVICE AND INSTALLATION",
        folder=folder,
        folder_id="47000783696",
        body_text="Pdf attached",
        attachments=attachments,
    )


class TestUniqueDocuments:
    """Collapsing attachment references to documents."""

    def test_same_bytes_under_different_ids_collapse(self, temp_data_root: Path) -> None:
        """Five articles linking one manual is one document, not five.

        This is the real case: the April 2005 Braemar service guide is attached
        to five separate fault-diagnosis articles under five attachment IDs.
        """
        with ManifestWriter() as writer:
            for n in range(1, 6):
                writer.write(article(str(n), [stored(temp_data_root, f"a{n}", SHARED)]))

        documents = unique_documents()
        assert len(documents) == 1
        assert documents[0].reference_count == 5
        assert documents[0].is_shared is True

    def test_article_mapping_is_preserved(self, temp_data_root: Path) -> None:
        """Without this a shared manual can only cite one of its articles.

        Citation has to resolve to whichever article the installer arrived from,
        so collapsing the bytes must not collapse the provenance.
        """
        with ManifestWriter() as writer:
            writer.write(article("1", [stored(temp_data_root, "a1", SHARED)]))
            writer.write(article("2", [stored(temp_data_root, "a2", SHARED)]))

        document = unique_documents()[0]
        assert document.article_ids == ["1", "2"]
        assert document.attachment_ids == ["a1", "a2"]
        assert document.titles == ["Article 1", "Article 2"]

    def test_distinct_documents_stay_distinct(self, temp_data_root: Path) -> None:
        """Deduplication must not merge genuinely different manuals."""
        with ManifestWriter() as writer:
            writer.write(article("1", [stored(temp_data_root, "a1", SHARED)]))
            writer.write(article("2", [stored(temp_data_root, "a2", OTHER)]))
        assert len(unique_documents()) == 2

    def test_wasted_bytes_avoided(self, temp_data_root: Path) -> None:
        """The saving is (references - 1) copies of the document."""
        with ManifestWriter() as writer:
            for n in range(1, 4):
                writer.write(article(str(n), [stored(temp_data_root, f"a{n}", SHARED)]))
        document = unique_documents()[0]
        assert document.wasted_bytes_avoided == 2 * len(SHARED)

    def test_alternate_filenames_are_collected(self, temp_data_root: Path) -> None:
        """The same bytes are occasionally served under different names."""
        with ManifestWriter() as writer:
            writer.write(article("1", [stored(temp_data_root, "a1", SHARED, "guide-2005.pdf")]))
            writer.write(article("2", [stored(temp_data_root, "a2", SHARED, "braemar-2005.pdf")]))
        document = unique_documents()[0]
        assert set(document.filenames) == {"guide-2005.pdf", "braemar-2005.pdf"}
        assert document.primary_filename == "guide-2005.pdf"

    def test_folders_and_categories_are_collected(self, temp_data_root: Path) -> None:
        """A manual shared across folders carries all of them for routing."""
        with ManifestWriter() as writer:
            writer.write(
                article("1", [stored(temp_data_root, "a1", SHARED)], folder="Service Guides")
            )
            writer.write(article("2", [stored(temp_data_root, "a2", SHARED)], folder="Diagnostics"))
        document = unique_documents()[0]
        assert document.folders == ["Service Guides", "Diagnostics"]
        assert len(document.categories) == 1

    def test_undownloaded_attachments_are_omitted(self, temp_data_root: Path) -> None:
        """There are no bytes to parse, so there is no document."""
        never = Attachment(attachment_id="a9", filename="missing.pdf", url="https://x/9")
        with ManifestWriter() as writer:
            writer.write(article("1", [never]))
        assert unique_documents() == []

    def test_articles_without_attachments_contribute_nothing(self, temp_data_root: Path) -> None:
        """Diagnostic articles are their own content stream, not documents."""
        with ManifestWriter() as writer:
            writer.write(article("1", []))
        assert unique_documents() == []

    def test_one_article_linking_two_documents(self, temp_data_root: Path) -> None:
        """An article may attach several manuals; each is its own document."""
        with ManifestWriter() as writer:
            writer.write(
                article(
                    "1",
                    [
                        stored(temp_data_root, "a1", SHARED),
                        stored(temp_data_root, "a2", OTHER),
                    ],
                )
            )
        documents = unique_documents()
        assert len(documents) == 2
        assert all(d.article_ids == ["1"] for d in documents)

    def test_empty_manifest_yields_no_documents(self, temp_data_root: Path) -> None:
        """A dry or aborted run still answers the question."""
        with ManifestWriter():
            pass
        assert unique_documents() == []
