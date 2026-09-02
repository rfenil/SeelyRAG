"""Tests for crash-resume.

A full-corpus crawl is a ~35-minute unattended run against someone else's
server. If it dies at minute 30, restarting from zero is both wasteful and
impolite, so resuming is the default rather than an option. These tests pin down
the three properties that make it trustworthy: nothing is re-fetched, nothing is
duplicated, and a manifest truncated by a kill does not defeat it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pytest_httpx import HTTPXMock

from seeley_rag.acquire.attachments import AttachmentDownloader
from seeley_rag.acquire.base import Article, Attachment
from seeley_rag.acquire.manifest import (
    ManifestWriter,
    compact,
    load_manifest,
    load_progress,
    validate,
)
from seeley_rag.acquire.portal import PortalScraper
from seeley_rag.settings import Settings

BASE = "https://seeleyinternationalhelp.freshdesk.com"
PDF_BYTES = b"%PDF-1.4 a service manual"


def article(article_id: str, attachments: list[Attachment] | None = None) -> Article:
    """Build a manifest article.

    Args:
        article_id: Freshdesk article ID.
        attachments: Attachments to record.

    Returns:
        An :class:`Article`.
    """
    return Article(
        article_id=article_id,
        title=f"Article {article_id}",
        url=f"https://x/support/solutions/articles/{article_id}-slug",
        category="DUCTED GAS HEATING (DGH) SERVICE AND INSTALLATION",
        folder="Service Guides",
        folder_id="47000783696",
        body_text="Pdf attached",
        attachments=attachments or [],
    )


def downloaded_attachment(tmp: Path, attachment_id: str, content: bytes = PDF_BYTES) -> Attachment:
    """Write a PDF to the raw store and return the matching attachment record.

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
        url=f"{BASE}/helpdesk/attachments/{attachment_id}",
        sha256=digest,
        size_bytes=len(content),
        stored_path=str(target),
    )


class TestLoadProgress:
    """Reading what an earlier run finished."""

    def test_absent_manifest_is_not_an_error(self, temp_data_root: Path) -> None:
        """The first run has nothing to resume, which is normal, not a failure."""
        progress = load_progress()
        assert progress.is_empty
        assert progress.article_ids == set()

    def test_records_completed_articles_and_attachments(self, temp_data_root: Path) -> None:
        """Both halves are needed: articles to skip, attachments to not re-fetch."""
        attachment = downloaded_attachment(temp_data_root, "a1")
        with ManifestWriter() as writer:
            writer.write(article("1", [attachment]))
            writer.write(article("2"))

        progress = load_progress()
        assert progress.article_ids == {"1", "2"}
        assert "a1" in progress.attachments
        assert progress.attachments["a1"].sha256 == attachment.sha256

    def test_attachment_whose_file_vanished_is_not_considered_done(
        self, temp_data_root: Path
    ) -> None:
        """Deleting a PDF is enough to make the next run fetch it again.

        The manifest alone is not proof the bytes are still there.
        """
        attachment = downloaded_attachment(temp_data_root, "a1")
        with ManifestWriter() as writer:
            writer.write(article("1", [attachment]))
        Path(attachment.stored_path).unlink()

        progress = load_progress()
        assert progress.article_ids == {"1"}
        assert progress.attachments == {}

    def test_truncated_final_row_does_not_defeat_resume(self, temp_data_root: Path) -> None:
        """A process killed mid-write leaves half a line.

        Refusing to resume because of it would force exactly the full re-crawl
        that resume exists to avoid.
        """
        from seeley_rag.paths import MANIFEST_PATH

        with ManifestWriter() as writer:
            writer.write(article("1"))
            writer.write(article("2"))
        with MANIFEST_PATH.open("a", encoding="utf-8") as handle:
            handle.write('{"article_id": "3", "title": "half-writ')

        progress = load_progress()
        assert progress.article_ids == {"1", "2"}
        assert progress.malformed_lines == 1
        assert progress.needs_compaction is True

    def test_rows_are_flushed_immediately(self, temp_data_root: Path) -> None:
        """Buffered rows would be lost on a kill and re-fetched."""
        from seeley_rag.paths import MANIFEST_PATH

        with ManifestWriter() as writer:
            writer.write(article("1"))
            # Read from a separate handle while the writer is still open.
            assert MANIFEST_PATH.read_text(encoding="utf-8").strip()


class TestCompact:
    """Repairing a manifest after a crash."""

    def test_drops_malformed_rows(self, temp_data_root: Path) -> None:
        """The corrupt tail is removed so the strict reader works again."""
        from seeley_rag.paths import MANIFEST_PATH

        with ManifestWriter() as writer:
            writer.write(article("1"))
        with MANIFEST_PATH.open("a", encoding="utf-8") as handle:
            handle.write('{"broken')

        assert compact() == 1
        assert load_progress().malformed_lines == 0
        assert [a.article_id for a in load_manifest()] == ["1"]

    def test_drops_duplicate_articles(self, temp_data_root: Path) -> None:
        """Duplicates from pre-resume runs are collapsed to the first occurrence."""
        with ManifestWriter() as writer:
            writer.write(article("1"))
            writer.write(article("2"))
            writer.write(article("1"))

        assert compact() == 1
        assert [a.article_id for a in load_manifest()] == ["1", "2"]
        assert not [p for p in validate() if "duplicate" in p]

    def test_clean_manifest_is_left_alone(self, temp_data_root: Path) -> None:
        """Nothing to repair means nothing is rewritten."""
        with ManifestWriter() as writer:
            writer.write(article("1"))
        assert compact() == 0
        assert [a.article_id for a in load_manifest()] == ["1"]


class TestResumedDownloads:
    """Attachments already held are not re-fetched."""

    def test_known_attachment_is_skipped_without_a_request(
        self, settings: Settings, temp_data_root: Path, httpx_mock: HTTPXMock
    ) -> None:
        """This is the expensive half of resume.

        Content addressing alone cannot help: the hash is only known after the
        bytes have been streamed, so without the manifest map a resumed run
        re-downloads every manual just to discover it already had them.
        """
        known = downloaded_attachment(temp_data_root, "a1")
        downloader = AttachmentDownloader(settings=settings, known={"a1": known})

        result = downloader.download(
            Attachment(
                attachment_id="a1",
                filename="manual.pdf",
                url=f"{BASE}/helpdesk/attachments/a1",
            )
        )

        assert result.sha256 == known.sha256
        assert downloader.summary()["skipped_resumed"] == 1
        assert downloader.summary()["bytes_fetched"] == 0
        # No mocked response was registered, so any request would have errored.
        assert httpx_mock.get_requests() == []

    def test_unknown_attachment_is_still_downloaded(
        self, settings: Settings, temp_data_root: Path, httpx_mock: HTTPXMock
    ) -> None:
        """Resume must not make the crawler skip work it has not done."""
        downloader = AttachmentDownloader(settings=settings, known={})
        httpx_mock.add_response(url=f"{BASE}/helpdesk/attachments/a2", content=PDF_BYTES)
        result = downloader.download(
            Attachment(attachment_id="a2", filename="m.pdf", url=f"{BASE}/helpdesk/attachments/a2")
        )
        assert result.is_downloaded
        assert downloader.summary()["downloaded"] == 1

    def test_repeat_within_a_run_is_skipped_too(
        self, settings: Settings, temp_data_root: Path, httpx_mock: HTTPXMock
    ) -> None:
        """The same attachment ID on two articles costs one download."""
        downloader = AttachmentDownloader(settings=settings)
        httpx_mock.add_response(url=f"{BASE}/helpdesk/attachments/a3", content=PDF_BYTES)
        payload = Attachment(
            attachment_id="a3", filename="m.pdf", url=f"{BASE}/helpdesk/attachments/a3"
        )
        downloader.download(payload)
        downloader.download(payload)
        assert len(httpx_mock.get_requests()) == 1
        assert downloader.summary()["skipped_resumed"] == 1


class TestResumedWalk:
    """Already-acquired articles are skipped without fetching their pages."""

    @pytest.fixture
    def scraper(self, settings: Settings) -> PortalScraper:
        """A scraper with caching and throttling disabled."""
        instance = PortalScraper(settings=settings, use_cache=False)
        instance.limiter.delay = 0.0
        return instance

    def test_skipped_articles_are_never_fetched(
        self, scraper: PortalScraper, httpx_mock: HTTPXMock, article_stub_html: str
    ) -> None:
        """Skipping must happen before the page request, or resume saves nothing."""
        solutions = (
            '<html><body><div class="cs-s">'
            '<h3 class="heading accordion-heading">'
            '<a href="/support/solutions/1">DUCTED GAS HEATING (DGH)</a></h3>'
            '<section class="cs-g article-list"><div class="list-lead">'
            '<a href="/support/solutions/folders/10" title="Service Guides">SG</a>'
            "</div></section></div></body></html>"
        )
        listing = (
            '<html><body><section class="article-list c-list">'
            + "".join(
                '<div class="c-row c-article-row"><div class="ellipsis article-title">'
                f'<a href="/support/solutions/articles/{i}-slug" class="c-link">A{i}</a>'
                "</div></div>"
                for i in ("1", "2", "3")
            )
            + "</section></body></html>"
        )
        httpx_mock.add_response(url=f"{BASE}/support/solutions", text=solutions)
        httpx_mock.add_response(url=f"{BASE}/support/solutions/folders/10", text=listing)
        httpx_mock.add_response(
            url=f"{BASE}/support/solutions/folders/10/page/2",
            text='<html><body><section class="article-list c-list"></section></body></html>',
        )
        # Only article 3 may be fetched; registering just its page proves it.
        httpx_mock.add_response(
            url=f"{BASE}/support/solutions/articles/3-slug", text=article_stub_html
        )

        articles = list(scraper.iter_articles(None, skip_article_ids={"1", "2"}))

        assert len(articles) == 1
        assert scraper.articles_skipped == 2
        assert not any("1-slug" in str(r.url) for r in httpx_mock.get_requests())

    def test_limit_counts_only_new_articles(
        self, scraper: PortalScraper, httpx_mock: HTTPXMock, article_stub_html: str
    ) -> None:
        """``--limit 1`` on a resumed run means one more, not one in total."""
        solutions = (
            '<html><body><div class="cs-s">'
            '<h3 class="heading accordion-heading">'
            '<a href="/support/solutions/1">DGH</a></h3>'
            '<section class="cs-g article-list"><div class="list-lead">'
            '<a href="/support/solutions/folders/10" title="SG">SG</a>'
            "</div></section></div></body></html>"
        )
        listing = (
            '<html><body><section class="article-list c-list">'
            + "".join(
                '<div class="c-row c-article-row"><div class="ellipsis article-title">'
                f'<a href="/support/solutions/articles/{i}-slug" class="c-link">A{i}</a>'
                "</div></div>"
                for i in ("1", "2")
            )
            + "</section></body></html>"
        )
        httpx_mock.add_response(url=f"{BASE}/support/solutions", text=solutions)
        httpx_mock.add_response(url=f"{BASE}/support/solutions/folders/10", text=listing)
        # The listing walk always reads one page past the last to find the end.
        httpx_mock.add_response(
            url=f"{BASE}/support/solutions/folders/10/page/2",
            text='<html><body><section class="article-list c-list"></section></body></html>',
        )
        httpx_mock.add_response(
            url=f"{BASE}/support/solutions/articles/2-slug", text=article_stub_html
        )

        articles = list(scraper.iter_articles(None, limit=1, skip_article_ids={"1"}))
        assert [a.article_id for a in articles] == ["2"]
