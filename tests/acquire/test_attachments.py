"""Tests for the attachment downloader.

Deduplication is a correctness requirement here, not an optimisation: the same
manual is attached across multiple folders, and parsing a 200-page PDF four
times wastes machine hours and crowds retrieval with duplicate results.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from pytest_httpx import HTTPXMock

from seeley_rag.acquire.attachments import AttachmentDownloader, sha256_file
from seeley_rag.acquire.base import Attachment
from seeley_rag.exceptions import AcquisitionError, RateLimitedError
from seeley_rag.settings import Settings

BASE = "https://seeleyinternationalhelp.freshdesk.com"
PDF_BYTES = b"%PDF-1.4 fake manual bytes for testing"
OTHER_BYTES = b"%PDF-1.4 a completely different manual"


def attachment(attachment_id: str, filename: str = "manual.pdf") -> Attachment:
    """Build an attachment pointing at the portal.

    Args:
        attachment_id: Freshdesk attachment ID.
        filename: Display filename.

    Returns:
        An :class:`Attachment` with no bytes yet.
    """
    return Attachment(
        attachment_id=attachment_id,
        filename=filename,
        url=f"{BASE}/helpdesk/attachments/{attachment_id}",
    )


@pytest.fixture
def downloader(settings: Settings, temp_data_root: Path) -> AttachmentDownloader:
    """A downloader writing into a temporary data tree."""
    return AttachmentDownloader(settings=settings)


class TestDownload:
    """Single-file download behaviour."""

    def test_stores_content_addressed(
        self, downloader: AttachmentDownloader, httpx_mock: HTTPXMock, temp_data_root: Path
    ) -> None:
        """The filename is the hash, so the path itself proves the contents."""
        httpx_mock.add_response(url=f"{BASE}/helpdesk/attachments/1", content=PDF_BYTES)
        result = downloader.download(attachment("1"))

        expected = hashlib.sha256(PDF_BYTES).hexdigest()
        assert result.sha256 == expected
        assert result.size_bytes == len(PDF_BYTES)
        # The path is repo-relative in a real run; here the temporary data tree
        # lies outside the repo, so relative_to_root falls back to an absolute
        # path. Either way the tail is the content-addressed name.
        assert result.stored_path.endswith(f"data/00_raw/pdf/{expected}.pdf")
        assert (temp_data_root / "00_raw" / "pdf" / f"{expected}.pdf").read_bytes() == PDF_BYTES

    def test_input_attachment_is_not_mutated(
        self, downloader: AttachmentDownloader, httpx_mock: HTTPXMock
    ) -> None:
        """A new model is returned so parsed metadata stays inspectable."""
        httpx_mock.add_response(url=f"{BASE}/helpdesk/attachments/1", content=PDF_BYTES)
        original = attachment("1")
        result = downloader.download(original)
        assert original.sha256 is None
        assert result.sha256 is not None

    def test_follows_redirects_to_s3(
        self, downloader: AttachmentDownloader, httpx_mock: HTTPXMock
    ) -> None:
        """Attachment URLs 302 to S3.

        Without redirect following, every download returns an empty 302 body and
        the manifest quietly fills with zero-byte PDFs.
        """
        httpx_mock.add_response(
            url=f"{BASE}/helpdesk/attachments/1",
            status_code=302,
            headers={"Location": "https://s3.amazonaws.com/bucket/manual.pdf"},
        )
        httpx_mock.add_response(url="https://s3.amazonaws.com/bucket/manual.pdf", content=PDF_BYTES)
        result = downloader.download(attachment("1"))
        assert result.size_bytes == len(PDF_BYTES)

    @pytest.mark.parametrize("status", [403, 429])
    def test_blocked_status_raises_immediately(
        self, downloader: AttachmentDownloader, httpx_mock: HTTPXMock, status: int
    ) -> None:
        """Being blocked stops the whole run; it is never retried."""
        httpx_mock.add_response(url=f"{BASE}/helpdesk/attachments/1", status_code=status)
        with pytest.raises(RateLimitedError):
            downloader.download(attachment("1"))

    def test_missing_attachment_raises(
        self, downloader: AttachmentDownloader, httpx_mock: HTTPXMock
    ) -> None:
        """A 404 is reported rather than silently stored as an empty file."""
        httpx_mock.add_response(url=f"{BASE}/helpdesk/attachments/1", status_code=404)
        with pytest.raises(AcquisitionError):
            downloader.download(attachment("1"))

    def test_failed_download_leaves_no_partial_file(
        self, downloader: AttachmentDownloader, httpx_mock: HTTPXMock, temp_data_root: Path
    ) -> None:
        """A partial download must never occupy a content-addressed path.

        If it did, a later run would find the path present and trust corrupt
        bytes forever, because the path is supposed to prove the contents.
        """
        httpx_mock.add_response(url=f"{BASE}/helpdesk/attachments/1", status_code=404)
        with pytest.raises(AcquisitionError):
            downloader.download(attachment("1"))
        assert list((temp_data_root / "00_raw" / "pdf").iterdir()) == []


class TestDeduplication:
    """Content-hash deduplication."""

    def test_same_bytes_under_two_ids_stored_once(
        self, downloader: AttachmentDownloader, httpx_mock: HTTPXMock, temp_data_root: Path
    ) -> None:
        """The same manual attached to two articles is downloaded once, stored once."""
        httpx_mock.add_response(url=f"{BASE}/helpdesk/attachments/1", content=PDF_BYTES)
        httpx_mock.add_response(url=f"{BASE}/helpdesk/attachments/2", content=PDF_BYTES)

        first = downloader.download(attachment("1"))
        second = downloader.download(attachment("2"))

        assert first.sha256 == second.sha256
        assert first.is_duplicate_of is None
        assert second.is_duplicate_of == "1"
        assert len(list((temp_data_root / "00_raw" / "pdf").iterdir())) == 1

    def test_different_bytes_stored_separately(
        self, downloader: AttachmentDownloader, httpx_mock: HTTPXMock, temp_data_root: Path
    ) -> None:
        """Distinct manuals get distinct paths."""
        httpx_mock.add_response(url=f"{BASE}/helpdesk/attachments/1", content=PDF_BYTES)
        httpx_mock.add_response(url=f"{BASE}/helpdesk/attachments/2", content=OTHER_BYTES)
        downloader.download(attachment("1"))
        second = downloader.download(attachment("2"))
        assert second.is_duplicate_of is None
        assert len(list((temp_data_root / "00_raw" / "pdf").iterdir())) == 2

    def test_existing_file_is_not_rewritten(
        self, downloader: AttachmentDownloader, httpx_mock: HTTPXMock, temp_data_root: Path
    ) -> None:
        """data/00_raw is write-once, so a re-run must leave bytes untouched."""
        digest = hashlib.sha256(PDF_BYTES).hexdigest()
        destination = temp_data_root / "00_raw" / "pdf" / f"{digest}.pdf"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(PDF_BYTES)
        original_mtime = destination.stat().st_mtime_ns

        httpx_mock.add_response(url=f"{BASE}/helpdesk/attachments/1", content=PDF_BYTES)
        result = downloader.download(attachment("1"))

        assert destination.stat().st_mtime_ns == original_mtime
        assert result.stored_path.endswith(f"{digest}.pdf")
        assert downloader.summary()["deduplicated"] == 1

    def test_summary_separates_fetched_from_stored_bytes(
        self, downloader: AttachmentDownloader, httpx_mock: HTTPXMock
    ) -> None:
        """The gap between the two is what deduplication saved."""
        httpx_mock.add_response(url=f"{BASE}/helpdesk/attachments/1", content=PDF_BYTES)
        httpx_mock.add_response(url=f"{BASE}/helpdesk/attachments/2", content=PDF_BYTES)
        downloader.download(attachment("1"))
        downloader.download(attachment("2"))

        stats = downloader.summary()
        assert stats["unique_documents"] == 1
        assert stats["deduplicated"] == 1
        assert stats["bytes_fetched"] == 2 * len(PDF_BYTES)
        assert stats["bytes_stored"] == len(PDF_BYTES)
        assert stats["bytes_saved_by_dedupe"] == len(PDF_BYTES)


class TestDownloadAll:
    """Batch behaviour and failure isolation."""

    def test_one_failure_does_not_abort_the_batch(
        self, downloader: AttachmentDownloader, httpx_mock: HTTPXMock
    ) -> None:
        """One dead link must not cost the other 899 articles."""
        httpx_mock.add_response(url=f"{BASE}/helpdesk/attachments/1", content=PDF_BYTES)
        httpx_mock.add_response(url=f"{BASE}/helpdesk/attachments/2", status_code=404)
        httpx_mock.add_response(url=f"{BASE}/helpdesk/attachments/3", content=OTHER_BYTES)

        results = downloader.download_all([attachment("1"), attachment("2"), attachment("3")])

        assert len(results) == 3
        assert results[0].is_downloaded is True
        assert results[1].is_downloaded is False
        assert results[2].is_downloaded is True
        assert "2" in downloader.failed

    def test_failed_attachment_is_still_recorded(
        self, downloader: AttachmentDownloader, httpx_mock: HTTPXMock
    ) -> None:
        """The manifest must show that an attachment existed but was not retrieved."""
        httpx_mock.add_response(url=f"{BASE}/helpdesk/attachments/2", status_code=404)
        results = downloader.download_all([attachment("2", "missing-manual.pdf")])
        assert results[0].filename == "missing-manual.pdf"
        assert results[0].stored_path is None

    def test_being_blocked_aborts_the_batch(
        self, downloader: AttachmentDownloader, httpx_mock: HTTPXMock
    ) -> None:
        """RateLimitedError is a whole-run condition and must not be swallowed."""
        httpx_mock.add_response(url=f"{BASE}/helpdesk/attachments/1", status_code=429)
        with pytest.raises(RateLimitedError):
            downloader.download_all([attachment("1"), attachment("2")])


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    """The streaming hash agrees with a one-shot hash."""
    target = tmp_path / "x.bin"
    target.write_bytes(PDF_BYTES)
    assert sha256_file(target) == hashlib.sha256(PDF_BYTES).hexdigest()
