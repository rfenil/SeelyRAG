"""Tests for attachment content-type detection.

The portal answers some restricted attachments with **HTTP 200 and a login
page**. Nothing about the status code, the headers or the manifest schema
catches that, so it has to be caught by looking at the bytes. Observed live:
attachment 47106044429 returned 20 KB of Freshdesk sign-in HTML, which was
hashed and recorded as a service manual.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pytest_httpx import HTTPXMock

from seeley_rag.acquire.attachments import AttachmentDownloader, detect_suffix, looks_like_html
from seeley_rag.acquire.base import Attachment
from seeley_rag.exceptions import AcquisitionError
from seeley_rag.settings import Settings

BASE = "https://seeleyinternationalhelp.freshdesk.com"

LOGIN_PAGE = (
    b"<!DOCTYPE html>\n<html><head><title>Sign into : Seeley International Pty Ltd"
    b"</title></head><body><h2>Login to the support portal</h2></body></html>"
)


def attachment(attachment_id: str = "1") -> Attachment:
    """Build an attachment pointing at the portal."""
    return Attachment(
        attachment_id=attachment_id,
        filename="",
        url=f"{BASE}/helpdesk/attachments/{attachment_id}",
    )


class TestDetectSuffix:
    """Magic-number sniffing."""

    @pytest.mark.parametrize(
        ("head", "expected"),
        [
            (b"%PDF-1.4 ...", ".pdf"),
            (b"\x89PNG\r\n\x1a\n...", ".png"),
            (b"\xff\xd8\xff\xe0...", ".jpg"),
            (b"GIF89a...", ".gif"),
            (b"PK\x03\x04...", ".zip"),
            (b"\xd0\xcf\x11\xe0...", ".doc"),
            (b"II*\x00...", ".tif"),
        ],
    )
    def test_recognised_types(self, head: bytes, expected: str) -> None:
        """Each observed corpus type maps to its real extension."""
        assert detect_suffix(head) == expected

    def test_unrecognised_returns_none(self) -> None:
        """An unknown type is reported, not guessed at."""
        assert detect_suffix(b"\x00\x01\x02\x03") is None


class TestLooksLikeHtml:
    """Detecting a page where a file was expected."""

    @pytest.mark.parametrize(
        "body",
        [
            LOGIN_PAGE,
            b"<html><body>error</body></html>",
            b"\n\n  <!doctype html>",
            b"<?xml version='1.0'?><error/>",
        ],
    )
    def test_html_bodies_are_detected(self, body: bytes) -> None:
        """Leading whitespace and case variations must not hide it."""
        assert looks_like_html(body) is True

    def test_pdf_is_not_html(self) -> None:
        """A real PDF must never be mistaken for an error page."""
        assert looks_like_html(b"%PDF-1.4 real manual") is False

    def test_html_mentioned_later_in_a_pdf_is_not_html(self) -> None:
        """PDFs can embed HTML strings; only the leading bytes decide."""
        assert looks_like_html(b"%PDF-1.4 ... <html> inside an annotation") is False


class TestDownloadTypeHandling:
    """How the downloader reacts to what it actually received."""

    def test_login_page_is_a_failed_download_not_a_document(
        self, settings: Settings, temp_data_root: Path, httpx_mock: HTTPXMock
    ) -> None:
        """The bug this exists to prevent.

        Storing it would put "Login to the support portal" into the retrieval
        index as though it were a service manual, with a valid manifest row and
        a passing validation.
        """
        downloader = AttachmentDownloader(settings=settings)
        httpx_mock.add_response(url=f"{BASE}/helpdesk/attachments/1", content=LOGIN_PAGE)

        with pytest.raises(AcquisitionError, match="HTML page"):
            downloader.download(attachment("1"))

        assert list((temp_data_root / "00_raw" / "pdf").iterdir()) == []

    def test_login_page_is_recorded_as_a_failure_in_a_batch(
        self, settings: Settings, temp_data_root: Path, httpx_mock: HTTPXMock
    ) -> None:
        """It must surface in the run summary rather than vanish."""
        downloader = AttachmentDownloader(settings=settings)
        httpx_mock.add_response(url=f"{BASE}/helpdesk/attachments/1", content=LOGIN_PAGE)

        results = downloader.download_all([attachment("1")])

        assert results[0].is_downloaded is False
        assert "1" in downloader.failed
        assert "HTML" in downloader.failed["1"]

    def test_error_message_explains_the_cause(
        self, settings: Settings, temp_data_root: Path, httpx_mock: HTTPXMock
    ) -> None:
        """A human reading the log should know it is an auth problem."""
        downloader = AttachmentDownloader(settings=settings)
        httpx_mock.add_response(url=f"{BASE}/helpdesk/attachments/1", content=LOGIN_PAGE)
        downloader.download_all([attachment("1")])
        message = downloader.failed["1"]
        assert "signed-in" in message or "login" in message
        assert "NOT stored" in message

    def test_pdf_keeps_the_pdf_suffix(
        self, settings: Settings, temp_data_root: Path, httpx_mock: HTTPXMock
    ) -> None:
        """The overwhelming majority case is unchanged."""
        downloader = AttachmentDownloader(settings=settings)
        httpx_mock.add_response(
            url=f"{BASE}/helpdesk/attachments/1", content=b"%PDF-1.4 a real manual"
        )
        result = downloader.download(attachment("1"))
        assert result.stored_path.endswith(".pdf")

    def test_image_is_stored_with_its_real_extension(
        self, settings: Settings, temp_data_root: Path, httpx_mock: HTTPXMock
    ) -> None:
        """A JPEG named .pdf makes every later stage try to parse it as a PDF."""
        downloader = AttachmentDownloader(settings=settings)
        httpx_mock.add_response(
            url=f"{BASE}/helpdesk/attachments/1", content=b"\xff\xd8\xff\xe0 jpeg bytes"
        )
        result = downloader.download(attachment("1"))
        assert result.stored_path.endswith(".jpg")

    def test_office_document_is_stored_as_zip(
        self, settings: Settings, temp_data_root: Path, httpx_mock: HTTPXMock
    ) -> None:
        """docx/xlsx are zip containers; the corpus contains a couple."""
        downloader = AttachmentDownloader(settings=settings)
        httpx_mock.add_response(
            url=f"{BASE}/helpdesk/attachments/1", content=b"PK\x03\x04 docx bytes"
        )
        result = downloader.download(attachment("1"))
        assert result.stored_path.endswith(".zip")

    def test_unknown_type_is_stored_as_bin_with_a_warning(
        self, settings: Settings, temp_data_root: Path, httpx_mock: HTTPXMock
    ) -> None:
        """Unrecognised is not the same as unwanted; keep the bytes, flag them."""
        downloader = AttachmentDownloader(settings=settings)
        httpx_mock.add_response(
            url=f"{BASE}/helpdesk/attachments/1", content=b"\x00\x01\x02\x03 mystery"
        )
        result = downloader.download(attachment("1"))
        assert result.stored_path.endswith(".bin")
