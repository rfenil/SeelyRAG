"""Tests for the shared rate limiter.

The 1 req/sec guarantee is about the run as a whole, not about any one
component. These tests pin that down, because the failure mode is invisible
locally: everything works, and Seeley's server simply sees twice the traffic we
promised.
"""

from __future__ import annotations

import pytest
from pytest_httpx import HTTPXMock

from seeley_rag.acquire.attachments import AttachmentDownloader
from seeley_rag.acquire.base import Attachment
from seeley_rag.acquire.portal import PortalScraper
from seeley_rag.acquire.throttle import RateLimiter
from seeley_rag.settings import Settings

BASE = "https://seeleyinternationalhelp.freshdesk.com"


class TestRateLimiter:
    """The limiter itself."""

    def test_first_call_does_not_wait(self) -> None:
        """Nothing has been requested yet, so there is nothing to space out."""
        limiter = RateLimiter(rps=1000.0)
        limiter.wait()
        assert limiter.calls == 1
        assert limiter.waits == 0

    def test_subsequent_call_waits(self) -> None:
        """A second request inside the window is delayed."""
        limiter = RateLimiter(rps=100.0)  # 10ms spacing
        limiter.wait()
        limiter.wait()
        assert limiter.calls == 2
        assert limiter.waits == 1

    def test_delay_is_the_inverse_of_rps(self) -> None:
        """1 rps means one second between requests."""
        assert RateLimiter(rps=1.0).delay == 1.0
        assert RateLimiter(rps=2.0).delay == 0.5

    def test_reset_clears_the_window(self) -> None:
        """After a reset the next call proceeds immediately."""
        limiter = RateLimiter(rps=100.0)
        limiter.wait()
        limiter.reset()
        limiter.wait()
        assert limiter.waits == 0

    def test_non_positive_rps_is_rejected(self) -> None:
        """A zero or negative rate is a configuration error, not an infinite wait."""
        with pytest.raises(ValueError):
            RateLimiter(rps=0)


class TestSharedLimiter:
    """Scraper and downloader must share one limiter."""

    def test_downloader_requests_go_through_the_limiter(
        self, settings: Settings, temp_data_root: object, httpx_mock: HTTPXMock
    ) -> None:
        """Attachment downloads are requests too.

        Unthrottled, a full-corpus run would fire ~600 PDF fetches outside the
        rate limit, on top of the page fetches that are inside it.
        """
        limiter = RateLimiter(rps=1000.0)
        downloader = AttachmentDownloader(settings=settings, limiter=limiter)
        httpx_mock.add_response(url=f"{BASE}/helpdesk/attachments/1", content=b"%PDF-1 a")
        httpx_mock.add_response(url=f"{BASE}/helpdesk/attachments/2", content=b"%PDF-1 b")

        for attachment_id in ("1", "2"):
            downloader.download(
                Attachment(
                    attachment_id=attachment_id,
                    filename="m.pdf",
                    url=f"{BASE}/helpdesk/attachments/{attachment_id}",
                )
            )
        downloader.close()
        assert limiter.calls == 2

    def test_page_and_pdf_requests_share_one_window(
        self, settings: Settings, temp_data_root: object, httpx_mock: HTTPXMock
    ) -> None:
        """Interleaved page and attachment fetches are spaced against each other.

        This is the whole point of sharing the object: two components each
        honouring 1 rps independently still produces 2 rps at the server.
        """
        limiter = RateLimiter(rps=1000.0)
        scraper = PortalScraper(settings=settings, use_cache=False, limiter=limiter)
        downloader = AttachmentDownloader(settings=settings, limiter=limiter)

        httpx_mock.add_response(url=f"{BASE}/x", text="page")
        httpx_mock.add_response(url=f"{BASE}/helpdesk/attachments/1", content=b"%PDF-1 a")

        scraper.get("/x")
        downloader.download(
            Attachment(attachment_id="1", filename="m.pdf", url=f"{BASE}/helpdesk/attachments/1")
        )
        scraper.close()
        downloader.close()

        assert scraper.limiter is downloader.limiter
        assert limiter.calls == 2

    def test_components_build_their_own_limiter_when_not_given_one(
        self, settings: Settings
    ) -> None:
        """Standalone use still rate-limits; sharing is an optimisation of honesty."""
        scraper = PortalScraper(settings=settings, use_cache=False)
        assert scraper.limiter.delay == pytest.approx(1.0 / settings.crawl.rps)
        scraper.close()
