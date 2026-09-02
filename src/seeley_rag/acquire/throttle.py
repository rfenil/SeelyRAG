"""Shared request rate limiting.

build-plan.md section 3.2, rule 2: 1 request per second, single-threaded.

The limiter is a separate object rather than a method on the scraper because a
crawl run issues two kinds of request -- article pages from
:class:`~seeley_rag.acquire.portal.PortalScraper` and attachment downloads from
:class:`~seeley_rag.acquire.attachments.AttachmentDownloader` -- against the same
server. If each throttled itself independently, the combined rate would be the
sum of the two, and the "1 req/sec" the project promises would quietly become
closer to 2. Sharing one limiter between them makes the guarantee true of the
run as a whole, which is the only unit Seeley's server cares about.
"""

from __future__ import annotations

import time


class RateLimiter:
    """Spaces requests by at least a fixed delay.

    Not thread-safe, deliberately: the crawl is single-threaded by design, and a
    lock here would imply otherwise.

    Args:
        rps: Requests per second.

    Attributes:
        delay: Minimum seconds between requests.
        calls: Number of times the limiter was consulted. This, not ``waits``,
            is what tells you every request path is actually going through it.
        waits: Number of times a caller was made to sleep. Necessarily lower
            than ``calls`` whenever real work already filled the window.
    """

    def __init__(self, rps: float = 1.0) -> None:
        if rps <= 0:
            raise ValueError("rps must be positive.")
        self.delay = 1.0 / rps
        self.calls = 0
        self.waits = 0
        self._last_request_at: float | None = None

    def wait(self) -> None:
        """Block until enough time has passed since the previous request."""
        self.calls += 1
        if self._last_request_at is not None:
            remaining = self.delay - (time.monotonic() - self._last_request_at)
            if remaining > 0:
                self.waits += 1
                time.sleep(remaining)
        self._last_request_at = time.monotonic()

    def reset(self) -> None:
        """Forget the last request time, so the next call does not wait."""
        self._last_request_at = None
