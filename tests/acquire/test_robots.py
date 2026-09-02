"""Tests for the robots.txt gate.

This is the highest-consequence module in the acquisition layer. With no
Freshdesk API key, a wrong "allowed" verdict means crawling a site that forbade
it, and a wrong "blocked" verdict means abandoning a viable project. Both
directions are tested, as is the refusal to guess when the answer is unclear.
"""

from __future__ import annotations

import pytest
from pytest_httpx import HTTPXMock

from seeley_rag.acquire.robots import RobotsGate, check_portal
from seeley_rag.exceptions import AcquisitionError, RobotsDisallowedError
from seeley_rag.settings import Settings

BASE = "https://seeleyinternationalhelp.freshdesk.com"
ROBOTS_URL = f"{BASE}/robots.txt"

#: The portal's actual robots.txt, captured 2026-08-20. Note that
#: `Allow: /helpdesk/attachments` precedes `Disallow: /helpdesk/`, which is what
#: makes the manual PDFs fetchable at all.
REAL_ROBOTS = """
User-agent: *
Disallow: /support/search
Disallow: /support/tickets/
Disallow: /support/login
Disallow: /login/normal/
Allow: /helpdesk/attachments
Disallow: /helpdesk/
Disallow: /public/tickets/
Sitemap: https://seeleyinternationalhelp.freshdesk.com/support/sitemap.xml
"""

BLOCKING_ROBOTS = """
User-agent: *
Disallow: /support/solutions/
"""


@pytest.fixture
def gate(settings: Settings) -> RobotsGate:
    """A gate pointed at the configured portal."""
    return RobotsGate()


class TestFetch:
    """Fetching robots.txt."""

    def test_returns_body(self, gate: RobotsGate, httpx_mock: HTTPXMock) -> None:
        """A 200 gives us the rules to evaluate."""
        httpx_mock.add_response(url=ROBOTS_URL, text=REAL_ROBOTS)
        assert "Disallow: /support/search" in gate.fetch()

    def test_404_means_nothing_is_disallowed(self, gate: RobotsGate, httpx_mock: HTTPXMock) -> None:
        """No published robots.txt is permission by omission, not a failure."""
        httpx_mock.add_response(url=ROBOTS_URL, status_code=404)
        assert gate.fetch() == ""
        assert gate.report().allowed is True

    def test_server_error_is_undetermined_not_allowed(
        self, gate: RobotsGate, httpx_mock: HTTPXMock
    ) -> None:
        """An ambiguous answer must never be read as permission.

        Defaulting to "probably fine" on a 500 is exactly the wrong behaviour
        for a gate protecting someone else's server.
        """
        httpx_mock.add_response(url=ROBOTS_URL, status_code=500)
        with pytest.raises(AcquisitionError):
            gate.fetch()

    def test_network_failure_is_undetermined(self, gate: RobotsGate, httpx_mock: HTTPXMock) -> None:
        """A connection failure leaves the verdict undecided, so refuse to crawl."""
        import httpx

        httpx_mock.add_exception(httpx.ConnectError("no route to host"))
        with pytest.raises(AcquisitionError, match="undecided"):
            gate.fetch()

    def test_is_allowed_before_fetch_raises(self, gate: RobotsGate) -> None:
        """Evaluating rules that were never loaded would silently allow everything."""
        with pytest.raises(AcquisitionError, match="fetch"):
            gate.is_allowed("/support/solutions")


class TestVerdict:
    """Evaluating the required paths."""

    def test_real_robots_permits_the_crawl(self, gate: RobotsGate, httpx_mock: HTTPXMock) -> None:
        """The live rules allow solutions browsing and the attachment endpoint."""
        httpx_mock.add_response(url=ROBOTS_URL, text=REAL_ROBOTS)
        report = gate.report()
        assert report.allowed is True
        assert report.disallowed_paths == []

    def test_attachments_allow_beats_helpdesk_disallow(
        self, gate: RobotsGate, httpx_mock: HTTPXMock
    ) -> None:
        """The manuals live behind /helpdesk/attachments.

        A naive reading of `Disallow: /helpdesk/` would abandon the project even
        though the more specific Allow rule permits exactly what we need.
        """
        httpx_mock.add_response(url=ROBOTS_URL, text=REAL_ROBOTS)
        gate.fetch()
        assert gate.is_allowed("/helpdesk/attachments/47234382931") is True
        assert gate.is_allowed("/helpdesk/tickets") is False

    def test_disallowed_solutions_blocks_the_project(
        self, gate: RobotsGate, httpx_mock: HTTPXMock
    ) -> None:
        """The project-ending case is detected and reported, not glossed over.

        Note which path trips: ``Disallow: /support/solutions/`` has a trailing
        slash, so by the robots prefix rule it does *not* match the bare
        ``/support/solutions`` -- only the article path underneath it. A gate
        that checked the category listing alone would return "allowed" for a
        robots.txt that in fact forbids every article we need. That is why
        ``required_paths`` enumerates the article and attachment paths too.
        """
        httpx_mock.add_response(url=ROBOTS_URL, text=BLOCKING_ROBOTS)
        report = gate.report()
        assert report.allowed is False
        assert "/support/solutions/articles" in report.disallowed_paths
        assert report.results["/support/solutions"] is True

    def test_search_is_disallowed_but_is_not_required(
        self, gate: RobotsGate, httpx_mock: HTTPXMock
    ) -> None:
        """We never crawl /support/search, so its Disallow is irrelevant."""
        httpx_mock.add_response(url=ROBOTS_URL, text=REAL_ROBOTS)
        gate.fetch()
        assert gate.is_allowed("/support/search") is False
        assert gate.report().allowed is True

    def test_crawl_delay_is_surfaced(self, gate: RobotsGate, httpx_mock: HTTPXMock) -> None:
        """If the portal asks for a slower rate, we need to know so we can honour it."""
        httpx_mock.add_response(url=ROBOTS_URL, text="User-agent: *\nCrawl-delay: 5\nAllow: /\n")
        assert gate.report().crawl_delay == 5.0

    def test_absent_crawl_delay_is_none(self, gate: RobotsGate, httpx_mock: HTTPXMock) -> None:
        """No declared delay means our configured 1 rps stands."""
        httpx_mock.add_response(url=ROBOTS_URL, text=REAL_ROBOTS)
        assert gate.report().crawl_delay is None


class TestAssertCrawlable:
    """The hard gate."""

    def test_passes_when_allowed(self, gate: RobotsGate, httpx_mock: HTTPXMock) -> None:
        """A permitted crawl returns the report and proceeds."""
        httpx_mock.add_response(url=ROBOTS_URL, text=REAL_ROBOTS)
        assert gate.assert_crawlable().allowed is True

    def test_raises_with_an_actionable_message(
        self, gate: RobotsGate, httpx_mock: HTTPXMock
    ) -> None:
        """The exception has to tell a human what to do next.

        There is no plan C: the resolution is an API key, a bulk export, or
        written permission. The message names all three.
        """
        httpx_mock.add_response(url=ROBOTS_URL, text=BLOCKING_ROBOTS)
        with pytest.raises(RobotsDisallowedError) as excinfo:
            gate.assert_crawlable()

        message = str(excinfo.value)
        assert "/support/solutions" in message
        assert "API key" in message
        assert "written permission" in message
        assert "0002-crawl-instead-of-api" in message

    def test_evaluates_against_our_real_user_agent(
        self, gate: RobotsGate, httpx_mock: HTTPXMock
    ) -> None:
        """The verdict must describe the crawl we actually run."""
        httpx_mock.add_response(url=ROBOTS_URL, text=REAL_ROBOTS)
        report = gate.report()
        assert "SeeleyInstallerBot" in report.user_agent

    def test_agent_specific_disallow_is_honoured(
        self, gate: RobotsGate, httpx_mock: HTTPXMock
    ) -> None:
        """A rule naming our bot applies to us even when '*' is permissive."""
        httpx_mock.add_response(
            url=ROBOTS_URL,
            text=(
                "User-agent: *\nAllow: /\n\n"
                "User-agent: SeeleyInstallerBot\nDisallow: /support/solutions/\n"
            ),
        )
        assert gate.report().allowed is False


def test_check_portal_convenience_wrapper(settings: Settings, httpx_mock: HTTPXMock) -> None:
    """The wrapper used by scripts returns a full report."""
    httpx_mock.add_response(url=ROBOTS_URL, text=REAL_ROBOTS)
    assert check_portal().allowed is True
