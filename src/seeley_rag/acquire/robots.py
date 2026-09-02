"""The ``robots.txt`` gate.

build-plan.md section 3.0, and risk 1 in section 13.

This is a gate, not a checkbox. With no Freshdesk API key available, the public
crawl is the only acquisition path. If ``robots.txt`` disallows
``/support/solutions/``, the acquisition stage is dead and no amount of
engineering fixes it -- the resolution is a human conversation with Seeley:

1. ask for an API key, or a bulk export of the manual PDFs;
2. ask for written permission to crawl.

There is no plan C. Run this first, before anything else is built on top of the
assumption that a crawl is possible.
"""

from __future__ import annotations

import urllib.robotparser
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse

import httpx

from seeley_rag.exceptions import AcquisitionError, RobotsDisallowedError
from seeley_rag.logging_conf import get_logger
from seeley_rag.settings import get_settings

log = get_logger(__name__)


@dataclass
class RobotsReport:
    """Outcome of evaluating ``robots.txt`` against the paths the crawl needs.

    Attributes:
        robots_url: Where ``robots.txt`` was fetched from.
        fetched: Whether the file was retrieved. A 404 counts as fetched-and-
            empty: no ``robots.txt`` means nothing is disallowed.
        user_agent: The agent string the rules were evaluated against.
        crawl_delay: ``Crawl-delay`` for our agent, if the file declares one.
        results: ``path -> allowed`` for every required path.
        raw: The file's contents, for the record.
    """

    robots_url: str
    fetched: bool
    user_agent: str
    crawl_delay: float | None = None
    results: dict[str, bool] = field(default_factory=dict)
    raw: str = ""

    @property
    def allowed(self) -> bool:
        """Whether every required path is crawlable."""
        return all(self.results.values())

    @property
    def disallowed_paths(self) -> list[str]:
        """The required paths that are forbidden."""
        return [path for path, ok in self.results.items() if not ok]


class RobotsGate:
    """Fetches and evaluates the portal's ``robots.txt``.

    Args:
        base_url: Portal origin. Defaults to the configured ``crawl.base_url``.
        user_agent: Agent to evaluate rules against. Defaults to the configured
            crawl User-Agent -- the same string the scraper actually sends, so
            the verdict describes the real crawl and not a hypothetical one.
        timeout: Fetch timeout in seconds.
    """

    def __init__(
        self,
        base_url: str | None = None,
        user_agent: str | None = None,
        timeout: float | None = None,
    ) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.crawl.base_url).rstrip("/")
        self.user_agent = user_agent or settings.crawl.user_agent(settings.project.crawler_version)
        self.timeout = timeout if timeout is not None else settings.crawl.timeout_seconds
        self.required_paths = list(settings.crawl.required_paths)
        self._parser: urllib.robotparser.RobotFileParser | None = None
        self._raw: str = ""
        self._fetched: bool = False

    @property
    def robots_url(self) -> str:
        """Absolute URL of the portal's ``robots.txt``."""
        return urljoin(self.base_url + "/", "robots.txt")

    def fetch(self, client: httpx.Client | None = None) -> str:
        """Fetch ``robots.txt`` and parse it.

        Args:
            client: Optional pre-configured HTTP client. One is created and
                closed automatically when omitted.

        Returns:
            The raw file contents. Empty string when the portal returns 404,
            which means no restrictions are published.

        Raises:
            AcquisitionError: If the file cannot be fetched, or returns a status
                other than 2xx/404. A 5xx or a connection failure is genuinely
                ambiguous, and guessing "probably allowed" on an ambiguous
                answer is exactly the wrong default for a gate.
        """
        owns_client = client is None
        if client is None:
            client = httpx.Client(
                headers={"User-Agent": self.user_agent},
                timeout=self.timeout,
                follow_redirects=True,
            )
        try:
            response = client.get(self.robots_url)
        except httpx.HTTPError as exc:
            raise AcquisitionError(
                f"Could not fetch {self.robots_url}: {exc}. The crawl gate is "
                "undecided, so no crawl may proceed. Retry, or resolve the "
                "network issue first."
            ) from exc
        finally:
            if owns_client:
                client.close()

        if response.status_code == 404:
            log.warning(
                "no robots.txt published; treating all paths as allowed",
                extra={"robots_url": self.robots_url},
            )
            self._raw = ""
        elif 200 <= response.status_code < 300:
            self._raw = response.text
        else:
            raise AcquisitionError(
                f"{self.robots_url} returned HTTP {response.status_code}. The crawl "
                "gate is undecided, so no crawl may proceed."
            )

        parser = urllib.robotparser.RobotFileParser()
        parser.set_url(self.robots_url)
        parser.parse(self._raw.splitlines())
        self._parser = parser
        self._fetched = True
        return self._raw

    def is_allowed(self, path: str) -> bool:
        """Whether the configured user-agent may fetch a path.

        Args:
            path: Absolute URL, or a path relative to the portal root.

        Returns:
            True if crawling is permitted.

        Raises:
            AcquisitionError: If :meth:`fetch` has not been called.
        """
        if self._parser is None:
            raise AcquisitionError("Call RobotsGate.fetch() before is_allowed().")
        url = path if urlparse(path).scheme else urljoin(self.base_url + "/", path.lstrip("/"))
        return bool(self._parser.can_fetch(self.user_agent, url))

    def crawl_delay(self) -> float | None:
        """Return the declared ``Crawl-delay`` for our agent, if any.

        Returns:
            Delay in seconds, or ``None`` when the file declares none. When the
            portal asks for a slower rate than our configured 1 rps, honour it.
        """
        if self._parser is None:
            return None
        raw_delay = self._parser.crawl_delay(self.user_agent)
        return float(raw_delay) if raw_delay is not None else None

    def report(self, client: httpx.Client | None = None) -> RobotsReport:
        """Evaluate every required path and summarise the verdict.

        Args:
            client: Optional pre-configured HTTP client.

        Returns:
            A :class:`RobotsReport`. This never raises on a disallow -- it
            reports. Use :meth:`assert_crawlable` when you need the gate to stop
            execution.
        """
        if not self._fetched:
            self.fetch(client=client)
        results = {path: self.is_allowed(path) for path in self.required_paths}
        return RobotsReport(
            robots_url=self.robots_url,
            fetched=self._fetched,
            user_agent=self.user_agent,
            crawl_delay=self.crawl_delay(),
            results=results,
            raw=self._raw,
        )

    def assert_crawlable(self, client: httpx.Client | None = None) -> RobotsReport:
        """Raise unless every required path is crawlable.

        Args:
            client: Optional pre-configured HTTP client.

        Returns:
            The report, when the crawl is permitted.

        Raises:
            RobotsDisallowedError: Naming the disallowed paths and the agreed
                fallbacks. Acquisition must not proceed past this.
        """
        report = self.report(client=client)
        if report.allowed:
            log.info(
                "robots.txt permits the crawl",
                extra={
                    "robots_url": report.robots_url,
                    "paths": self.required_paths,
                    "crawl_delay": report.crawl_delay,
                },
            )
            return report

        blocked = ", ".join(report.disallowed_paths)
        log.error(
            "robots.txt forbids the crawl",
            extra={"robots_url": report.robots_url, "disallowed": report.disallowed_paths},
        )
        raise RobotsDisallowedError(
            f"{report.robots_url} disallows {blocked} for User-Agent "
            f"'{report.user_agent}'.\n\n"
            "ACQUISITION IS BLOCKED. The public crawl is the only path -- "
            "/api/v2/* returns 401 and no key is available -- so this cannot be "
            "engineered around. Escalate to a human, who must obtain one of:\n"
            "  (a) a Freshdesk API key, or a bulk export of the manual PDFs;\n"
            "  (b) written permission from Seeley International to crawl.\n"
            "See _context/02-decisions/0002-crawl-instead-of-api.md."
        )


def check_portal(client: httpx.Client | None = None) -> RobotsReport:
    """Convenience wrapper: build a gate and report on the configured portal.

    Args:
        client: Optional pre-configured HTTP client.

    Returns:
        The :class:`RobotsReport`.
    """
    return RobotsGate().report(client=client)
