"""Tests for the portal scraper.

Every HTTP interaction is mocked. No test touches the live Seeley portal.
"""

from __future__ import annotations

import httpx
import pytest
from pytest_httpx import HTTPXMock

from seeley_rag.acquire.portal import PortalScraper, collapse_whitespace, strip_boilerplate
from seeley_rag.exceptions import AcquisitionError, RateLimitedError
from seeley_rag.settings import Settings

BASE = "https://seeleyinternationalhelp.freshdesk.com"

SOLUTIONS_HTML = """
<html><body>
  <div class="cs-s">
    <h3 class="heading accordion-heading">
      <a href="/support/solutions/47000154481">DUCTED GAS HEATING (DGH) SERVICE AND INSTALLATION</a>
    </h3>
    <div class="cs-g-c accordion-content">
      <section class="cs-g article-list"><div class="list-lead">
        <a href="/support/solutions/folders/47000783696" title="Service Guides">
          Service Guides <span class='item-count'>5</span></a>
      </div></section>
      <section class="cs-g article-list"><div class="list-lead">
        <a href="/support/solutions/folders/47000225980"
           title="Diagnostics and Specific Fault Finding">
          Diagnostics <span class='item-count'>80</span></a>
      </div></section>
    </div>
  </div>
  <div class="cs-s">
    <h3 class="heading accordion-heading">
      <a href="/support/solutions/47000154484">REVERSE CYCLE SERVICE AND INSTALLATION</a>
    </h3>
    <div class="cs-g-c accordion-content">
      <section class="cs-g article-list"><div class="list-lead">
        <a href="/support/solutions/folders/47000792152" title="Reverse Cycle R/C Braemar">
          RC Braemar <span class='item-count'>6</span></a>
      </div></section>
    </div>
  </div>
</body></html>
"""


def _category_block(category_id: str, category: str, folder_id: str, folder: str) -> str:
    """Render one category block of the solutions index."""
    return f"""
  <div class="cs-s">
    <h3 class="heading accordion-heading">
      <a href="/support/solutions/{category_id}">{category}</a>
    </h3>
    <div class="cs-g-c accordion-content">
      <section class="cs-g article-list"><div class="list-lead">
        <a href="/support/solutions/folders/{folder_id}" title="{folder}">{folder}</a>
      </div></section>
    </div>
  </div>
"""


#: Reproduces the real naming collision: the pilot Reverse Cycle category's full
#: name is a substring of the VRF category's.
VRF_SOLUTIONS_HTML = (
    "<html><body>"
    + _category_block("1", "REVERSE CYCLE SERVICE AND INSTALLATION", "100", "Service Manuals")
    + _category_block("2", "VRF REVERSE CYCLE SERVICE AND INSTALLATION", "101", "VRF Manuals")
    + _category_block(
        "3",
        "COMMERCIAL COOLING - BRAEMAR DIRECT EVAPORATIVE &amp; CLIMATE WIZARD",
        "102",
        "Evap Manuals",
    )
    + "</body></html>"
)


def page_html(*article_ids: str) -> str:
    """Build a folder page listing the given article IDs.

    Args:
        *article_ids: Article IDs to list. No IDs makes an empty page, which is
            what the portal returns past the last page.

    Returns:
        Folder page HTML.
    """
    rows = "".join(
        f'<div class="c-row c-article-row"><div class="ellipsis article-title">'
        f'<a href="/support/solutions/articles/{aid}-slug" class="c-link">Article {aid}</a>'
        f"</div></div>"
        for aid in article_ids
    )
    return f'<html><body><section class="article-list c-list">{rows}</section></body></html>'


@pytest.fixture
def scraper(settings: Settings) -> PortalScraper:
    """A scraper with caching and throttling disabled, for fast offline tests."""
    instance = PortalScraper(settings=settings, use_cache=False)
    instance.delay = 0.0
    return instance


class TestFetching:
    """Request behaviour, retries, and blocking."""

    def test_get_returns_body(self, scraper: PortalScraper, httpx_mock: HTTPXMock) -> None:
        """A 200 returns the response text and counts as a network fetch."""
        httpx_mock.add_response(url=f"{BASE}/support/solutions", text="<html>ok</html>")
        assert scraper.get("/support/solutions") == "<html>ok</html>"
        assert scraper.fetch_count == 1

    @pytest.mark.parametrize("status", [403, 429])
    def test_blocked_status_raises_immediately(
        self, scraper: PortalScraper, httpx_mock: HTTPXMock, status: int
    ) -> None:
        """429 and 403 stop the crawl dead rather than retrying into a block.

        With no API key there is no fallback channel, so retrying is how the
        project ends.
        """
        httpx_mock.add_response(url=f"{BASE}/x", status_code=status)
        with pytest.raises(RateLimitedError):
            scraper.get("/x")
        assert len(httpx_mock.get_requests()) == 1

    def test_server_error_is_retried_then_gives_up(
        self, scraper: PortalScraper, httpx_mock: HTTPXMock
    ) -> None:
        """5xx is retried up to the configured limit, then reported."""
        scraper.config.retry_backoff_seconds = 0.001
        for _ in range(scraper.config.max_retries):
            httpx_mock.add_response(url=f"{BASE}/x", status_code=503)
        with pytest.raises(AcquisitionError):
            scraper.get("/x")
        assert len(httpx_mock.get_requests()) == scraper.config.max_retries

    def test_server_error_then_success(self, scraper: PortalScraper, httpx_mock: HTTPXMock) -> None:
        """A transient 5xx recovers on retry."""
        scraper.config.retry_backoff_seconds = 0.001
        httpx_mock.add_response(url=f"{BASE}/x", status_code=502)
        httpx_mock.add_response(url=f"{BASE}/x", text="recovered")
        assert scraper.get("/x") == "recovered"

    def test_404_is_not_retried(self, scraper: PortalScraper, httpx_mock: HTTPXMock) -> None:
        """A 404 is a permanent answer; retrying it wastes politeness budget."""
        httpx_mock.add_response(url=f"{BASE}/missing", status_code=404)
        with pytest.raises(AcquisitionError):
            scraper.get("/missing")
        assert len(httpx_mock.get_requests()) == 1

    def test_honest_user_agent_with_contact(
        self, scraper: PortalScraper, httpx_mock: HTTPXMock
    ) -> None:
        """The crawl identifies itself and gives Seeley a way to reach us."""
        httpx_mock.add_response(url=f"{BASE}/x", text="ok")
        scraper.get("/x")
        agent = httpx_mock.get_requests()[0].headers["User-Agent"]
        assert "SeeleyInstallerBot" in agent
        assert "@" in agent


class TestCache:
    """The on-disk HTML cache."""

    def test_second_fetch_is_served_from_disk(
        self, settings: Settings, temp_data_root: object, httpx_mock: HTTPXMock
    ) -> None:
        """Re-running the crawl must not re-hit the server.

        Without this the crawl is 25 minutes and ~1,500 requests every single
        iteration, against someone else's production site.
        """
        instance = PortalScraper(settings=settings, use_cache=True)
        instance.delay = 0.0
        httpx_mock.add_response(url=f"{BASE}/support/solutions", text="<html>cached</html>")

        first = instance.get("/support/solutions")
        second = instance.get("/support/solutions")

        assert first == second == "<html>cached</html>"
        assert len(httpx_mock.get_requests()) == 1
        assert instance.fetch_count == 1
        assert instance.cache_hits == 1

    def test_cache_bytes_are_tracked_separately(
        self, settings: Settings, temp_data_root: object, httpx_mock: HTTPXMock
    ) -> None:
        """Network and cache volume are reported apart, so a re-run is legible."""
        instance = PortalScraper(settings=settings, use_cache=True)
        instance.delay = 0.0
        httpx_mock.add_response(url=f"{BASE}/x", text="12345")
        instance.get("/x")
        instance.get("/x")
        stats = instance.stats()
        assert stats["html_bytes_fetched"] == 5
        assert stats["html_bytes_from_cache"] == 5
        assert stats["html_bytes_total"] == 10


class TestListFolders:
    """Folder discovery from the solutions index."""

    def test_folders_carry_their_category(
        self, scraper: PortalScraper, httpx_mock: HTTPXMock
    ) -> None:
        """Category comes from the enclosing block, and drives product routing."""
        httpx_mock.add_response(url=f"{BASE}/support/solutions", text=SOLUTIONS_HTML)
        folders = scraper.list_folders()
        assert len(folders) == 3
        by_id = {f["id"]: f for f in folders}
        assert by_id["47000783696"]["category"].startswith("DUCTED GAS HEATING")
        assert by_id["47000792152"]["category"].startswith("REVERSE CYCLE")

    def test_folder_name_comes_from_title_attribute(
        self, scraper: PortalScraper, httpx_mock: HTTPXMock
    ) -> None:
        """The link text has the article count appended; the title is clean."""
        httpx_mock.add_response(url=f"{BASE}/support/solutions", text=SOLUTIONS_HTML)
        folders = {f["id"]: f["name"] for f in scraper.list_folders()}
        assert folders["47000783696"] == "Service Guides"
        assert "5" not in folders["47000783696"]

    def test_select_folders_matches_category_case_insensitively(
        self, scraper: PortalScraper, httpx_mock: HTTPXMock
    ) -> None:
        """The portal's category names are long; a short configured name must match.

        "Ducted Gas Heating (DGH)" has to select "DUCTED GAS HEATING (DGH)
        SERVICE AND INSTALLATION", or the pilot silently crawls nothing.
        """
        httpx_mock.add_response(url=f"{BASE}/support/solutions", text=SOLUTIONS_HTML)
        selected = scraper.select_folders(["Ducted Gas Heating (DGH)"])
        assert len(selected) == 2
        assert all(f["category"].startswith("DUCTED") for f in selected)

    def test_no_categories_selects_everything(
        self, scraper: PortalScraper, httpx_mock: HTTPXMock
    ) -> None:
        """An empty selection means the whole corpus, not nothing."""
        httpx_mock.add_response(url=f"{BASE}/support/solutions", text=SOLUTIONS_HTML)
        assert len(scraper.select_folders(None)) == 3

    def test_prefix_match_excludes_the_vrf_product_line(
        self, scraper: PortalScraper, httpx_mock: HTTPXMock
    ) -> None:
        """ "Reverse Cycle ..." must not drag in "VRF Reverse Cycle ...".

        The pilot category's full name is a substring of the VRF category's, so
        pure substring matching cannot separate them and would silently widen
        the pilot by ~123 articles.
        """
        httpx_mock.add_response(url=f"{BASE}/support/solutions", text=VRF_SOLUTIONS_HTML)
        selected = scraper.select_folders(["Reverse Cycle Service and Installation"])
        assert [f["id"] for f in selected] == ["100"]

    def test_substring_fallback_still_works(
        self, scraper: PortalScraper, httpx_mock: HTTPXMock
    ) -> None:
        """A needle matching nothing by prefix falls back to substring.

        This keeps short mid-name selections like "Evaporative" usable against
        "COMMERCIAL COOLING - BRAEMAR DIRECT EVAPORATIVE ...".
        """
        httpx_mock.add_response(url=f"{BASE}/support/solutions", text=VRF_SOLUTIONS_HTML)
        selected = scraper.select_folders(["evaporative"])
        assert [f["id"] for f in selected] == ["102"]

    def test_selection_is_deduplicated(self, scraper: PortalScraper, httpx_mock: HTTPXMock) -> None:
        """Overlapping needles must not crawl a folder twice."""
        httpx_mock.add_response(url=f"{BASE}/support/solutions", text=VRF_SOLUTIONS_HTML)
        selected = scraper.select_folders(["Reverse Cycle", "Reverse Cycle Service"])
        assert len(selected) == len({f["id"] for f in selected})


class TestPagination:
    """Folder pagination.

    The portal paginates at /folders/{id}/page/{N}. The ?page=N form documented
    in the build plan silently returns page 1.
    """

    def test_follows_pages_until_empty(self, scraper: PortalScraper, httpx_mock: HTTPXMock) -> None:
        """All pages are collected, and the empty page past the end stops the walk."""
        folder = "47000225980"
        httpx_mock.add_response(
            url=f"{BASE}/support/solutions/folders/{folder}", text=page_html("1", "2")
        )
        httpx_mock.add_response(
            url=f"{BASE}/support/solutions/folders/{folder}/page/2", text=page_html("3", "4")
        )
        httpx_mock.add_response(
            url=f"{BASE}/support/solutions/folders/{folder}/page/3", text=page_html()
        )

        articles = scraper.list_articles(folder)
        assert [a["id"] for a in articles] == ["1", "2", "3", "4"]

    def test_uses_path_pagination_not_query_string(
        self, scraper: PortalScraper, httpx_mock: HTTPXMock
    ) -> None:
        """Guards the deviation from the build plan.

        If this ever regresses to ?page=N, the crawl captures 10 articles from an
        80-article folder and reports success.
        """
        folder = "47000225980"
        httpx_mock.add_response(
            url=f"{BASE}/support/solutions/folders/{folder}", text=page_html("1")
        )
        httpx_mock.add_response(
            url=f"{BASE}/support/solutions/folders/{folder}/page/2", text=page_html()
        )
        scraper.list_articles(folder)
        requested = [str(r.url) for r in httpx_mock.get_requests()]
        assert f"{BASE}/support/solutions/folders/{folder}/page/2" in requested
        assert not any("?page=" in url for url in requested)

    def test_repeated_page_stops_the_walk(
        self, scraper: PortalScraper, httpx_mock: HTTPXMock
    ) -> None:
        """A page echoing the previous one terminates instead of looping forever."""
        folder = "999"
        httpx_mock.add_response(
            url=f"{BASE}/support/solutions/folders/{folder}", text=page_html("1", "2")
        )
        httpx_mock.add_response(
            url=f"{BASE}/support/solutions/folders/{folder}/page/2", text=page_html("1", "2")
        )
        assert [a["id"] for a in scraper.list_articles(folder)] == ["1", "2"]


class TestArticleParsing:
    """Parsing an article page into an Article."""

    def test_stub_article(self, scraper: PortalScraper, article_stub_html: str) -> None:
        """The representative stub parses to a short body and one attachment."""
        article = scraper.parse_article(
            article_stub_html, article_id="47001247136", url="https://x/a/47001247136"
        )
        assert article.title == "TQ Service Guide Gas Ducted Heater 644066 M"
        assert article.category.startswith("DUCTED GAS HEATING")
        assert article.folder == "Service Guides"
        assert article.folder_id == "47000783696"
        assert article.is_stub is True
        assert article.content_stream == "pdf"
        assert len(article.attachments) == 1

    def test_stub_attachment_uses_full_filename(
        self, scraper: PortalScraper, article_stub_html: str
    ) -> None:
        """The displayed link text is truncated; the title attribute is complete.

        The filename carries the manual's part number, which is how a citation
        is recognised by an installer.
        """
        article = scraper.parse_article(
            article_stub_html, article_id="47001247136", url="https://x/a/1"
        )
        attachment = article.attachments[0]
        assert attachment.filename == "644066-M MANUAL SERVICE TQ SERIES.pdf"
        assert attachment.attachment_id == "47234382931"
        assert not attachment.filename.endswith("...")

    def test_content_article(self, scraper: PortalScraper, article_content_html: str) -> None:
        """A real diagnostic article is content, not a stub."""
        article = scraper.parse_article(
            article_content_html, article_id="47001137472", url="https://x/a/47001137472"
        )
        assert article.is_stub is False
        assert article.content_stream == "diagnostic_article"
        assert article.attachments == []
        assert "FC7" in article.body_text

    def test_boilerplate_is_stripped_from_both_streams(
        self, scraper: PortalScraper, article_stub_html: str, article_content_html: str
    ) -> None:
        """The shared safety notice is removed before classification.

        Left in, its 1026 characters push every stub past the 200-char threshold
        and ~900 card-catalogue articles are indexed as content.
        """
        stub = scraper.parse_article(article_stub_html, article_id="1", url="https://x/1")
        content = scraper.parse_article(article_content_html, article_id="2", url="https://x/2")
        for article in (stub, content):
            assert "must only be installed, commissioned" not in article.body_text
            assert "Safety takes priority over diagnosis" not in article.body_text
        assert stub.body_char_count < 200
        assert content.body_char_count > 200

    def test_updated_at_is_parsed(self, scraper: PortalScraper, article_stub_html: str) -> None:
        """The article page exposes a modified date, so record it."""
        article = scraper.parse_article(article_stub_html, article_id="1", url="https://x/1")
        assert article.updated_at is not None
        assert "2026" in article.updated_at

    def test_explicit_folder_context_wins_over_breadcrumb(
        self, scraper: PortalScraper, article_stub_html: str
    ) -> None:
        """During a folder walk the listing context is authoritative."""
        article = scraper.parse_article(
            article_stub_html,
            article_id="1",
            url="https://x/1",
            category="Custom Category",
            folder="Custom Folder",
            folder_id="123",
        )
        assert article.category == "Custom Category"
        assert article.folder == "Custom Folder"

    def test_missing_body_raises_actionable_error(self, scraper: PortalScraper) -> None:
        """A markup change must fail loudly, naming what to re-check."""
        with pytest.raises(AcquisitionError, match="selectors"):
            scraper.parse_article("<html><body>nothing</body></html>", "1", "https://x/1")

    def test_inline_images_are_not_treated_as_attachments(self, scraper: PortalScraper) -> None:
        """Body images point at S3 directly and are not documents."""
        html = (
            '<html><body><article class="article-body">'
            '<img src="https://s3.amazonaws.com/cdn.freshdesk.com/data/helpdesk/'
            'attachments/production/1/original/x.png">body text</article></body></html>'
        )
        article = scraper.parse_article(html, "1", "https://x/1")
        assert article.attachments == []


class TestIterArticles:
    """The folder walk."""

    def test_limit_stops_the_walk(
        self, scraper: PortalScraper, httpx_mock: HTTPXMock, article_stub_html: str
    ) -> None:
        """--limit exists for smoke tests and must not over-fetch."""
        httpx_mock.add_response(url=f"{BASE}/support/solutions", text=SOLUTIONS_HTML)
        httpx_mock.add_response(
            url=f"{BASE}/support/solutions/folders/47000783696", text=page_html("11", "12", "13")
        )
        httpx_mock.add_response(
            url=f"{BASE}/support/solutions/folders/47000783696/page/2", text=page_html()
        )
        # Only the first two article pages may be fetched; registering the third
        # would fail the run, which is exactly the assertion we want.
        for article_id in ("11", "12"):
            httpx_mock.add_response(
                url=f"{BASE}/support/solutions/articles/{article_id}-slug",
                text=article_stub_html,
            )

        articles = list(scraper.iter_articles(["Ducted Gas Heating (DGH)"], limit=2))
        assert len(articles) == 2
        assert scraper.articles_parsed == 2
        assert not any("13-slug" in str(r.url) for r in httpx_mock.get_requests())


class TestHelpers:
    """Text helpers."""

    def test_collapse_whitespace(self) -> None:
        """Extracted text arrives full of newlines and runs of spaces."""
        assert collapse_whitespace("  a\n\n  b\t c ") == "a b c"

    def test_strip_boilerplate_is_a_no_op_without_markers(self, settings: Settings) -> None:
        """Articles that never carried the notice are left untouched."""
        settings.articles.boilerplate_markers = []
        assert strip_boilerplate("hello world", settings) == "hello world"

    def test_strip_boilerplate_tolerates_a_missing_end_marker(self, settings: Settings) -> None:
        """A half-matched marker must not silently truncate the body."""
        text = "before Safety Notice and then nothing closes it"
        assert strip_boilerplate(text, settings) == text


def test_absolute_url_leaves_absolute_urls_alone(scraper: PortalScraper) -> None:
    """Article URLs arrive already absolute from the listing."""
    assert scraper.absolute_url("https://other/x") == "https://other/x"
    assert scraper.absolute_url("/support/solutions") == f"{BASE}/support/solutions"


def test_scraper_closes_client_it_created(settings: Settings) -> None:
    """The scraper owns the client it made, and not one that was injected."""
    injected = httpx.Client()
    instance = PortalScraper(settings=settings, client=injected)
    instance.close()
    assert not injected.is_closed
    injected.close()
