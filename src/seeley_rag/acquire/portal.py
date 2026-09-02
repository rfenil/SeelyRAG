"""Public-portal scraper. The only implementation of :class:`FreshdeskSource`.

build-plan.md sections 3.2 and 3.4.

Crawl etiquette is load-bearing, not decorative. With no API key there is no
fallback channel, so being blocked ends the project (ADR 0002). The rules, all
enforced below:

1. **Cache every fetch to disk, keyed by URL.** You will re-run this many times
   in a day; without the cache each run is 25 minutes and another 1,500
   requests against someone else's server.
2. **1 req/sec, single-threaded.** Concurrency buys nothing here and risks a
   block.
3. **Honest User-Agent with a contact address.**
4. **Stop immediately on 429 or 403.** Never retry into a block. Retries are for
   5xx and timeouts only.

PAGINATION
----------
Folder pagination is ``/support/solutions/folders/{id}/page/{N}``. The
``?page=N`` form documented in the build plan is a silent no-op -- the portal
returns page 1 for it, so a crawler using it would capture 10 of 80 articles and
then stop, believing the folder exhausted. Verified against the live portal; see
``_context/03-research/portal-recon.md`` and ADR 0004.
"""

from __future__ import annotations

import re
from typing import Iterator
from urllib.parse import urljoin, urlparse

import httpx
from selectolax.parser import HTMLParser, Node
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from seeley_rag.acquire.base import Article, Attachment, FreshdeskSource
from seeley_rag.acquire.throttle import RateLimiter
from seeley_rag.exceptions import AcquisitionError, RateLimitedError
from seeley_rag.logging_conf import get_logger
from seeley_rag.paths import RAW_HTML_DIR, html_cache_path
from seeley_rag.settings import Settings, get_settings

log = get_logger(__name__)

#: ``/support/solutions/articles/47001247136-tq-service-guide-...`` -> the ID.
ARTICLE_ID_RE = re.compile(r"/support/solutions/articles/(\d+)")
#: ``/support/solutions/folders/47000783696`` -> the ID.
FOLDER_ID_RE = re.compile(r"/support/solutions/folders/(\d+)")
#: ``/support/solutions/47000154481`` -> the category ID (no ``folders`` segment).
CATEGORY_ID_RE = re.compile(r"/support/solutions/(\d+)")
#: ``/helpdesk/attachments/47234382931`` -> the attachment ID.
ATTACHMENT_ID_RE = re.compile(r"/helpdesk/attachments/(\d+)")
#: ``Modified on: Thu, 16 Jul, 2026 at  9:50 AM``
MODIFIED_RE = re.compile(r"Modified on:\s*(.+?)\s*$", re.MULTILINE)


class _RetryableStatus(Exception):
    """Internal signal: a 5xx worth retrying. Never escapes this module."""


def collapse_whitespace(text: str) -> str:
    """Collapse all runs of whitespace to single spaces and strip the ends.

    Args:
        text: Raw extracted text.

    Returns:
        Normalised single-line text.
    """
    return re.sub(r"\s+", " ", text).strip()


def strip_boilerplate(text: str, settings: Settings | None = None) -> str:
    """Remove the shared safety-notice boilerplate from an article body.

    The portal appends a 1026-character safety notice to article bodies,
    byte-identical across articles. Left in place it does two kinds of damage:
    it pushes every card-catalogue stub past the 200-character threshold so the
    stub rule misclassifies them as content, and it dilutes the embedded text of
    the genuinely valuable diagnostic articles with a notice shared by ~900
    others.

    Removing it is not a loss of safety information: the generation prompt
    carries the licensed-technician framing (build-plan section 8), which is
    where that requirement actually belongs.

    Args:
        text: Collapsed body text.
        settings: Settings override, for tests.

    Returns:
        The body with each configured boilerplate span removed. Markers that do
        not match are skipped, so this is safe on articles that lack the notice.
    """
    resolved = settings or get_settings()
    result = text
    for marker in resolved.articles.boilerplate_markers:
        start = result.find(marker.start)
        if start < 0:
            continue
        end = result.find(marker.end, start)
        if end < 0:
            continue
        result = result[:start] + " " + result[end + len(marker.end) :]
    return collapse_whitespace(result)


class PortalScraper(FreshdeskSource):
    """Scrapes solution categories, folders and articles from the public portal.

    Args:
        settings: Settings override, for tests.
        client: Pre-configured HTTP client. One is built from settings when
            omitted. Tests inject a mock transport here.
        rps: Requests per second override. Ignored when ``limiter`` is given.
            Raising it is not a supported way to make the crawl faster.
        limiter: Shared rate limiter. Pass the same instance to the attachment
            downloader so the run as a whole honours 1 rps.
        use_cache: Whether to read and write the on-disk HTML cache. Only turn
            this off deliberately -- it is what makes iteration cheap.

    Attributes:
        fetch_count: Requests actually sent over the network this run.
        cache_hits: Requests served from disk this run.
        bytes_fetched: HTML bytes pulled over the network this run.
        bytes_from_cache: HTML bytes served from the on-disk cache this run.
        folders_listed: Folders returned by the most recent listing.
        articles_parsed: Articles parsed by :meth:`iter_articles` this run.
        articles_skipped: Articles skipped because a previous run already
            acquired them. Their pages are never fetched.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        client: httpx.Client | None = None,
        rps: float | None = None,
        use_cache: bool = True,
        limiter: RateLimiter | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.config = self.settings.crawl
        self.version = self.settings.project.crawler_version
        # A shared limiter keeps the whole run at 1 rps, not 1 rps per component.
        self.limiter = limiter or RateLimiter(rps or self.config.rps)
        self.use_cache = use_cache
        self.fetch_count = 0
        self.cache_hits = 0
        self.bytes_fetched = 0
        self.bytes_from_cache = 0
        self.folders_listed = 0
        self.articles_parsed = 0
        self.articles_skipped = 0
        self._owns_client = client is None
        self.client = client or httpx.Client(
            headers={"User-Agent": self.config.user_agent(self.version)},
            timeout=self.config.timeout_seconds,
            follow_redirects=True,
        )

    # -- lifecycle ---------------------------------------------------------

    def stats(self) -> dict[str, int]:
        """Return counters describing how much this scraper has pulled.

        Returns:
            Request counts and HTML byte volumes, network and cache separated.
            The split matters: a re-run that shows all cache and no network is
            the crawl behaving correctly, not a crawl that did nothing.
        """
        return {
            "requests": self.fetch_count,
            "cache_hits": self.cache_hits,
            "html_bytes_fetched": self.bytes_fetched,
            "html_bytes_from_cache": self.bytes_from_cache,
            "html_bytes_total": self.bytes_fetched + self.bytes_from_cache,
            "folders_listed": self.folders_listed,
            "articles_parsed": self.articles_parsed,
            "articles_skipped": self.articles_skipped,
        }

    def close(self) -> None:
        """Close the HTTP client, if this scraper created it."""
        if self._owns_client:
            self.client.close()

    def __enter__(self) -> PortalScraper:
        """Enter the context manager."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the client on exit."""
        self.close()

    # -- fetching ----------------------------------------------------------

    def absolute_url(self, path: str) -> str:
        """Resolve a portal-relative path to an absolute URL.

        Args:
            path: Absolute URL or root-relative path.

        Returns:
            An absolute URL.
        """
        if urlparse(path).scheme:
            return path
        return urljoin(self.config.base_url + "/", path.lstrip("/"))

    def _request(self, url: str) -> str:
        """Perform one throttled HTTP GET, retrying only 5xx and timeouts.

        Args:
            url: Absolute URL.

        Returns:
            The response body.

        Raises:
            RateLimitedError: On 429 or 403. Never retried.
            AcquisitionError: On any other non-2xx, or after retries are spent.
        """

        @retry(
            retry=retry_if_exception_type((_RetryableStatus, httpx.TimeoutException)),
            stop=stop_after_attempt(self.config.max_retries),
            wait=wait_exponential(multiplier=self.config.retry_backoff_seconds),
            reraise=True,
        )
        def _attempt() -> str:
            self.limiter.wait()
            try:
                response = self.client.get(url)
            except httpx.TimeoutException:
                log.warning("timeout; will retry", extra={"url": url})
                raise
            except httpx.HTTPError as exc:
                raise AcquisitionError(f"GET {url} failed: {exc}") from exc

            status = response.status_code
            if status in (403, 429):
                # Stop dead. With no API key there is no fallback channel, and
                # retrying into a block is how the project ends.
                log.error("blocked by portal; halting", extra={"url": url, "status": status})
                raise RateLimitedError(
                    f"GET {url} returned HTTP {status}. Halting immediately rather than "
                    "retrying into a block: the public crawl is the only acquisition "
                    "path. Escalate to a human before running this again, and consider "
                    "that the portal may have rate-limited this User-Agent or IP."
                )
            if 500 <= status < 600:
                log.warning("server error; will retry", extra={"url": url, "status": status})
                raise _RetryableStatus(f"HTTP {status}")
            if not 200 <= status < 300:
                raise AcquisitionError(f"GET {url} returned HTTP {status}.")

            self.fetch_count += 1
            log.info(
                "fetched", extra={"url": url, "status": status, "bytes": len(response.content)}
            )
            return response.text

        try:
            return _attempt()
        except _RetryableStatus as exc:
            raise AcquisitionError(
                f"GET {url} still failing after {self.config.max_retries} attempts: {exc}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise AcquisitionError(
                f"GET {url} timed out after {self.config.max_retries} attempts."
            ) from exc

    def get(self, path: str) -> str:
        """Fetch a page, using the on-disk cache when possible.

        Args:
            path: Absolute URL or root-relative portal path.

        Returns:
            The page HTML.

        Raises:
            RateLimitedError: If the portal blocks us.
            AcquisitionError: On any other fetch failure.
        """
        url = self.absolute_url(path)
        cache_path = html_cache_path(url)
        if self.use_cache and cache_path.exists():
            cached = cache_path.read_text(encoding="utf-8")
            self.cache_hits += 1
            self.bytes_from_cache += len(cached.encode("utf-8"))
            log.debug("cache hit", extra={"url": url, "path": str(cache_path)})
            return cached

        html = self._request(url)
        self.bytes_fetched += len(html.encode("utf-8"))
        if self.use_cache:
            RAW_HTML_DIR.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(html, encoding="utf-8")
        return html

    # -- FreshdeskSource ---------------------------------------------------

    def list_folders(self) -> list[dict[str, str]]:
        """List every solution folder, with the category it belongs to.

        The solutions index carries all 143 folders grouped under their
        categories, so this is a single request. Category names come from the
        enclosing ``div.cs-s`` block's heading.

        Returns:
            One dict per folder with ``id``, ``name``, ``url``, ``category`` and
            ``category_id``.
        """
        tree = HTMLParser(self.get(self.config.solutions_path))
        folders: list[dict[str, str]] = []
        seen: set[str] = set()

        for section in tree.css("div.cs-s"):
            heading = section.css_first("h3.heading a, h3.accordion-heading a")
            category = collapse_whitespace(heading.text()) if heading else ""
            category_id = ""
            if heading:
                match = CATEGORY_ID_RE.search(heading.attributes.get("href") or "")
                if match:
                    category_id = match.group(1)

            for anchor in section.css("a[href*='/support/solutions/folders/']"):
                href = anchor.attributes.get("href") or ""
                match = FOLDER_ID_RE.search(href)
                if not match or match.group(1) in seen:
                    continue
                folder_id = match.group(1)
                seen.add(folder_id)
                # The title attribute is the clean name; the link text has the
                # article count appended by a nested <span>.
                name = anchor.attributes.get("title") or collapse_whitespace(anchor.text())
                folders.append(
                    {
                        "id": folder_id,
                        "name": collapse_whitespace(name),
                        "url": self.absolute_url(href),
                        "category": category,
                        "category_id": category_id,
                    }
                )

        self.folders_listed = len(folders)
        log.info("listed folders", extra={"count": len(folders)})
        return folders

    def _folder_page_url(self, folder_id: str, page: int) -> str:
        """Build a folder page URL.

        Args:
            folder_id: Freshdesk folder ID.
            page: 1-based page number.

        Returns:
            The folder URL for page 1, or the ``/page/{N}`` form beyond it.
        """
        base = f"{self.config.solutions_path}/folders/{folder_id}"
        return base if page <= 1 else f"{base}/page/{page}"

    @staticmethod
    def _article_links(tree: HTMLParser) -> list[dict[str, str]]:
        """Extract article links from a folder page.

        Args:
            tree: Parsed folder page.

        Returns:
            One dict per article with ``id``, ``title`` and ``url``.
        """
        articles: list[dict[str, str]] = []
        selector = "div.article-title a, a.c-link[href*='/support/solutions/articles/']"
        seen: set[str] = set()
        for anchor in tree.css(selector):
            href = anchor.attributes.get("href") or ""
            match = ARTICLE_ID_RE.search(href)
            if not match or match.group(1) in seen:
                continue
            seen.add(match.group(1))
            articles.append(
                {
                    "id": match.group(1),
                    "title": collapse_whitespace(anchor.text()),
                    "url": href,
                }
            )
        return articles

    def list_articles(self, folder_id: str) -> list[dict[str, str]]:
        """List every article in a folder, following pagination to the end.

        Uses ``/folders/{id}/page/{N}``. Stops when a page yields no article
        link not already seen -- which covers both the empty page the portal
        returns past the end and any future change that starts echoing page 1.

        Args:
            folder_id: Freshdesk folder ID.

        Returns:
            One dict per article with ``id``, ``title`` and ``url``.
        """
        collected: list[dict[str, str]] = []
        seen: set[str] = set()

        for page in range(1, self.config.max_pages_per_folder + 1):
            tree = HTMLParser(self.get(self._folder_page_url(folder_id, page)))
            found = self._article_links(tree)
            fresh = [a for a in found if a["id"] not in seen]
            if not fresh:
                break
            for article in fresh:
                seen.add(article["id"])
                article["url"] = self.absolute_url(article["url"])
                collected.append(article)
        else:
            log.warning(
                "hit max_pages_per_folder; folder may be truncated",
                extra={"folder_id": folder_id, "max_pages": self.config.max_pages_per_folder},
            )

        log.info("listed articles", extra={"folder_id": folder_id, "count": len(collected)})
        return collected

    # -- article parsing ---------------------------------------------------

    @staticmethod
    def _breadcrumbs(tree: HTMLParser) -> list[Node]:
        """Return the breadcrumb anchors, if the page has a breadcrumb."""
        crumb = tree.css_first("div.breadcrumb")
        return crumb.css("a") if crumb else []

    def _parse_attachments(self, tree: HTMLParser) -> list[Attachment]:
        """Extract attachment metadata from an article page.

        Scoped to ``/helpdesk/attachments/`` hrefs, which excludes the inline
        images in article bodies -- those point at S3 directly and are not
        documents.

        Args:
            tree: Parsed article page.

        Returns:
            One :class:`Attachment` per attached file, without bytes. The
            downloader fills in ``sha256``, ``size_bytes`` and ``stored_path``.
        """
        attachments: list[Attachment] = []
        seen: set[str] = set()
        for anchor in tree.css("a[href*='/helpdesk/attachments/']"):
            href = anchor.attributes.get("href") or ""
            match = ATTACHMENT_ID_RE.search(href)
            if not match or match.group(1) in seen:
                continue
            attachment_id = match.group(1)
            seen.add(attachment_id)
            # The link text is truncated for display ("644066-M MAN..."); the
            # full filename is in the title attribute.
            filename = anchor.attributes.get("title") or collapse_whitespace(anchor.text())
            attachments.append(
                Attachment(
                    attachment_id=attachment_id,
                    filename=collapse_whitespace(filename),
                    url=self.absolute_url(href),
                )
            )
        return attachments

    def parse_article(
        self,
        html: str,
        article_id: str,
        url: str,
        category: str = "",
        folder: str = "",
        folder_id: str = "",
    ) -> Article:
        """Parse an article page into an :class:`Article`.

        Category and folder are taken from the page's own breadcrumb when the
        caller does not supply them, so a single article URL can be fetched
        without first walking the folder tree.

        Args:
            html: The article page HTML.
            article_id: Freshdesk article ID.
            url: Absolute article URL, used as the citation target.
            category: Category name; falls back to the breadcrumb.
            folder: Folder name; falls back to the breadcrumb.
            folder_id: Folder ID; falls back to the breadcrumb.

        Returns:
            The parsed article, with ``is_stub`` and ``content_stream`` computed.

        Raises:
            AcquisitionError: If the page has no article body.
        """
        tree = HTMLParser(html)

        crumbs = self._breadcrumbs(tree)
        if not category or not folder:
            for anchor in crumbs:
                href = anchor.attributes.get("href") or ""
                folder_match = FOLDER_ID_RE.search(href)
                if folder_match:
                    folder = folder or collapse_whitespace(anchor.text())
                    folder_id = folder_id or folder_match.group(1)
                    continue
                category_match = CATEGORY_ID_RE.search(href)
                if category_match:
                    category = category or collapse_whitespace(anchor.text())

        heading = tree.css_first("h2.heading")
        title = ""
        if heading:
            # The heading contains a nested "Print" control; drop it first.
            for node in heading.css("a, span"):
                node.decompose()
            title = collapse_whitespace(heading.text())

        body_node = tree.css_first("article.article-body, #article-body")
        if body_node is None:
            raise AcquisitionError(
                f"No article body found at {url}. The portal markup may have changed; "
                "re-check the selectors in acquire/portal.py against a live page."
            )
        body_text = strip_boilerplate(
            collapse_whitespace(body_node.text(separator=" ")), self.settings
        )

        updated_at = None
        modified = MODIFIED_RE.search(tree.body.text() if tree.body else "")
        if modified:
            updated_at = collapse_whitespace(modified.group(1))

        return Article(
            article_id=article_id,
            title=title,
            url=url,
            category=category,
            folder=folder,
            folder_id=folder_id,
            body_text=body_text,
            updated_at=updated_at,
            attachments=self._parse_attachments(tree),
            crawler_version=self.version,
        )

    def get_article(self, article_id: str) -> Article:
        """Fetch and parse one article by ID.

        Args:
            article_id: Freshdesk article ID.

        Returns:
            The parsed article. Attachment bytes are not downloaded here; that
            is :mod:`seeley_rag.acquire.attachments`.

        Raises:
            AcquisitionError: If the article cannot be fetched or parsed.
        """
        path = f"{self.config.solutions_path}/articles/{article_id}"
        url = self.absolute_url(path)
        return self.parse_article(self.get(path), article_id=article_id, url=url)

    def get_article_from_link(self, link: dict[str, str], folder: dict[str, str]) -> Article:
        """Fetch an article using listing context already in hand.

        Preferred over :meth:`get_article` during a folder walk: it keeps the
        article's canonical slug URL for citation, and carries the folder's
        category down rather than re-deriving it.

        Args:
            link: An entry from :meth:`list_articles`.
            folder: An entry from :meth:`list_folders`.

        Returns:
            The parsed article.
        """
        return self.parse_article(
            self.get(link["url"]),
            article_id=link["id"],
            url=self.absolute_url(link["url"]),
            category=folder.get("category", ""),
            folder=folder.get("name", ""),
            folder_id=folder.get("id", ""),
        )

    # -- convenience -------------------------------------------------------

    def iter_articles(
        self,
        categories: list[str] | None = None,
        limit: int | None = None,
        skip_article_ids: set[str] | None = None,
    ) -> Iterator[Article]:
        """Walk folders and yield parsed articles.

        Args:
            categories: Category names or leading fragments, matched
                case-insensitively by :meth:`select_folders`. ``None`` walks
                every category.
            limit: Stop after this many **newly parsed** articles. Articles
                skipped via ``skip_article_ids`` do not count toward it, so
                ``--limit 10`` on a resumed run means ten more, not ten total.
            skip_article_ids: Articles already acquired. These are skipped from
                the folder listing without fetching their pages at all, which is
                what makes resuming cheap rather than merely idempotent.

        Yields:
            Parsed articles, in folder order.
        """
        folders = self.select_folders(categories)
        already_done = skip_article_ids or set()
        count = 0
        for folder in folders:
            for link in self.list_articles(folder["id"]):
                if link["id"] in already_done:
                    self.articles_skipped += 1
                    continue
                if limit is not None and count >= limit:
                    return
                yield self.get_article_from_link(link, folder)
                self.articles_parsed += 1
                count += 1

    def select_folders(self, categories: list[str] | None = None) -> list[dict[str, str]]:
        """Return the folders whose category matches any of ``categories``.

        Matching is case-insensitive, and per needle it prefers a **prefix**
        match, falling back to a substring match only when nothing matches by
        prefix.

        Both halves are needed because of how the portal names categories.
        Prefix matching is what separates ``Reverse Cycle Service and
        Installation`` from ``VRF Reverse Cycle Service and Installation`` -- a
        separate commercial product line that pure substring matching cannot
        exclude, since the former's full name is contained in the latter's.
        Selecting it by accident would add ~123 articles to a pilot that is
        meant to be ~326. The substring fallback keeps short mid-name needles
        like ``Evaporative`` working against
        ``COMMERCIAL COOLING - BRAEMAR DIRECT EVAPORATIVE ...``.

        Args:
            categories: Category names or leading fragments. ``None`` or empty
                returns every folder.

        Returns:
            The matching folders, in listing order and without duplicates.
        """
        folders = self.list_folders()
        if not categories:
            return folders

        selected_ids: set[str] = set()
        for needle in (c.lower() for c in categories):
            matches = [f for f in folders if f["category"].lower().startswith(needle)]
            if not matches:
                matches = [f for f in folders if needle in f["category"].lower()]
            selected_ids.update(f["id"] for f in matches)

        selected = [f for f in folders if f["id"] in selected_ids]
        log.info(
            "selected folders by category",
            extra={"requested": categories, "matched": len(selected), "total": len(folders)},
        )
        return selected
