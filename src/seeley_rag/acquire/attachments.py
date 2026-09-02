"""Attachment downloader with SHA-256 content addressing and deduplication.

build-plan.md section 3.3.

Two facts drive this module:

* ``/helpdesk/attachments/{id}`` **302s to S3**, so redirects must be followed.
* **The same manual is attached to multiple articles.** Expect 15-30%
  duplication. Parsing a 200-page manual four times is the easiest hour to waste
  in this project, so deduplication is a correctness requirement rather than an
  optimisation.

Files are stored at ``data/00_raw/pdf/{sha256}.pdf``. Content addressing makes
deduplication fall out for free and makes re-downloads idempotent: if the path
exists, the bytes are already correct, because the path *is* the hash.

Download is streamed to a temporary file first, then hashed, then moved into
place. That ordering matters -- the hash is unknown until the bytes are read, and
a partial download must never be able to occupy a content-addressed path where a
later run would trust it.
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from pathlib import Path

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from seeley_rag.acquire.base import Attachment
from seeley_rag.acquire.throttle import RateLimiter
from seeley_rag.exceptions import AcquisitionError, RateLimitedError
from seeley_rag.logging_conf import get_logger
from seeley_rag.paths import RAW_PDF_DIR, raw_blob_path, relative_to_root
from seeley_rag.settings import Settings, get_settings

log = get_logger(__name__)

#: Read size when streaming a download. Manuals run to a few MB.
CHUNK_BYTES = 64 * 1024


#: Magic number -> file suffix. The corpus is overwhelmingly PDFs, but a few
#: attachments are images or Office documents, and storing those as ``.pdf``
#: makes every later stage try to parse them as PDFs.
MAGIC_SUFFIXES: tuple[tuple[bytes, str], ...] = (
    (b"%PDF", ".pdf"),
    (b"\x89PNG\r\n\x1a\n", ".png"),
    (b"\xff\xd8\xff", ".jpg"),
    (b"GIF8", ".gif"),
    (b"RIFF", ".webp"),
    (b"II*\x00", ".tif"),
    (b"MM\x00*", ".tif"),
    (b"PK\x03\x04", ".zip"),  # docx / xlsx / pptx are all zip containers
    (b"\xd0\xcf\x11\xe0", ".doc"),  # legacy OLE: doc / xls / msg
)

#: Markers that identify an HTML response body. An attachment endpoint that
#: returns HTML is not serving a file.
HTML_MARKERS: tuple[bytes, ...] = (b"<!DOCTYPE", b"<!doctype", b"<html", b"<HTML", b"<?xml")


class _RetryableStatus(Exception):
    """Internal signal: a 5xx worth retrying. Never escapes this module."""


def detect_suffix(head: bytes) -> str | None:
    """Identify a file's type from its leading bytes.

    Args:
        head: The first few hundred bytes of the file.

    Returns:
        A suffix including the leading dot, or ``None`` if unrecognised.
    """
    for magic, suffix in MAGIC_SUFFIXES:
        if head.startswith(magic):
            return suffix
    return None


def looks_like_html(head: bytes) -> bool:
    """Whether a response body is an HTML page rather than a file.

    Freshdesk serves some attachments only to authenticated users, and does so
    by returning **HTTP 200 with a login page** rather than a 401 or 403. Left
    unchecked, that page is hashed and recorded in the manifest as a manual --
    a silent corruption that no status-code check and no validation would catch,
    and that would put sign-in boilerplate into the retrieval index.

    Args:
        head: The first few hundred bytes of the response body.

    Returns:
        True if the body appears to be HTML.
    """
    sample = head.lstrip()[:256]
    return any(sample.startswith(marker) for marker in HTML_MARKERS)


def sha256_file(path: Path) -> str:
    """Compute the SHA-256 of a file's contents.

    Args:
        path: File to hash.

    Returns:
        Lower-case hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


class AttachmentDownloader:
    """Downloads attachments, deduplicating by content hash.

    A single instance should span one crawl run: it remembers which hashes it
    has seen so it can populate ``is_duplicate_of`` with the attachment ID that
    first carried each set of bytes.

    Args:
        settings: Settings override, for tests.
        client: Pre-configured HTTP client. One is built from settings when
            omitted. Tests inject a mock transport here.
        limiter: Shared rate limiter. Pass the same instance the scraper uses so
            the run as a whole honours 1 rps.

    Attributes:
        downloaded: Attachments whose bytes were fetched over the network.
        deduplicated: Attachments whose bytes were already present, whether from
            an earlier run or earlier in this one.
        failed: ``attachment_id -> error message`` for attachments that could
            not be fetched. A failure never aborts the crawl -- one dead link
            must not cost you the other 899 articles -- but every failure is
            recorded and surfaced in the run summary.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        client: httpx.Client | None = None,
        limiter: RateLimiter | None = None,
        known: dict[str, Attachment] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.config = self.settings.crawl
        # Attachment downloads are requests too. Without this they bypass the
        # crawl's rate limit entirely -- ~600 unthrottled PDF fetches on a full
        # corpus run, on top of the page fetches that ARE throttled.
        self.limiter = limiter or RateLimiter(self.config.rps)
        self._owns_client = client is None
        self.client = client or httpx.Client(
            headers={"User-Agent": self.config.user_agent(self.settings.project.crawler_version)},
            timeout=self.config.timeout_seconds,
            # Attachment URLs 302 to S3. Without this every download returns an
            # empty 302 body and the manifest fills with 0-byte "PDFs".
            follow_redirects=True,
        )
        #: sha256 -> the attachment_id that first produced those bytes.
        self._hash_owner: dict[str, str] = {}
        #: attachment_id -> Attachment already downloaded by an earlier run.
        #: Content addressing alone cannot skip a re-download, because the hash
        #: is only known after the bytes have been streamed. This map is what
        #: lets a resumed run skip the transfer entirely.
        self.known = dict(known or {})
        for attachment in self.known.values():
            if attachment.sha256 and attachment.sha256 not in self._hash_owner:
                self._hash_owner[attachment.sha256] = attachment.attachment_id
        self.downloaded: list[Attachment] = []
        self.deduplicated: list[Attachment] = []
        self.skipped: list[Attachment] = []
        self.failed: dict[str, str] = {}
        #: Every byte pulled over the wire, duplicates included. This is the
        #: honest measure of what the crawl cost Seeley's server; bytes_stored
        #: is what it cost our disk, and the gap between them is the dedupe win.
        self.bytes_fetched = 0

    def close(self) -> None:
        """Close the HTTP client, if this downloader created it."""
        if self._owns_client:
            self.client.close()

    def __enter__(self) -> AttachmentDownloader:
        """Enter the context manager."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the client on exit."""
        self.close()

    def _stream_to_temp(self, url: str) -> tuple[Path, int]:
        """Stream a URL to a temporary file.

        Args:
            url: Absolute attachment URL.

        Returns:
            The temporary file path and its size in bytes.

        Raises:
            RateLimitedError: On 429 or 403. Never retried.
            AcquisitionError: On any other non-2xx, or once retries are spent.
        """

        @retry(
            retry=retry_if_exception_type((_RetryableStatus, httpx.TimeoutException)),
            stop=stop_after_attempt(self.config.max_retries),
            wait=wait_exponential(multiplier=self.config.retry_backoff_seconds),
            reraise=True,
        )
        def _attempt() -> tuple[Path, int]:
            # The part-file is created inside the destination directory, not the
            # OS temp dir, so the later move is a same-filesystem rename: atomic,
            # and free rather than a multi-megabyte copy. The data tree and the
            # system temp dir are routinely on different drives.
            RAW_PDF_DIR.mkdir(parents=True, exist_ok=True)
            self.limiter.wait()
            handle = tempfile.NamedTemporaryFile(delete=False, suffix=".part", dir=str(RAW_PDF_DIR))
            temp_path = Path(handle.name)
            size = 0
            try:
                with self.client.stream("GET", url) as response:
                    status = response.status_code
                    if status in (403, 429):
                        raise RateLimitedError(
                            f"GET {url} returned HTTP {status}. Halting rather than "
                            "retrying into a block; escalate to a human."
                        )
                    if 500 <= status < 600:
                        raise _RetryableStatus(f"HTTP {status}")
                    if not 200 <= status < 300:
                        raise AcquisitionError(f"GET {url} returned HTTP {status}.")
                    for block in response.iter_bytes(CHUNK_BYTES):
                        handle.write(block)
                        size += len(block)
            except BaseException:
                handle.close()
                temp_path.unlink(missing_ok=True)
                raise
            handle.close()
            return temp_path, size

        try:
            return _attempt()
        except _RetryableStatus as exc:
            raise AcquisitionError(f"GET {url} failed after retries: {exc}") from exc
        except httpx.TimeoutException as exc:
            raise AcquisitionError(f"GET {url} timed out after retries.") from exc
        except httpx.HTTPError as exc:
            raise AcquisitionError(f"GET {url} failed: {exc}") from exc

    def download(self, attachment: Attachment) -> Attachment:
        """Download one attachment, deduplicating by content hash.

        Args:
            attachment: Attachment metadata parsed from an article page.

        Returns:
            A new :class:`Attachment` with ``sha256``, ``size_bytes``,
            ``stored_path`` and, when the bytes were already known,
            ``is_duplicate_of`` populated. The input is not mutated.

        Raises:
            RateLimitedError: If the portal blocks us. Callers must not swallow
                this -- it means stop the whole run.
            AcquisitionError: On any other download failure.
        """
        previous = self.known.get(attachment.attachment_id)
        if previous is not None:
            # A previous run already fetched these bytes and the file is still
            # on disk (load_progress checked). Skipping here is the whole point
            # of resume: no transfer, no duplicate row, no wasted politeness
            # budget against Seeley's server.
            self.skipped.append(previous)
            log.info(
                "attachment already acquired; skipping download",
                extra={"attachment_id": attachment.attachment_id, "sha256": previous.sha256},
            )
            return previous

        RAW_PDF_DIR.mkdir(parents=True, exist_ok=True)
        temp_path, size = self._stream_to_temp(attachment.url)
        self.bytes_fetched += size
        try:
            with temp_path.open("rb") as handle:
                head = handle.read(512)

            if looks_like_html(head):
                # HTTP 200 carrying a login or error page, not a file. Treated as
                # a failed download so it shows up in the run summary instead of
                # entering the corpus as a manual.
                raise AcquisitionError(
                    f"{attachment.url} returned an HTML page, not a file "
                    f"({size} bytes). This attachment is almost certainly "
                    "restricted to signed-in users; the portal answers with a "
                    "login page rather than a 401. Recorded as a failed "
                    "download and NOT stored."
                )

            suffix = detect_suffix(head)
            if suffix is None:
                log.warning(
                    "unrecognised attachment type; storing as .bin",
                    extra={"attachment_id": attachment.attachment_id, "head": head[:8].hex()},
                )
                suffix = ".bin"

            digest = sha256_file(temp_path)
            destination = raw_blob_path(digest, suffix)
            first_owner = self._hash_owner.get(digest)
            already_on_disk = destination.exists()

            if already_on_disk:
                # data/00_raw is write-once: the path IS the hash, so whatever is
                # there is byte-identical to what we just fetched. Never rewrite.
                temp_path.unlink(missing_ok=True)
            else:
                shutil.move(str(temp_path), str(destination))

            result = attachment.model_copy(
                update={
                    "sha256": digest,
                    "size_bytes": size,
                    "stored_path": relative_to_root(destination),
                    "is_duplicate_of": first_owner,
                }
            )

            if first_owner is None:
                self._hash_owner[digest] = attachment.attachment_id
            # Remember it so the same attachment ID encountered again -- another
            # article linking the same file -- skips rather than re-downloads.
            self.known[attachment.attachment_id] = result

            if first_owner is not None or already_on_disk:
                self.deduplicated.append(result)
                log.info(
                    "attachment deduplicated",
                    extra={
                        "attachment_id": attachment.attachment_id,
                        "sha256": digest,
                        "duplicate_of": first_owner,
                        "already_on_disk": already_on_disk,
                    },
                )
            else:
                self.downloaded.append(result)
                log.info(
                    "attachment stored",
                    extra={
                        "attachment_id": attachment.attachment_id,
                        "sha256": digest,
                        "bytes": size,
                        "path": result.stored_path,
                    },
                )
            return result
        finally:
            temp_path.unlink(missing_ok=True)

    def download_all(self, attachments: list[Attachment]) -> list[Attachment]:
        """Download several attachments, recording rather than raising failures.

        A single dead attachment link must not cost the other 899 articles, so
        :class:`AcquisitionError` is caught and recorded. :class:`RateLimitedError`
        is deliberately **not** caught: being blocked is a whole-run condition.

        Args:
            attachments: Attachments to fetch.

        Returns:
            One entry per input. Failed downloads come back as the original
            metadata with no ``sha256`` or ``stored_path``, so the manifest still
            records that the article had an attachment we could not retrieve.

        Raises:
            RateLimitedError: If the portal blocks us at any point.
        """
        results: list[Attachment] = []
        for attachment in attachments:
            try:
                results.append(self.download(attachment))
            except RateLimitedError:
                raise
            except AcquisitionError as exc:
                self.failed[attachment.attachment_id] = str(exc)
                log.error(
                    "attachment download failed",
                    extra={"attachment_id": attachment.attachment_id, "error": str(exc)},
                )
                results.append(attachment)
        return results

    def summary(self) -> dict[str, int]:
        """Return counters for the run summary.

        Returns:
            Download, dedupe and failure counts, plus two distinct byte figures:
            ``bytes_fetched`` is everything pulled over the wire including
            duplicates, and ``bytes_stored`` counts only unique documents kept
            on disk. ``bytes_saved_by_dedupe`` is the difference -- the download
            and parse work deduplication avoided.
        """
        stored = sum(a.size_bytes or 0 for a in self.downloaded)
        return {
            "downloaded": len(self.downloaded),
            "deduplicated": len(self.deduplicated),
            "skipped_resumed": len(self.skipped),
            "bytes_skipped_by_resume": sum(a.size_bytes or 0 for a in self.skipped),
            "failed": len(self.failed),
            "unique_documents": len(self._hash_owner),
            "bytes_fetched": self.bytes_fetched,
            "bytes_stored": stored,
            "bytes_saved_by_dedupe": max(0, self.bytes_fetched - stored),
        }
