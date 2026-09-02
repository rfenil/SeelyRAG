"""Manifest read, write, validate and summarise.

The manifest -- ``data/00_raw/manifest.jsonl``, one JSON object per line -- is
the index of everything acquisition produced. Every downstream stage starts by
reading it, and every citation the finished system emits traces back through it
to a specific fetch of a specific URL at a specific time.

JSONL rather than a single JSON array so that a long crawl can append as it goes
and a crash leaves a readable file rather than a truncated one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from pydantic import ValidationError

from seeley_rag.acquire.base import Article, Attachment
from seeley_rag.exceptions import ManifestError
from seeley_rag.logging_conf import get_logger
from seeley_rag.paths import MANIFEST_PATH, RAW_DIR
from seeley_rag.settings import REPO_ROOT

log = get_logger(__name__)


class ManifestWriter:
    """Append-safe JSONL writer.

    Opens in append mode by default so an interrupted crawl can be resumed
    without discarding what it already fetched. Pass ``overwrite=True`` for a
    clean run.

    Args:
        path: Manifest location. Defaults to ``data/00_raw/manifest.jsonl``.
        overwrite: Truncate an existing manifest instead of appending.

    Attributes:
        count: Rows written by this writer.
    """

    def __init__(self, path: Path | None = None, overwrite: bool = False) -> None:
        self.path = path or MANIFEST_PATH
        self.overwrite = overwrite
        self.count = 0
        self._handle: object = None

    def __enter__(self) -> ManifestWriter:
        """Open the manifest for writing."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if self.overwrite else "a"
        self._handle = self.path.open(mode, encoding="utf-8", newline="\n")
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the manifest."""
        if self._handle is not None:
            self._handle.close()  # type: ignore[attr-defined]
            self._handle = None

    def write(self, article: Article) -> None:
        """Append one article as a manifest row.

        Args:
            article: The article to record.

        Raises:
            ManifestError: If the writer is not open.
        """
        if self._handle is None:
            raise ManifestError("ManifestWriter must be used as a context manager.")
        row = json.dumps(article.to_manifest_row(), ensure_ascii=False)
        self._handle.write(row + "\n")  # type: ignore[attr-defined]
        # Flush every row. A crawl is a long unattended run that may be killed
        # at any moment; buffered rows would be lost and re-fetched, which is
        # the exact waste resume exists to avoid.
        self._handle.flush()  # type: ignore[attr-defined]
        self.count += 1

    def write_all(self, articles: list[Article]) -> int:
        """Append several articles.

        Args:
            articles: Articles to record.

        Returns:
            The number of rows written.
        """
        for article in articles:
            self.write(article)
        return len(articles)


def read_manifest(path: Path | None = None) -> Iterator[Article]:
    """Stream the manifest, one :class:`Article` per line.

    Streaming rather than loading: the manifest is small today, but every
    downstream stage reading it lazily costs nothing and scales to the full
    corpus without revisiting this function.

    Args:
        path: Manifest location. Defaults to ``data/00_raw/manifest.jsonl``.

    Yields:
        Each article in file order.

    Raises:
        ManifestError: If the file is missing, or any line is not a valid row.
    """
    resolved = path or MANIFEST_PATH
    if not resolved.exists():
        raise ManifestError(
            f"No manifest at {resolved}. Run `make acquire` (or "
            "`python scripts/02_acquire.py`) first."
        )
    with resolved.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield Article.model_validate_json(stripped)
            except ValidationError as exc:
                raise ManifestError(f"{resolved}:{number} is not a valid manifest row: {exc}")
            except json.JSONDecodeError as exc:
                raise ManifestError(f"{resolved}:{number} is not valid JSON: {exc}") from exc


@dataclass
class CrawlProgress:
    """What a previous crawl already completed, for resuming after a failure.

    Attributes:
        article_ids: Articles already written to the manifest. These are skipped
            without fetching their pages at all.
        attachments: ``attachment_id -> Attachment`` for every attachment already
            downloaded and still present on disk. This is what makes resume
            cheap: without it, a re-run has to stream every PDF back over the
            network before it can hash it and discover it already had it.
        malformed_lines: Rows that could not be parsed, usually a half-written
            final line from a process killed mid-write.
        rows: Total valid rows read.
    """

    article_ids: set[str] = field(default_factory=set)
    attachments: dict[str, Attachment] = field(default_factory=dict)
    malformed_lines: int = 0
    rows: int = 0

    @property
    def is_empty(self) -> bool:
        """Whether there is nothing to resume from."""
        return self.rows == 0

    @property
    def needs_compaction(self) -> bool:
        """Whether the manifest contains unparseable rows that should be dropped."""
        return self.malformed_lines > 0


def load_progress(path: Path | None = None) -> CrawlProgress:
    """Read an existing manifest to work out what a previous run finished.

    Deliberately tolerant, unlike :func:`read_manifest`. A crawl killed
    mid-write leaves a truncated final line, and refusing to resume because of
    it would force a full re-crawl — precisely the outcome resume exists to
    prevent. Malformed rows are counted and skipped; the articles they described
    are simply re-fetched, which is self-healing.

    An attachment is only treated as done if its recorded file still exists, so
    deleting something from ``data/00_raw/pdf/`` is enough to make the next run
    fetch it again.

    Args:
        path: Manifest location. Defaults to ``data/00_raw/manifest.jsonl``.

    Returns:
        A :class:`CrawlProgress`. An absent manifest yields an empty one rather
        than an error — the first run has nothing to resume.
    """
    resolved = path or MANIFEST_PATH
    progress = CrawlProgress()
    if not resolved.exists():
        return progress

    with resolved.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                article = Article.model_validate_json(stripped)
            except (ValidationError, json.JSONDecodeError):
                progress.malformed_lines += 1
                continue
            progress.rows += 1
            progress.article_ids.add(article.article_id)
            for attachment in article.attachments:
                if not attachment.is_downloaded or attachment.stored_path is None:
                    continue
                if (REPO_ROOT / attachment.stored_path).exists() or Path(
                    attachment.stored_path
                ).exists():
                    progress.attachments[attachment.attachment_id] = attachment

    if progress.malformed_lines:
        log.warning(
            "manifest contains unparseable rows; they will be dropped on compaction",
            extra={"path": str(resolved), "malformed": progress.malformed_lines},
        )
    log.info(
        "loaded crawl progress",
        extra={
            "articles_done": len(progress.article_ids),
            "attachments_done": len(progress.attachments),
        },
    )
    return progress


def compact(path: Path | None = None) -> int:
    """Rewrite the manifest keeping only valid, unique rows.

    Used when resuming from a manifest whose final line was truncated by a kill.
    Written to a sibling temp file and moved into place, so an interruption
    during compaction cannot leave a worse manifest than it started with.

    Args:
        path: Manifest location.

    Returns:
        Number of rows dropped.
    """
    resolved = path or MANIFEST_PATH
    if not resolved.exists():
        return 0

    kept: list[str] = []
    seen: set[str] = set()
    dropped = 0
    with resolved.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                article = Article.model_validate_json(stripped)
            except (ValidationError, json.JSONDecodeError):
                dropped += 1
                continue
            if article.article_id in seen:
                dropped += 1
                continue
            seen.add(article.article_id)
            kept.append(stripped)

    if dropped:
        temp = resolved.with_suffix(".jsonl.compacting")
        temp.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
        temp.replace(resolved)
        log.info("compacted manifest", extra={"kept": len(kept), "dropped": dropped})
    return dropped


def load_manifest(path: Path | None = None) -> list[Article]:
    """Read the whole manifest into memory.

    Args:
        path: Manifest location.

    Returns:
        Every article in the manifest.

    Raises:
        ManifestError: If the file is missing or malformed.
    """
    return list(read_manifest(path))


def validate(path: Path | None = None) -> list[str]:
    """Check that the manifest is internally consistent and complete on disk.

    Verifies, per row: that the required fields carry values; that the computed
    classification fields are present; and that every attachment claiming a
    ``stored_path`` actually has a file there whose name matches its hash.

    That last check is the one that matters. A manifest row pointing at a
    missing PDF is how a downstream stage silently indexes 400 of 600 manuals
    and reports success.

    Args:
        path: Manifest location.

    Returns:
        Human-readable problem descriptions. Empty means the manifest is sound.

    Raises:
        ManifestError: If the file is missing or malformed.
    """
    resolved = path or MANIFEST_PATH
    problems: list[str] = []
    seen_ids: set[str] = set()

    for index, article in enumerate(read_manifest(resolved), start=1):
        where = f"row {index} (article {article.article_id})"

        for name in ("article_id", "title", "url", "fetched_at", "crawler_version"):
            if not getattr(article, name, None):
                problems.append(f"{where}: missing required field '{name}'")
        if not article.category:
            problems.append(f"{where}: empty category; product routing will be unreliable")
        if article.article_id in seen_ids:
            problems.append(f"{where}: duplicate article_id")
        seen_ids.add(article.article_id)

        for attachment in article.attachments:
            if attachment.stored_path is None:
                problems.append(
                    f"{where}: attachment {attachment.attachment_id} "
                    f"({attachment.filename}) was never downloaded"
                )
                continue
            if attachment.sha256 is None:
                problems.append(
                    f"{where}: attachment {attachment.attachment_id} has a stored_path "
                    "but no sha256"
                )
                continue
            stored = REPO_ROOT / attachment.stored_path
            if not stored.exists():
                problems.append(
                    f"{where}: attachment {attachment.attachment_id} points at "
                    f"{attachment.stored_path}, which does not exist"
                )
            elif stored.stem != attachment.sha256:
                problems.append(
                    f"{where}: attachment {attachment.attachment_id} stored at "
                    f"{attachment.stored_path} does not match its sha256 "
                    f"{attachment.sha256}; content addressing is broken"
                )

    return problems


@dataclass
class Document:
    """One unique PDF, plus every article that points at it.

    build-plan.md section 3.3: "Keep ``doc_id -> [article_ids]`` so a chunk from
    a shared manual cites whichever article the user arrived from."

    This is the deduplicated view of the corpus, and it is what Stage 2 must
    iterate. Iterating articles instead would parse the *April 2005 Braemar
    Heaters Service Guide* five times -- five table-detection passes and, since
    2005-era manuals are the ones most likely to be scanned, five full vision
    transcriptions of the same pages.

    Attributes:
        sha256: Content hash. This is the document's identity and its ``doc_id``.
        stored_path: Repo-relative path to the single stored copy.
        size_bytes: Size of that copy.
        filenames: Every filename the portal served these bytes under. Usually
            one, but a shared manual is occasionally renamed per article.
        attachment_ids: Every Freshdesk attachment ID resolving to these bytes.
        article_ids: Every article that links it, in manifest order.
        titles: Those articles' titles, in the same order.
        categories: Distinct categories the linking articles belong to.
        folders: Distinct folders the linking articles belong to.
    """

    sha256: str
    stored_path: str
    size_bytes: int = 0
    filenames: list[str] = field(default_factory=list)
    attachment_ids: list[str] = field(default_factory=list)
    article_ids: list[str] = field(default_factory=list)
    titles: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    folders: list[str] = field(default_factory=list)

    @property
    def reference_count(self) -> int:
        """How many articles link this document."""
        return len(self.article_ids)

    @property
    def is_shared(self) -> bool:
        """Whether more than one article links it."""
        return self.reference_count > 1

    @property
    def primary_filename(self) -> str:
        """The filename to show in a citation."""
        return self.filenames[0] if self.filenames else ""

    @property
    def wasted_bytes_avoided(self) -> int:
        """Bytes that naive per-article storage would have duplicated."""
        return (self.reference_count - 1) * self.size_bytes


def unique_documents(path: Path | None = None) -> list[Document]:
    """Return the deduplicated document set, each with its linking articles.

    The work-list for every stage after acquisition. Parse, render, embed and
    index all operate per *document*; only citation resolution needs the
    per-article view, and that is what ``article_ids`` carries.

    Attachments recorded in the manifest but never successfully downloaded are
    omitted, since there are no bytes to process.

    Args:
        path: Manifest location.

    Returns:
        Documents in first-seen order.

    Raises:
        ManifestError: If the manifest is missing or malformed.
    """
    documents: dict[str, Document] = {}
    for article in read_manifest(path):
        for attachment in article.attachments:
            if attachment.sha256 is None or attachment.stored_path is None:
                continue
            document = documents.get(attachment.sha256)
            if document is None:
                document = Document(
                    sha256=attachment.sha256,
                    stored_path=attachment.stored_path,
                    size_bytes=attachment.size_bytes or 0,
                )
                documents[attachment.sha256] = document
            if attachment.filename not in document.filenames:
                document.filenames.append(attachment.filename)
            if attachment.attachment_id not in document.attachment_ids:
                document.attachment_ids.append(attachment.attachment_id)
            if article.article_id not in document.article_ids:
                document.article_ids.append(article.article_id)
                document.titles.append(article.title)
            if article.category and article.category not in document.categories:
                document.categories.append(article.category)
            if article.folder and article.folder not in document.folders:
                document.folders.append(article.folder)
    return list(documents.values())


class ManifestSummary(dict):
    """Counters describing a manifest. A dict, so it serialises without fuss."""

    def render(self) -> str:
        """Render the summary as human-readable lines.

        Returns:
            A multi-line string suitable for printing at the end of a crawl.
        """
        lines = [
            "Manifest summary",
            "----------------",
            f"  articles                 {self['articles']}",
            f"    stubs (-> pdf)         {self['stub_articles']}",
            f"    content articles       {self['content_articles']}",
            f"  attachments referenced   {self['attachments']}",
            f"    downloaded             {self['attachments_downloaded']}",
            f"    unique documents       {self['unique_documents']}",
            f"    duplicates             {self['duplicate_attachments']}",
            f"    missing / failed       {self['attachments_missing']}",
            f"  bytes stored             {self['bytes_stored']:,}",
            f"  categories               {self['categories']}",
            f"  folders                  {self['folders']}",
        ]
        if self["attachments"]:
            lines.append(f"  duplication rate         {self['duplication_rate']:.1%}")
        return "\n".join(lines)


def summarise(path: Path | None = None) -> ManifestSummary:
    """Summarise a manifest.

    Args:
        path: Manifest location.

    Returns:
        A :class:`ManifestSummary` with article, attachment, deduplication and
        byte counts, plus the stub versus content-article split.

    Raises:
        ManifestError: If the file is missing or malformed.
    """
    articles = load_manifest(path)

    unique_hashes: set[str] = set()
    attachments = downloaded = duplicates = missing = 0
    bytes_stored = 0
    categories: set[str] = set()
    folders: set[str] = set()

    for article in articles:
        categories.add(article.category)
        folders.add(article.folder)
        for attachment in article.attachments:
            attachments += 1
            if attachment.stored_path is None or attachment.sha256 is None:
                missing += 1
                continue
            downloaded += 1
            if attachment.sha256 in unique_hashes:
                duplicates += 1
            else:
                unique_hashes.add(attachment.sha256)
                # Count bytes once per unique document, not once per reference.
                bytes_stored += attachment.size_bytes or 0

    stubs = sum(1 for a in articles if a.is_stub)
    return ManifestSummary(
        articles=len(articles),
        stub_articles=stubs,
        content_articles=len(articles) - stubs,
        attachments=attachments,
        attachments_downloaded=downloaded,
        attachments_missing=missing,
        unique_documents=len(unique_hashes),
        duplicate_attachments=duplicates,
        duplication_rate=(duplicates / attachments) if attachments else 0.0,
        bytes_stored=bytes_stored,
        categories=len(categories),
        folders=len(folders),
    )


def manifest_exists(path: Path | None = None) -> bool:
    """Whether a manifest is present.

    Args:
        path: Manifest location.

    Returns:
        True if the file exists.
    """
    return (path or MANIFEST_PATH).exists()


def raw_dir_is_writable() -> bool:
    """Whether the write-once raw directory exists and can be created.

    Returns:
        True once ``data/00_raw`` is present.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    return RAW_DIR.is_dir()
