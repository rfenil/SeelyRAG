"""Acquisition data models and the source interface.

build-plan.md section 3.1.

:class:`Article` and :class:`Attachment` are the manifest schema. A manifest row
is exactly ``Article.model_dump_json()``; nothing reshapes it on the way to
disk, so the schema here and the file on disk cannot drift apart.

Every artefact carries provenance -- ``fetched_at``, the source ``url``, and
``crawler_version``. When someone later asks "where did this answer come from",
the chain has to resolve to a byte range in a specific fetch, and that only
works if provenance is recorded at acquisition time rather than reconstructed.
"""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

#: How an article's content reaches the index.
#:
#: ``pdf``                 -- the article is a card-catalogue stub; the knowledge
#:                            is in its attachment.
#: ``diagnostic_article``  -- the article body is real content (the fault-finding
#:                            articles); it is ingested directly and gets a
#:                            retrieval boost.
ContentStream = Literal["pdf", "diagnostic_article"]


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string with a ``Z`` suffix.

    Returns:
        e.g. ``2026-08-20T09:14:22Z``.
    """
    return dt.datetime.now(tz=dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Attachment(BaseModel):
    """One file attached to a Freshdesk article -- in practice, a manual PDF.

    Attributes:
        attachment_id: Freshdesk's numeric attachment ID, as a string.
        filename: The filename Freshdesk reports, e.g.
            ``644066-M MANUAL SERVICE TQ SERIES.pdf``.
        url: The ``/helpdesk/attachments/{id}`` URL. It 302s to S3, so any
            fetch of it must follow redirects.
        sha256: Hex SHA-256 of the downloaded bytes. This is the document's
            identity: the same manual is attached across multiple folders, so
            the hash -- not the attachment ID -- is what deduplicates.
        size_bytes: Size of the downloaded file.
        stored_path: Repo-relative content-addressed path,
            ``data/00_raw/pdf/{sha256}.pdf``.
        is_duplicate_of: The ``attachment_id`` this file duplicates, when the
            same hash was already seen. ``None`` for the first occurrence. The
            bytes are stored once; both attachment IDs point at them.
    """

    model_config = ConfigDict(extra="forbid")

    attachment_id: str
    filename: str
    url: str
    sha256: str | None = None
    size_bytes: int | None = None
    stored_path: str | None = None
    is_duplicate_of: str | None = None

    @property
    def is_downloaded(self) -> bool:
        """Whether this attachment has bytes on disk."""
        return self.sha256 is not None and self.stored_path is not None


class Article(BaseModel):
    """One Freshdesk solution article. This is the manifest row schema.

    ``is_stub`` and ``content_stream`` are computed here, at acquisition time,
    so no downstream stage ever re-derives them from a threshold that has since
    drifted (build-plan section 4.4).

    Attributes:
        article_id: Freshdesk's numeric article ID, as a string.
        title: Article title. Carries model codes (``TQ``, ``CQ4``, ``TE4`` ...)
            and is worth keeping even for stubs, since it decorates the chunks
            of the PDF the stub points at.
        url: Absolute article URL, used as the citation target.
        category: Solution category name, e.g. ``Ducted Gas Heating (DGH)``.
        folder: Folder name, e.g. ``Service Guides``. Folder names give
            ``doc_type`` at near-perfect accuracy for zero LLM cost.
        folder_id: Freshdesk folder ID.
        body_text: Article body as plain text, nav and boilerplate stripped.
        updated_at: Last-modified timestamp if the page exposes one. Usually
            ``None``: without an API key there is no reliable ``updated_at``,
            which is why change detection is re-crawl-and-hash (ADR 0002).
        attachments: Files attached to the article.
        fetched_at: When this row was produced. Provenance.
        crawler_version: Which crawler produced it. Provenance.
    """

    model_config = ConfigDict(extra="forbid")

    #: Serialised into every manifest row but derived, never stored. Reading a
    #: row back must therefore drop them rather than reject them -- see
    #: :meth:`_drop_computed_fields`.
    COMPUTED_FIELDS: ClassVar[tuple[str, ...]] = ("body_char_count", "is_stub", "content_stream")

    article_id: str
    title: str
    url: str
    category: str
    folder: str
    folder_id: str
    body_text: str = ""
    updated_at: str | None = None
    attachments: list[Attachment] = Field(default_factory=list)
    fetched_at: str = Field(default_factory=utc_now_iso)
    crawler_version: str = "0.1.0"

    @model_validator(mode="before")
    @classmethod
    def _drop_computed_fields(cls, data: Any) -> Any:
        """Discard computed fields when validating a manifest row back in.

        The classification fields are written into every row so downstream
        stages never re-derive them, but they are derived, not stored. Without
        this, ``extra="forbid"`` would reject every row the writer produced --
        meaning the manifest could be written but never read.

        Dropping them here keeps ``extra="forbid"`` doing its real job: catching
        a misspelled *genuine* field instead of silently ignoring it.

        Args:
            data: Raw input to validation.

        Returns:
            The input with the computed keys removed.
        """
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if k not in cls.COMPUTED_FIELDS}
        return data

    # -- Computed classification ------------------------------------------
    # Serialised into every manifest row. Declared as computed fields so they
    # can never disagree with body_text and attachments.

    @computed_field  # type: ignore[prop-decorator]
    @property
    def body_char_count(self) -> int:
        """Length of the stripped body text."""
        return len(self.body_text.strip())

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_stub(self) -> bool:
        """Whether this article is a card-catalogue entry rather than content.

        True when the body is shorter than the configured threshold **and** the
        article has at least one attachment. Both halves matter: a short body
        with no attachment is a genuinely thin article, not a pointer, and
        indexing a "Pdf attached" sentence as content pollutes retrieval.
        """
        from seeley_rag.settings import get_settings

        threshold = get_settings().articles.stub_max_body_chars
        return self.body_char_count < threshold and len(self.attachments) > 0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def content_stream(self) -> ContentStream:
        """Which ingestion path this article's content takes."""
        return "pdf" if self.is_stub else "diagnostic_article"

    def to_manifest_row(self) -> dict[str, Any]:
        """Render this article as a manifest row.

        Returns:
            A JSON-serialisable dict including the computed classification
            fields, with keys in the documented manifest order.
        """
        return self.model_dump(mode="json")


class FreshdeskSource(ABC):
    """Interface to a source of Seeley solution articles.

    :class:`~seeley_rag.acquire.portal.PortalScraper` is currently the only
    implementation, because ``/api/v2/*`` returns 401 and no key is obtainable
    (ADR 0002). The abstraction is kept anyway: it costs half an hour, and if a
    key ever appears, an ``ApiClient`` implementing these three methods is the
    entire change.

    **That swap must require no downstream change.** Concretely, an
    implementation must:

    * return :class:`Article` instances with every field populated that it can
      populate, and ``None`` -- never a placeholder -- for what it cannot;
    * store attachment bytes content-addressed under ``data/00_raw/pdf/`` and
      set ``sha256``, ``size_bytes`` and ``stored_path`` accordingly;
    * stamp ``fetched_at`` and ``crawler_version`` on every article;
    * leave ``is_stub`` and ``content_stream`` to be computed by the model, not
      set them itself.

    An ``ApiClient`` would additionally be able to populate ``updated_at``
    honestly, which is the one field the scraper cannot, and which would unlock
    incremental sync.
    """

    @abstractmethod
    def list_folders(self) -> list[dict[str, str]]:
        """List every solution folder.

        Returns:
            One dict per folder with at least ``id``, ``name``, ``url`` and
            ``category`` keys.
        """

    @abstractmethod
    def list_articles(self, folder_id: str) -> list[dict[str, str]]:
        """List every article in a folder, following pagination.

        Args:
            folder_id: Freshdesk folder ID.

        Returns:
            One dict per article with at least ``id``, ``title`` and ``url``.
        """

    @abstractmethod
    def get_article(self, article_id: str) -> Article:
        """Fetch and parse a single article.

        Args:
            article_id: Freshdesk article ID.

        Returns:
            The parsed article, attachment metadata included. Whether the
            attachment bytes have been downloaded depends on the implementation.

        Raises:
            AcquisitionError: If the article cannot be fetched or parsed.
        """
