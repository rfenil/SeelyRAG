"""Stage 1 -- acquisition.

build-plan.md section 3. The only stage that talks to the portal, and the only
one permitted to write under ``data/00_raw``.

Public surface:

* :class:`Article` / :class:`Attachment` -- the manifest schema.
* :class:`FreshdeskSource` -- the source interface; :class:`PortalScraper` is
  its only implementation (ADR 0002).
* :class:`RobotsGate` -- the project gate that must pass before any crawl.
* :class:`AttachmentDownloader` -- content-addressed, deduplicating downloader.
* :mod:`manifest` helpers -- write, read, validate, summarise.
"""

from __future__ import annotations

from seeley_rag.acquire.attachments import AttachmentDownloader
from seeley_rag.acquire.base import Article, Attachment, ContentStream, FreshdeskSource
from seeley_rag.acquire.manifest import (
    CrawlProgress,
    Document,
    ManifestWriter,
    load_manifest,
    read_manifest,
    summarise,
    unique_documents,
    validate,
)
from seeley_rag.acquire.portal import PortalScraper
from seeley_rag.acquire.robots import RobotsGate, RobotsReport

__all__ = [
    "Article",
    "Attachment",
    "AttachmentDownloader",
    "ContentStream",
    "FreshdeskSource",
    "CrawlProgress",
    "Document",
    "ManifestWriter",
    "PortalScraper",
    "RobotsGate",
    "RobotsReport",
    "load_manifest",
    "read_manifest",
    "summarise",
    "unique_documents",
    "validate",
]
