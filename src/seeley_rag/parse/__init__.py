"""Stage 2 -- parsing.

build-plan.md section 4. Where the project is won or lost.

Implemented:
    :mod:`triage`      -- Stage 0 corpus triage; sets the vision budget.
    :mod:`base`        -- the ``pages.jsonl`` schema and metadata resolution.
    :mod:`pdf`         -- text, gated table detection, page images.
    :mod:`pagelabels`  -- printed-label reconciliation.
    :mod:`html`        -- the diagnostic-article content stream.

Stubbed:
    :mod:`vision` -- transcription for scanned pages and captions for
    diagram-heavy ones. Pages needing it are recorded with
    ``needs_vision=True``, so it can be added without re-parsing.
"""

from __future__ import annotations

from seeley_rag.parse.base import (
    Page,
    PagesWriter,
    Table,
    parsed_doc_ids,
    read_pages,
    resolve_doc_type,
    resolve_model_series,
    resolve_product_family,
)
from seeley_rag.parse.html import article_to_markdown, ingest_articles
from seeley_rag.parse.pagelabels import detect_offset, label_from_text, resolve_label
from seeley_rag.parse.pdf import extract_tables, has_table_signal, parse_pdf
from seeley_rag.parse.triage import (
    DocumentTriage,
    PageTriage,
    TriageSummary,
    triage_corpus,
    triage_pdf,
)

__all__ = [
    "DocumentTriage",
    "Page",
    "PageTriage",
    "PagesWriter",
    "Table",
    "TriageSummary",
    "article_to_markdown",
    "detect_offset",
    "extract_tables",
    "has_table_signal",
    "ingest_articles",
    "label_from_text",
    "parse_pdf",
    "parsed_doc_ids",
    "read_pages",
    "resolve_doc_type",
    "resolve_label",
    "resolve_model_series",
    "resolve_product_family",
    "triage_corpus",
    "triage_pdf",
]
