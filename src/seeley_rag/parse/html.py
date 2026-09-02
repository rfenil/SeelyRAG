"""Stage 2b -- diagnostic-article ingestion.

build-plan.md section 4.4.

The diagnostic articles are the corpus's second content stream: short, Q&A
shaped, written in installer language, and disproportionately valuable per byte.
Version 1 of the build plan named them as high-value, gave them a retrieval
boost, and then never ingested them -- so the boost would have applied to zero
rows. This module is what stops that happening.

Stage 1 already did the hard part. It stripped the shared safety boilerplate and
set ``is_stub`` and ``content_stream`` on every manifest row, so the split here
is a filter rather than a judgement:

* ``is_stub`` -- a card-catalogue pointer. **Not indexed as content.** A "Pdf
  attached" chunk pollutes retrieval. Its title and metadata still decorate the
  chunks of the PDF it points at, which happens in :mod:`seeley_rag.parse.pdf`
  via the document's ``titles`` and ``folders``.
* otherwise -- real content. Converted to markdown and emitted as a synthetic
  single-page record with ``page_index=None`` and ``image_path=None``, citing
  the article URL directly.

Emitting the same :class:`~seeley_rag.parse.base.Page` schema as the PDF stream
means Stage 3 chunks both with one code path, while ``content_stream`` stays
available for the retrieval boost.
"""

from __future__ import annotations

import re

from seeley_rag.acquire.base import Article
from seeley_rag.logging_conf import get_logger
from seeley_rag.parse.base import (
    Page,
    resolve_doc_type,
    resolve_model_series,
    resolve_product_family,
)

log = get_logger(__name__)

#: Bullet glyphs the portal uses, normalised to markdown list items.
_BULLETS = re.compile(r"^\s*[•▪◦·]\s*", re.MULTILINE)

#: Runs of blank lines, collapsed to a single paragraph break.
_BLANK_RUN = re.compile(r"\n{3,}")

#: A heading-ish line: short, no terminal punctuation, title-like.
_HEADING_MAX_CHARS = 60


def article_to_markdown(article: Article) -> str:
    """Convert a content article's body to markdown.

    Stage 1 stores ``body_text`` already stripped of navigation and of the shared
    safety notice, so this is light-touch normalisation rather than HTML parsing:
    bullets become list items, run-on whitespace is collapsed, and the title is
    promoted to an H1 so the chunk text names the product even when the body does
    not.

    Args:
        article: A manifest article whose ``content_stream`` is
            ``diagnostic_article``.

    Returns:
        The body as markdown, title heading included.
    """
    body = article.body_text.strip()
    body = _BULLETS.sub("- ", body)
    body = _BLANK_RUN.sub("\n\n", body)
    if not body:
        return f"# {article.title}".strip()
    return f"# {article.title}\n\n{body}"


def article_to_page(article: Article) -> Page:
    """Convert one content article into a synthetic single-page record.

    Args:
        article: A manifest article.

    Returns:
        A :class:`Page` with ``page_index`` and ``image_path`` unset -- an
        article has no printed page and no rendered image, and pretending
        otherwise would produce a citation that cannot be verified.
    """
    return Page(
        doc_id=f"article:{article.article_id}",
        page_index=None,
        page_label=None,
        label_source="none",
        text=article_to_markdown(article),
        tables=[],
        tier="plain_text",
        needs_vision=False,
        image_path=None,
        source_article_ids=[article.article_id],
        product_family=resolve_product_family(article.category, article.folder, article.title),
        doc_type=resolve_doc_type(article.folder, article.title),
        model_series=resolve_model_series(article.title),
        title=article.title,
        source_url=article.url,
        article_url=article.url,
        category=article.category,
        folder=article.folder,
        content_stream="diagnostic_article",
    )


def ingest_articles(articles: list[Article]) -> list[Page]:
    """Turn content articles into indexable pages, discarding stubs.

    Args:
        articles: Every article from the manifest.

    Returns:
        One page per content article. Stubs are dropped, and articles whose body
        is empty after boilerplate stripping are dropped too -- there is nothing
        left in them to retrieve.
    """
    pages: list[Page] = []
    stubs = 0
    empty = 0
    for article in articles:
        if article.is_stub:
            stubs += 1
            continue
        page = article_to_page(article)
        # An article can survive the stub test on attachment count alone and
        # still have nothing in it once the shared notice is removed.
        if not article.body_text.strip():
            empty += 1
            continue
        pages.append(page)

    log.info(
        "ingested diagnostic articles",
        extra={
            "articles": len(articles),
            "ingested": len(pages),
            "stubs_dropped": stubs,
            "empty_dropped": empty,
        },
    )
    return pages
