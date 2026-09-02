"""Tests for the diagnostic-article content stream.

build-plan.md section 4.4. Version 1 of the plan gave these articles a retrieval
boost and then never ingested them, so the boost would have applied to zero rows.
These tests are what keep that from happening again.
"""

from __future__ import annotations

from seeley_rag.acquire.base import Article, Attachment
from seeley_rag.parse.html import article_to_markdown, article_to_page, ingest_articles


def article(
    article_id: str = "47001137472",
    title: str = "TQ Gas Heater FC7 Ignition Failure",
    body: str = "What FC7 means. The heater attempted ignition but no flame was detected.",
    attachments: list[Attachment] | None = None,
    folder: str = "Diagnostics and Specific Fault Finding",
) -> Article:
    """Build a manifest article.

    Args:
        article_id: Freshdesk article ID.
        title: Article title.
        body: Body text, already boilerplate-stripped by Stage 1.
        attachments: Attachments to record.
        folder: Folder name.

    Returns:
        An :class:`Article`.
    """
    return Article(
        article_id=article_id,
        title=title,
        url=f"https://x/support/solutions/articles/{article_id}-slug",
        category="DUCTED GAS HEATING (DGH) SERVICE AND INSTALLATION",
        folder=folder,
        folder_id="47000225980",
        body_text=body,
        attachments=attachments or [],
    )


class TestArticleToMarkdown:
    """Body normalisation."""

    def test_title_becomes_a_heading(self) -> None:
        """The chunk text must name the product even when the body does not."""
        rendered = article_to_markdown(article())
        assert rendered.startswith("# TQ Gas Heater FC7 Ignition Failure")

    def test_body_follows_the_heading(self) -> None:
        """The body is preserved, not summarised."""
        assert "no flame was detected" in article_to_markdown(article())

    def test_bullet_glyphs_become_list_items(self) -> None:
        """The portal uses several bullet glyphs; markdown wants one."""
        rendered = article_to_markdown(article(body="• first\n• second"))
        assert "- first" in rendered
        assert "- second" in rendered

    def test_blank_line_runs_are_collapsed(self) -> None:
        """The portal's markup leaves long runs of empty paragraphs."""
        rendered = article_to_markdown(article(body="one\n\n\n\n\ntwo"))
        assert "\n\n\n" not in rendered

    def test_empty_body_still_yields_the_title(self) -> None:
        """An article stripped to nothing is at least identifiable."""
        assert article_to_markdown(article(body="")) == "# TQ Gas Heater FC7 Ignition Failure"


class TestArticleToPage:
    """Synthetic page records for articles."""

    def test_doc_id_is_namespaced(self) -> None:
        """Article IDs must not collide with document hashes."""
        assert article_to_page(article()).doc_id == "article:47001137472"

    def test_no_page_index_or_image(self) -> None:
        """An article has no printed page and no rendered image.

        Inventing either would produce a citation an installer cannot verify.
        """
        page = article_to_page(article())
        assert page.page_index is None
        assert page.page_label is None
        assert page.image_path is None
        assert page.label_source == "none"

    def test_content_stream_marks_it_for_the_boost(self) -> None:
        """Retrieval multiplies this stream's fused score by 1.2."""
        assert article_to_page(article()).content_stream == "diagnostic_article"

    def test_citation_targets_the_article_url(self) -> None:
        """These chunks cite the article directly, not an attachment."""
        page = article_to_page(article())
        assert page.article_url == page.source_url
        assert "47001137472" in page.source_url

    def test_metadata_is_resolved(self) -> None:
        """Product routing applies to this stream too."""
        page = article_to_page(article())
        assert page.product_family == "DGH"
        assert page.doc_type == "fault_finding"
        assert "TQ" in page.model_series

    def test_never_needs_vision(self) -> None:
        """There is no image to transcribe."""
        assert article_to_page(article()).needs_vision is False


class TestIngestArticles:
    """The stub/content split."""

    def test_stubs_are_dropped(self) -> None:
        """A "Pdf attached" chunk pollutes retrieval.

        The stub's title and metadata still decorate the PDF's chunks; what is
        dropped is only its useless body.
        """
        stub = article(
            article_id="1",
            title="TQ Service Guide 644066 M",
            body="Pdf attached,",
            attachments=[Attachment(attachment_id="a1", filename="m.pdf", url="https://x/1")],
        )
        assert stub.is_stub is True
        assert ingest_articles([stub]) == []

    def test_content_articles_are_kept(self) -> None:
        """The high-value stream is what this module exists to ingest."""
        pages = ingest_articles([article(body="x" * 500)])
        assert len(pages) == 1
        assert pages[0].content_stream == "diagnostic_article"

    def test_mixed_input_splits_correctly(self) -> None:
        """The real corpus is mostly stubs with a valuable minority."""
        stub = article(
            article_id="1",
            body="Pdf attached,",
            attachments=[Attachment(attachment_id="a1", filename="m.pdf", url="https://x/1")],
        )
        content = article(article_id="2", body="Real diagnostic content. " * 20)
        pages = ingest_articles([stub, content])
        assert [p.doc_id for p in pages] == ["article:2"]

    def test_empty_bodied_article_is_dropped(self) -> None:
        """An article with nothing left after stripping has nothing to retrieve."""
        assert ingest_articles([article(body="   ")]) == []

    def test_short_article_without_an_attachment_is_kept(self) -> None:
        """Both halves of the stub rule matter.

        A thin article with no attachment is not a pointer -- there is no PDF for
        its metadata to decorate, so dropping it would discard it entirely.
        """
        thin = article(body="Check the flame sensor.")
        assert thin.is_stub is False
        assert len(ingest_articles([thin])) == 1
