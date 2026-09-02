"""Tests for the acquisition models and source interface."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from seeley_rag.acquire.base import Article, Attachment, FreshdeskSource, utc_now_iso


def make_attachment(**overrides: object) -> Attachment:
    """Build an attachment with sensible defaults.

    Args:
        **overrides: Field overrides.

    Returns:
        An :class:`Attachment`.
    """
    payload: dict[str, object] = {
        "attachment_id": "47234382931",
        "filename": "644066-M MANUAL SERVICE TQ SERIES.pdf",
        "url": "https://x/helpdesk/attachments/47234382931",
    }
    payload.update(overrides)
    return Attachment(**payload)  # type: ignore[arg-type]


def make_article(**overrides: object) -> Article:
    """Build an article with sensible defaults.

    Args:
        **overrides: Field overrides.

    Returns:
        An :class:`Article`.
    """
    payload: dict[str, object] = {
        "article_id": "47001247136",
        "title": "TQ Service Guide Gas Ducted Heater 644066 M",
        "url": "https://x/support/solutions/articles/47001247136-tq",
        "category": "DUCTED GAS HEATING (DGH) SERVICE AND INSTALLATION",
        "folder": "Service Guides",
        "folder_id": "47000783696",
        "body_text": "Pdf attached TQ Service Guide 644066 M",
    }
    payload.update(overrides)
    return Article(**payload)  # type: ignore[arg-type]


class TestAttachment:
    """The attachment model."""

    def test_undownloaded_attachment_has_no_bytes(self) -> None:
        """A freshly parsed attachment carries metadata but no stored bytes."""
        attachment = make_attachment()
        assert attachment.sha256 is None
        assert attachment.stored_path is None
        assert attachment.is_downloaded is False

    def test_downloaded_attachment_reports_itself_as_downloaded(self) -> None:
        """Once hash and path are set, the attachment counts as downloaded."""
        attachment = make_attachment(sha256="a3f1", stored_path="data/00_raw/pdf/a3f1.pdf")
        assert attachment.is_downloaded is True

    def test_unknown_fields_are_rejected(self) -> None:
        """The manifest schema is closed, so typos fail loudly rather than vanish."""
        with pytest.raises(ValidationError):
            Attachment(
                attachment_id="1",
                filename="x.pdf",
                url="https://x",
                shaa256="typo",  # type: ignore[call-arg]
            )


class TestArticleClassification:
    """``is_stub`` and ``content_stream``, computed at acquisition time."""

    def test_short_body_with_attachment_is_a_stub(self) -> None:
        """The representative case: a card-catalogue pointer to a manual."""
        article = make_article(attachments=[make_attachment()])
        assert article.body_char_count == 38
        assert article.is_stub is True
        assert article.content_stream == "pdf"

    def test_short_body_without_attachment_is_not_a_stub(self) -> None:
        """Both halves of the rule matter.

        A short body with nothing attached is a thin article, not a pointer --
        there is no PDF for its metadata to decorate, so treating it as a stub
        would discard it entirely.
        """
        article = make_article(attachments=[])
        assert article.is_stub is False
        assert article.content_stream == "diagnostic_article"

    def test_long_body_with_attachment_is_content(self) -> None:
        """A substantial body is real content even when a PDF is attached."""
        article = make_article(body_text="x" * 500, attachments=[make_attachment()])
        assert article.is_stub is False
        assert article.content_stream == "diagnostic_article"

    def test_body_char_count_ignores_surrounding_whitespace(self) -> None:
        """Whitespace must not push a stub over the threshold."""
        article = make_article(body_text="   abc   ")
        assert article.body_char_count == 3

    @pytest.mark.parametrize(
        ("length", "expected_stub"),
        [(199, True), (200, False), (201, False)],
    )
    def test_threshold_boundary(self, length: int, expected_stub: bool) -> None:
        """The threshold is exclusive: <200 is a stub, 200 is not."""
        article = make_article(body_text="x" * length, attachments=[make_attachment()])
        assert article.is_stub is expected_stub


class TestManifestRow:
    """Serialisation to a manifest row."""

    def test_row_contains_every_documented_field(self) -> None:
        """The manifest schema is a contract with every downstream stage."""
        row = make_article(attachments=[make_attachment()]).to_manifest_row()
        expected = {
            "article_id",
            "title",
            "url",
            "category",
            "folder",
            "folder_id",
            "body_text",
            "body_char_count",
            "is_stub",
            "content_stream",
            "updated_at",
            "attachments",
            "fetched_at",
            "crawler_version",
        }
        assert expected <= set(row)

    def test_computed_fields_are_serialised_not_recomputed(self) -> None:
        """Downstream stages read the classification; they never re-derive it."""
        row = make_article(attachments=[make_attachment()]).to_manifest_row()
        assert row["is_stub"] is True
        assert row["content_stream"] == "pdf"
        assert row["body_char_count"] == 38

    def test_row_is_json_serialisable(self) -> None:
        """Manifest rows are written as JSONL, so they must serialise directly."""
        row = make_article(attachments=[make_attachment()]).to_manifest_row()
        assert json.loads(json.dumps(row))["article_id"] == "47001247136"

    def test_provenance_is_stamped_automatically(self) -> None:
        """Every artefact carries when it was fetched and by which crawler."""
        article = make_article()
        assert article.fetched_at.endswith("Z")
        assert article.crawler_version

    def test_round_trip_through_json_preserves_classification(self) -> None:
        """Reading a manifest back must reproduce the same article."""
        original = make_article(attachments=[make_attachment()])
        restored = Article.model_validate_json(json.dumps(original.to_manifest_row()))
        assert restored.article_id == original.article_id
        assert restored.is_stub == original.is_stub
        assert restored.attachments[0].filename == original.attachments[0].filename


def test_utc_now_iso_format() -> None:
    """Provenance timestamps are UTC ISO-8601 with a Z suffix."""
    stamp = utc_now_iso()
    assert stamp.endswith("Z")
    assert len(stamp) == 20


class TestFreshdeskSource:
    """The source interface."""

    def test_cannot_be_instantiated_directly(self) -> None:
        """It is an ABC; an implementation must supply all three methods."""
        with pytest.raises(TypeError):
            FreshdeskSource()  # type: ignore[abstract]

    def test_a_partial_implementation_is_rejected(self) -> None:
        """Missing a method fails at construction, not at the first call."""

        class Partial(FreshdeskSource):
            def list_folders(self) -> list[dict[str, str]]:
                return []

        with pytest.raises(TypeError):
            Partial()  # type: ignore[abstract]

    def test_a_complete_implementation_works(self) -> None:
        """An ApiClient could be dropped in without any downstream change."""

        class FakeApiClient(FreshdeskSource):
            def list_folders(self) -> list[dict[str, str]]:
                return [{"id": "1", "name": "Service Guides", "url": "u", "category": "DGH"}]

            def list_articles(self, folder_id: str) -> list[dict[str, str]]:
                return [{"id": "2", "title": "t", "url": "u"}]

            def get_article(self, article_id: str) -> Article:
                return make_article(article_id=article_id)

        client = FakeApiClient()
        assert isinstance(client, FreshdeskSource)
        assert client.get_article("99").article_id == "99"
