"""Stage 3 chunking: page anchoring, sizing, breadcrumbs, determinism.

The load-bearing property here is rule 1 of build-plan section 5.1: a chunk
never spans two pages. Every citation the system emits depends on it, so it is
asserted directly rather than inferred from chunk sizes.
"""

from __future__ import annotations

import pytest

from seeley_rag.chunk.base import make_chunk_id
from seeley_rag.chunk.chunker import (
    build_breadcrumb,
    chunk_corpus,
    chunk_document,
    chunk_page,
    split_text,
)
from seeley_rag.chunk.tokens import count_tokens
from seeley_rag.parse.base import Page, Table

#: A body comfortably above the ``min_chunk_chars`` floor, so these tests
#: exercise chunking rather than the near-empty-page filter.
FLAME_BODY = "Check the flame sensor for continuity and confirm ignition."

#: Paragraph separator, spelled out to keep escape sequences out of the
#: test bodies.
SEPARATOR = chr(10) + chr(10)


def make_page(**overrides: object) -> Page:
    """Build a page with realistic Seeley metadata.

    Args:
        **overrides: Fields to replace.

    Returns:
        A page suitable for chunking.
    """
    defaults: dict[str, object] = {
        "doc_id": "a" * 64,
        "page_index": 41,
        "page_label": "42",
        "label_source": "embedded",
        "text": "Check the flame sensor for continuity and confirm the burner lights.",
        "tier": "plain_text",
        "title": "TQ Service Guide Gas Ducted Heater 644066-M",
        "category": "Ducted Gas Heating",
        "folder": "Service Guides",
        "product_family": "DGH",
        "model_series": ["TQ"],
        "source_url": "https://example.invalid/attachments/1",
    }
    defaults.update(overrides)
    return Page(**defaults)  # type: ignore[arg-type]


class TestBreadcrumb:
    """Rule 5: the breadcrumb puts the product name into every chunk's text."""

    def test_full_breadcrumb(self) -> None:
        """All four parts appear, in order, separated by ' > '."""
        crumb = build_breadcrumb("Ducted Gas Heating", "Service Guides", "TQ Service Guide", "42")
        assert crumb == "Ducted Gas Heating > Service Guides > TQ Service Guide > p.42"

    def test_missing_parts_are_dropped_not_blanked(self) -> None:
        """An absent folder must not leave an empty ' >  > ' segment."""
        assert build_breadcrumb("DGH", "", "TQ Guide", None) == "DGH > TQ Guide"

    def test_breadcrumb_is_prefixed_to_chunk_text(self) -> None:
        """The product name must be in the text that gets embedded.

        This is the whole point of rule 5: chunks whose body never names the
        product still retrieve for product-specific queries.
        """
        chunk = chunk_page(make_page(text=FLAME_BODY))[0]
        assert chunk.text.startswith("Ducted Gas Heating > Service Guides > TQ Service")
        assert FLAME_BODY in chunk.text


class TestPageAnchoring:
    """Rule 1: a chunk never crosses a page boundary."""

    def test_every_chunk_belongs_to_exactly_one_page(self) -> None:
        """Two pages of long text produce chunks that each name one page."""
        pages = [
            make_page(page_index=0, page_label="1", text="Alpha sentence here. " * 400),
            make_page(page_index=1, page_label="2", text="Beta sentence here. " * 400),
        ]
        chunks = chunk_document(pages)
        assert len(chunks) > 2, "long pages must split into several chunks"
        for chunk in chunks:
            assert chunk.page_index in (0, 1)
            # Content from one page must never appear in a chunk labelled the
            # other -- that is the failure this rule exists to prevent.
            body = chunk.text.split("\n\n", 1)[1]
            if chunk.page_index == 0:
                assert "Beta" not in body
            else:
                assert "Alpha" not in body

    def test_page_provenance_is_carried_onto_every_chunk(self) -> None:
        """Label and its source travel with the chunk, for citation and eval."""
        chunk = chunk_page(make_page(page_label="42", label_source="embedded"))[0]
        assert chunk.page_label == "42"
        assert chunk.label_source == "embedded"
        assert chunk.page_image is None
        assert chunk.product_family == "DGH"
        assert chunk.model_series == ["TQ"]

    def test_guessed_labels_stay_marked_as_guesses(self) -> None:
        """38.7% of labels are guesses; the eval must be able to exclude them."""
        chunk = chunk_page(make_page(label_source="index"))[0]
        assert chunk.label_source == "index"


class TestSizing:
    """Rule 2: ~800 target, ~1,200 hard max, ~120 overlap."""

    def test_short_text_stays_one_chunk(self) -> None:
        """Nothing is split that already fits."""
        assert split_text("A short procedure step.") == ["A short procedure step."]

    def test_empty_text_yields_nothing(self) -> None:
        """Whitespace must not become a chunk."""
        assert split_text("   \n\n  ") == []

    def test_no_piece_exceeds_the_hard_maximum(self) -> None:
        """The ceiling is an API limit, not a preference."""
        text = "\n\n".join(f"Paragraph {i} about gas pressure testing. " * 20 for i in range(40))
        for piece in split_text(text, target_tokens=200, max_tokens=300):
            assert count_tokens(piece) <= 300

    def test_unbreakable_text_is_still_capped(self) -> None:
        """Input with no paragraph, line or sentence break must still fit.

        A run-on table dumped as prose would otherwise sail past the ceiling.
        """
        for piece in split_text("wordy " * 5000, target_tokens=200, max_tokens=300):
            assert count_tokens(piece) <= 300

    def test_overlap_repeats_the_tail_of_the_previous_piece(self) -> None:
        """A procedure split across two chunks must read from either one."""
        text = "\n\n".join(f"Step {i}: isolate the appliance and test." for i in range(200))
        pieces = split_text(text, target_tokens=100, max_tokens=200, overlap_tokens=20)
        assert len(pieces) > 2
        tail_words = set(pieces[0].split()[-8:])
        assert tail_words & set(pieces[1].split()[:20]), "no overlap carried forward"

    def test_zero_overlap_is_honoured(self) -> None:
        """Overlap is configurable down to nothing.

        Asserted on a unique per-paragraph marker rather than raw words: the
        surrounding boilerplate recurs in every paragraph, so word overlap
        between adjacent pieces would prove nothing either way.
        """
        text = SEPARATOR.join(f"Paragraph MARKER{i} describing a service step." for i in range(100))
        pieces = split_text(text, target_tokens=50, max_tokens=100, overlap_tokens=0)
        assert len(pieces) > 1
        first = {w for w in pieces[0].split() if w.startswith("MARKER")}
        second = {w for w in pieces[1].split() if w.startswith("MARKER")}
        assert first and second
        assert not first & second, "no paragraph may appear in both pieces"


class TestDeterminism:
    """Deterministic ids are what make incremental re-indexing possible."""

    def test_chunk_ids_are_stable_across_runs(self) -> None:
        """The same page chunked twice yields identical ids and hashes."""
        page = make_page(text="Flame sensing fault diagnosis. " * 50)
        first = chunk_page(page)
        second = chunk_page(page)
        assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
        assert [c.content_hash for c in first] == [c.content_hash for c in second]

    def test_chunk_id_encodes_document_page_and_ordinal(self) -> None:
        """Ids must be derivable, not sequential, or upserts become duplicates."""
        assert make_chunk_id("abc", 41, 0) == "abc:p41:c00"
        assert make_chunk_id("article:47001247475", None, 3) == "article:47001247475:pna:c03"

    def test_changed_text_changes_the_hash_but_not_the_id(self) -> None:
        """That pairing is exactly what lets a re-index embed only what moved."""
        body = "Original body text describing the burner assembly procedure."
        original = chunk_page(make_page(text=body))[0]
        edited = chunk_page(make_page(text=body.replace("Original", "Edited")))[0]
        assert original.chunk_id == edited.chunk_id
        assert original.content_hash != edited.content_hash

    def test_hash_covers_the_breadcrumb_not_just_the_body(self) -> None:
        """The hash must describe the text actually embedded, prefix included."""
        body = "Identical body text describing the same diagnostic procedure."
        one = chunk_page(make_page(text=body))[0]
        other = chunk_page(make_page(text=body, title="Different Manual"))[0]
        assert one.content_hash != other.content_hash


class TestTablesInPages:
    """Rule 3: a detected table is one atomic chunk, and leads the page."""

    def test_table_becomes_its_own_chunk(self) -> None:
        """Table and prose must not be welded into one chunk."""
        page = make_page(
            text="Refer to the fault code table above for the full listing.",
            tables=[
                Table(
                    rows=[["Code", "Meaning"], ["E:04", "Flame sensing fault"]],
                    has_header=True,
                    n_columns=2,
                    bbox=(0.0, 0.0, 100.0, 50.0),
                )
            ],
        )
        chunks = chunk_page(page)
        assert len(chunks) == 2
        assert chunks[0].kind == "table"
        assert chunks[0].is_table
        assert "E:04" in chunks[0].text
        assert chunks[1].kind == "prose"

    def test_table_chunk_leads_the_page(self) -> None:
        """The table gets the lowest, most stable ordinal on its page."""
        page = make_page(
            text="Some surrounding prose that is long enough to survive.",
            tables=[Table(rows=[["A", "B"]], has_header=True, n_columns=2)],
        )
        assert chunk_page(page)[0].chunk_id.endswith(":c00")


class TestCorpusStreaming:
    """chunk_corpus groups a streamed pages.jsonl back into documents."""

    def test_documents_are_grouped_and_all_chunks_emitted(self) -> None:
        """Pages arrive contiguously per document; every one must be chunked."""
        pages = [
            make_page(
                doc_id="doc1",
                page_index=0,
                page_label="1",
                text="First document, page one, describing the burner assembly.",
            ),
            make_page(
                doc_id="doc1",
                page_index=1,
                page_label="2",
                text="First document, page two, describing the gas valve test.",
            ),
            make_page(
                doc_id="doc2",
                page_index=0,
                page_label="1",
                text="Second document, page one, describing the fan motor check.",
            ),
        ]
        chunks = list(chunk_corpus(pages))
        assert len(chunks) == 3
        assert [c.doc_id for c in chunks] == ["doc1", "doc1", "doc2"]

    def test_empty_input_yields_nothing(self) -> None:
        """An empty corpus must not raise."""
        assert list(chunk_corpus([])) == []


class TestArticleStream:
    """Diagnostic articles share the schema but have no page index."""

    def test_article_page_chunks_without_a_page_index(self) -> None:
        """The 630 diagnostic articles must chunk on the same code path."""
        page = make_page(
            doc_id="article:47001247475",
            page_index=None,
            page_label=None,
            label_source="none",
            content_stream="diagnostic_article",
            text="FC7 is an intermittent fault on TQ3 series heaters.",
        )
        chunk = chunk_page(page)[0]
        assert chunk.page_index is None
        assert chunk.content_stream == "diagnostic_article"
        assert chunk.chunk_id.endswith(":pna:c00")
        # No page label means no "p.N" segment in the breadcrumb.
        assert "p.None" not in chunk.text


@pytest.mark.parametrize("overlap", [0, 50, 120])
def test_overlap_never_breaks_the_ceiling(overlap: int) -> None:
    """The overlap is a convenience; the token ceiling is an API limit."""
    text = "\n\n".join(f"Paragraph {i} discussing burner pressure." for i in range(120))
    for piece in split_text(text, target_tokens=100, max_tokens=150, overlap_tokens=overlap):
        assert count_tokens(piece) <= 150
