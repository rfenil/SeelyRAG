"""Stage 4 build planning and the LanceDB store.

The partition in :func:`~seeley_rag.index.build.plan_build` is the mechanism the
user asked for: it is what lets the deferred vision transcriptions be folded in
later without re-embedding the corpus. It is tested against a real LanceDB
table, because the properties that matter -- upsert on ``chunk_id``, delete by
id, round-tripping every citation field -- are properties of the store, and a
mock would assert only that the code calls the methods it calls.

No network: LanceDB is embedded, and the embedder is a fake.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from seeley_rag.chunk.base import Chunk
from seeley_rag.index.build import build_index, indexed_hashes, plan_build, verify_index
from seeley_rag.index.embed_cache import EmbeddingCache
from seeley_rag.index.embedder import Embedder
from seeley_rag.index.store import StoreError, open_store

pytest.importorskip("lancedb", reason="Stage 4 store tests need lancedb")

DIM = 8


class FakeEmbeddings:
    """Returns a deterministic vector per input."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **payload: Any) -> Any:
        """Return one vector per input text.

        Args:
            **payload: The request.

        Returns:
            An OpenAI-shaped response.
        """
        self.calls.append(payload)
        return type(
            "Response",
            (),
            {
                "data": [
                    type("Item", (), {"index": i, "embedding": [float(len(t))] * DIM})()
                    for i, t in enumerate(payload["input"])
                ]
            },
        )()


class FakeClient:
    """An OpenAI-shaped client."""

    def __init__(self) -> None:
        self.embeddings = FakeEmbeddings()


@pytest.fixture
def embedder(tmp_path: Path) -> Embedder:
    """An embedder that never touches the network.

    Args:
        tmp_path: pytest's temporary directory.

    Returns:
        The embedder.
    """
    return Embedder(
        client=FakeClient(), cache=EmbeddingCache(root=tmp_path / "cache"), dimensions=DIM
    )


@pytest.fixture
def store(tmp_path: Path) -> Any:
    """An empty LanceDB store in a temporary directory.

    Args:
        tmp_path: pytest's temporary directory.

    Returns:
        The store.
    """
    return open_store(path=tmp_path / "index", table_name="chunks_test")


def make_chunk(chunk_id: str, text: str, **overrides: Any) -> Chunk:
    """Build a finalised chunk with realistic citation metadata.

    Args:
        chunk_id: Identifier.
        text: Chunk text.
        **overrides: Fields to replace.

    Returns:
        The chunk.
    """
    fields: dict[str, Any] = {
        "chunk_id": chunk_id,
        "doc_id": "d" * 64,
        "text": text,
        "page_index": 41,
        "page_label": "42",
        "label_source": "embedded",
        "title": "TQ Service Guide",
        "category": "Ducted Gas Heating",
        "folder": "Service Guides",
        "product_family": "DGH",
        "model_series": ["TQ"],
        "fault_codes": ["FC07"],
        "source_url": "https://example.invalid/a",
    }
    fields.update(overrides)
    return Chunk(**fields).finalise()


class TestPlanning:
    """Partition the corpus before spending anything."""

    def test_everything_is_new_against_an_empty_store(self, store: Any) -> None:
        """A first build embeds all of it."""
        chunks = [make_chunk("a:p0:c00", "First."), make_chunk("a:p1:c00", "Second.")]
        plan = plan_build(chunks, store)
        assert len(plan.new) == 2
        assert not plan.changed and not plan.unchanged and not plan.removed

    def test_an_unchanged_corpus_needs_no_embedding(self, store: Any, embedder: Embedder) -> None:
        """The property that makes re-indexing cheap.

        Re-running over untouched chunks must cost zero API calls.
        """
        chunks = [make_chunk("a:p0:c00", "First."), make_chunk("a:p1:c00", "Second.")]
        build_index(chunks, embedder=embedder, store=store, build_indexes=False)
        calls_before = len(embedder._client.embeddings.calls)

        plan = plan_build(chunks, store)
        assert len(plan.unchanged) == 2
        assert plan.to_embed == []

        build_index(chunks, embedder=embedder, store=store, build_indexes=False)
        assert len(embedder._client.embeddings.calls) == calls_before

    def test_edited_text_is_detected_as_changed(self, store: Any, embedder: Embedder) -> None:
        """Same id, different hash: re-embed exactly this one."""
        build_index(
            [make_chunk("a:p0:c00", "Original.")],
            embedder=embedder,
            store=store,
            build_indexes=False,
        )
        plan = plan_build([make_chunk("a:p0:c00", "Edited.")], store)
        assert len(plan.changed) == 1
        assert not plan.new

    def test_a_vanished_chunk_is_marked_removed(self, store: Any, embedder: Embedder) -> None:
        """A stale row would keep serving text that no longer exists."""
        build_index(
            [make_chunk("a:p0:c00", "First."), make_chunk("a:p1:c00", "Second.")],
            embedder=embedder,
            store=store,
            build_indexes=False,
        )
        plan = plan_build([make_chunk("a:p0:c00", "First.")], store)
        assert plan.removed == ["a:p1:c00"]

    def test_mixed_corpus_partitions_correctly(self, store: Any, embedder: Embedder) -> None:
        """The realistic case after a partial re-chunk."""
        build_index(
            [make_chunk("a:p0:c00", "Keep."), make_chunk("a:p1:c00", "Edit.")],
            embedder=embedder,
            store=store,
            build_indexes=False,
        )
        plan = plan_build(
            [
                make_chunk("a:p0:c00", "Keep."),
                make_chunk("a:p1:c00", "Edited now."),
                make_chunk("a:p2:c00", "Brand new."),
            ],
            store,
        )
        assert plan.summary() == {
            "unchanged": 1,
            "changed": 1,
            "new": 1,
            "removed": 0,
            "to_embed": 2,
        }

    def test_hashes_of_an_absent_table_are_empty(self, store: Any) -> None:
        """A first run must not raise on a missing table."""
        assert indexed_hashes(store) == {}


class TestVisionBackfillShape:
    """The scenario the incremental design exists for."""

    def test_transcribing_one_page_re_embeds_only_that_page(
        self, store: Any, embedder: Embedder
    ) -> None:
        """The user deferred vision on condition it could be added cheaply.

        Simulates it: three chunks indexed, one page later gains transcribed
        text. Only that chunk may reach the API.
        """
        original = [
            make_chunk("d:p0:c00", "Page one text.", needs_vision=False),
            make_chunk("d:p1:c00", "", needs_vision=True, tier="scanned"),
            make_chunk("d:p2:c00", "Page three text.", needs_vision=False),
        ]
        build_index(original, embedder=embedder, store=store, build_indexes=False)
        calls_before = len(embedder._client.embeddings.calls)

        after_vision = [
            original[0],
            make_chunk(
                "d:p1:c00",
                "Transcribed: E:04 flame sensing fault.",
                needs_vision=False,
                tier="scanned",
            ),
            original[2],
        ]
        report = build_index(after_vision, embedder=embedder, store=store, build_indexes=False)

        assert report["changed"] == 1
        assert report["unchanged"] == 2
        assert report["new"] == 0
        # Exactly one further request, carrying exactly one text.
        new_calls = embedder._client.embeddings.calls[calls_before:]
        assert len(new_calls) == 1
        assert len(new_calls[0]["input"]) == 1
        assert store.count() == 3, "the update must not duplicate rows"


class TestStore:
    """Upsert, delete, and the citation fields that must survive."""

    def test_upsert_updates_in_place_rather_than_appending(
        self, store: Any, embedder: Embedder
    ) -> None:
        """Append would duplicate every re-indexed chunk."""
        build_index(
            [make_chunk("a:p0:c00", "Original.")],
            embedder=embedder,
            store=store,
            build_indexes=False,
        )
        build_index(
            [make_chunk("a:p0:c00", "Edited.")], embedder=embedder, store=store, build_indexes=False
        )
        assert store.count() == 1
        assert "Edited." in store.get("a:p0:c00")["text"]

    def test_removed_rows_are_deleted(self, store: Any, embedder: Embedder) -> None:
        """A row with no source text must leave the index."""
        build_index(
            [make_chunk("a:p0:c00", "First."), make_chunk("a:p1:c00", "Second.")],
            embedder=embedder,
            store=store,
            build_indexes=False,
        )
        build_index(
            [make_chunk("a:p0:c00", "First.")], embedder=embedder, store=store, build_indexes=False
        )
        assert store.count() == 1
        assert store.get("a:p1:c00") is None

    def test_citation_fields_survive_the_round_trip(self, store: Any, embedder: Embedder) -> None:
        """Without these a retrieved chunk cannot be cited, which is the point."""
        build_index(
            [make_chunk("a:p41:c00", "Body.")], embedder=embedder, store=store, build_indexes=False
        )
        row = store.get("a:p41:c00")
        assert row["page_label"] == "42"
        assert row["label_source"] == "embedded"
        assert row["title"] == "TQ Service Guide"
        assert row["product_family"] == "DGH"
        assert list(row["model_series"]) == ["TQ"]
        assert list(row["fault_codes"]) == ["FC07"]

    def test_is_table_is_materialised_for_filtering(self, store: Any, embedder: Embedder) -> None:
        """It is a property on the model, so it must be written explicitly."""
        build_index(
            [make_chunk("a:p0:c00", "Rows.", kind="table")],
            embedder=embedder,
            store=store,
            build_indexes=False,
        )
        assert store.get("a:p0:c00")["is_table"] is True

    def test_vectors_are_not_returned_in_results(self, store: Any, embedder: Embedder) -> None:
        """3,072 floats per result would swamp every log and response."""
        build_index(
            [make_chunk("a:p0:c00", "Body.")], embedder=embedder, store=store, build_indexes=False
        )
        assert "vector" not in store.get("a:p0:c00")

    def test_mismatched_vector_count_is_rejected(self, store: Any) -> None:
        """A silent misalignment attaches vectors to the wrong chunks."""
        with pytest.raises(StoreError, match="one to one"):
            store.upsert([make_chunk("a:p0:c00", "Body.")], [])

    def test_querying_a_missing_table_names_the_fix(self, store: Any) -> None:
        """The error has to say what to run."""
        with pytest.raises(StoreError, match="05_embed"):
            _ = store.table


class TestSearch:
    """Both retrieval channels, over a real table."""

    @pytest.fixture
    def populated(self, store: Any, embedder: Embedder) -> Any:
        """A store with a handful of distinguishable chunks.

        Args:
            store: Empty store.
            embedder: Fake embedder.

        Returns:
            The populated store.
        """
        build_index(
            [
                make_chunk("a:p0:c00", "Flame sensing fault on the TQ series heater."),
                make_chunk("a:p1:c00", "Evaporative cooler pump replacement procedure."),
                make_chunk("a:p2:c00", "Gas valve pressure test at 1.0 kPa."),
            ],
            embedder=embedder,
            store=store,
            build_indexes=False,
        )
        store.create_fts_index()
        return store

    def test_bm25_finds_an_exact_term(self, populated: Any) -> None:
        """BM25 is what catches model numbers and code strings."""
        results = populated.search_bm25("evaporative pump", top_k=3)
        assert results
        assert "Evaporative" in results[0]["text"]

    def test_dense_search_returns_scored_rows(self, populated: Any) -> None:
        """Shape matters more than ranking with a fake embedder."""
        results = populated.search_dense([5.0] * DIM, top_k=2)
        assert len(results) == 2
        assert all("score" in row for row in results)
        assert all("vector" not in row for row in results)

    def test_results_carry_what_a_citation_needs(self, populated: Any) -> None:
        """A result that cannot be cited is not an answer."""
        row = populated.search_bm25("flame sensing", top_k=1)[0]
        assert row["title"] and row["page_label"] and row["chunk_id"]


class TestVerification:
    """Catch a malformed index where it is built, not at query time."""

    def test_verify_reports_the_vector_width(self, store: Any, embedder: Embedder) -> None:
        """A wrong width fails queries with a confusing error."""
        build_index(
            [make_chunk("a:p0:c00", "Body.")], embedder=embedder, store=store, build_indexes=False
        )
        checks = verify_index(store, expected_dim=DIM)
        assert checks["dim_matches"]
        assert checks["vector_dim"] == DIM
        assert checks["ids_present"]

    def test_verify_detects_a_width_mismatch(self, store: Any, embedder: Embedder) -> None:
        """Configuration says one thing, the index holds another."""
        build_index(
            [make_chunk("a:p0:c00", "Body.")], embedder=embedder, store=store, build_indexes=False
        )
        assert not verify_index(store, expected_dim=3072)["dim_matches"]

    def test_verify_refuses_a_missing_index(self, store: Any) -> None:
        """Nothing to verify is an error, not a pass."""
        with pytest.raises(StoreError):
            verify_index(store, expected_dim=DIM)
