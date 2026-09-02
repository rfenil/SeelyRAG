"""Stage 5 fusion, boosts and the cascade.

build-plan section 7.2. The property most worth protecting here is the *order*
of steps 5 and 6: boosts are applied to the whole fused list and only then is it
truncated for reranking. The plan calls out applying them after truncation as a
bug in v1, so there is a test that fails if it ever comes back.
"""

from __future__ import annotations

from typing import Any

import pytest

from seeley_rag.chunk.base import FaultCode
from seeley_rag.retrieve.hybrid import (
    CodeIndex,
    RetrievalError,
    apply_boosts,
    reciprocal_rank_fusion,
    retrieve,
)
from seeley_rag.retrieve.query import Understanding, understand_deterministic


def row(chunk_id: str, **overrides: Any) -> dict[str, Any]:
    """Build a retrieval result row.

    Args:
        chunk_id: Identifier.
        **overrides: Fields to replace.

    Returns:
        The row.
    """
    base: dict[str, Any] = {
        "chunk_id": chunk_id,
        "text": f"body of {chunk_id}",
        "title": f"Doc {chunk_id}",
        "product_family": "DGH",
        "model_series": [],
        "fault_codes": [],
        "content_stream": "pdf",
        "kind": "prose",
        "page_label": "42",
    }
    base.update(overrides)
    return base


class TestReciprocalRankFusion:
    """score = sum(1 / (k + rank)), and nothing else."""

    def test_single_list_preserves_order(self) -> None:
        """One channel in, the same order out."""
        fused = reciprocal_rank_fusion([[row("a"), row("b"), row("c")]])
        assert [r["chunk_id"] for r in fused] == ["a", "b", "c"]

    def test_scores_match_the_formula(self) -> None:
        """The formula is the whole algorithm; assert it literally."""
        fused = reciprocal_rank_fusion([[row("a"), row("b")]], k=60)
        assert fused[0]["fused_score"] == pytest.approx(1 / 61)
        assert fused[1]["fused_score"] == pytest.approx(1 / 62)

    def test_agreement_across_channels_wins(self) -> None:
        """A chunk both channels rank mid-list beats one channel's favourite.

        This is the entire reason for fusing rather than concatenating.
        """
        dense = [row("a"), row("shared"), row("c")]
        bm25 = [row("d"), row("shared"), row("f")]
        fused = reciprocal_rank_fusion([dense, bm25], k=60)
        assert fused[0]["chunk_id"] == "shared"

    def test_rank_not_score_is_what_fuses(self) -> None:
        """Immune to the two channels' incompatible score scales.

        Dense returns a cosine distance, BM25 an unbounded relevance score.
        Wildly different magnitudes must not change the fusion at all.
        """
        dense = [row("a", score=0.001), row("b", score=0.0005)]
        bm25 = [row("b", score=98.6), row("a", score=42.0)]
        fused = reciprocal_rank_fusion([dense, bm25], k=60)
        assert fused[0]["fused_score"] == pytest.approx(fused[1]["fused_score"])

    def test_per_channel_ranks_are_recorded(self) -> None:
        """`--explain` needs them, and so does anyone debugging a bad result."""
        fused = reciprocal_rank_fusion([[row("a")], [row("a")]])
        assert fused[0]["ranks"] == {"channel_0": 1, "channel_1": 1}

    def test_empty_input(self) -> None:
        """No channels, no results, no error."""
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[], []]) == []

    def test_a_row_without_a_chunk_id_is_fatal(self) -> None:
        """Fusing on a missing key would silently merge unrelated chunks."""
        with pytest.raises(RetrievalError, match="chunk_id"):
            reciprocal_rank_fusion([[{"text": "no id"}]])


class TestBoosts:
    """Step 5: multipliers on the fused score, never filters."""

    def understanding(self, **overrides: Any) -> Understanding:
        """Build an Understanding.

        Args:
            **overrides: Fields to replace.

        Returns:
            The understanding.
        """
        fields: dict[str, Any] = {"query": "q", "product_family": "DGH"}
        fields.update(overrides)
        return Understanding(**fields)

    def test_diagnostic_article_is_boosted(self) -> None:
        """Installer-written fault prose is worth more per byte than a manual."""
        fused = reciprocal_rank_fusion(
            [[row("pdf"), row("art", content_stream="diagnostic_article")]]
        )
        boosted = apply_boosts(fused, self.understanding(product_family="UNKNOWN"))
        assert boosted[0]["chunk_id"] == "art", "the boost failed to promote"
        assert "diagnostic_article" in boosted[0]["boosts"]

    def test_product_family_match_is_boosted(self) -> None:
        """The fix for BM25 drifting into a different product line."""
        fused = reciprocal_rank_fusion([[row("wrong", product_family="EVAP"), row("right")]])
        boosted = apply_boosts(fused, self.understanding(product_family="DGH"))
        assert boosted[0]["chunk_id"] == "right"

    def test_unknown_family_boosts_nothing(self) -> None:
        """An uninferred family must not accidentally boost chunks labelled UNKNOWN."""
        fused = reciprocal_rank_fusion([[row("a", product_family="UNKNOWN")]])
        boosted = apply_boosts(fused, self.understanding(product_family="UNKNOWN"))
        assert boosted[0]["boosts"] == []

    def test_fault_code_match_is_boosted(self) -> None:
        """Codes are the strongest exact signal in the corpus."""
        fused = reciprocal_rank_fusion([[row("no"), row("yes", fault_codes=["FC07"])]])
        boosted = apply_boosts(fused, self.understanding(fault_codes=["FC07"]))
        assert boosted[0]["chunk_id"] == "yes"
        assert "fault_code" in boosted[0]["boosts"]

    def test_model_series_match_is_boosted(self) -> None:
        """A TQ question should prefer TQ documents."""
        fused = reciprocal_rank_fusion([[row("no"), row("yes", model_series=["TQ"])]])
        boosted = apply_boosts(fused, self.understanding(model_series=["TQ"]))
        assert boosted[0]["chunk_id"] == "yes"

    def test_boosts_compound(self) -> None:
        """Several signals agreeing should outrank one."""
        fused = reciprocal_rank_fusion(
            [
                [
                    row("one", fault_codes=["FC07"]),
                    row(
                        "many",
                        fault_codes=["FC07"],
                        model_series=["TQ"],
                        content_stream="diagnostic_article",
                    ),
                ]
            ]
        )
        boosted = apply_boosts(fused, self.understanding(fault_codes=["FC07"], model_series=["TQ"]))
        assert boosted[0]["chunk_id"] == "many"
        assert len(boosted[0]["boosts"]) == 4

    def test_a_wrong_family_is_demoted_not_removed(self) -> None:
        """Soft-boost, never filter -- build-plan section 7.1.

        If the classifier guesses wrong and we filtered, the installer gets
        nothing and concludes the system is broken.
        """
        fused = reciprocal_rank_fusion([[row("evap", product_family="EVAP"), row("dgh")]])
        boosted = apply_boosts(fused, self.understanding(product_family="DGH"))
        assert {r["chunk_id"] for r in boosted} == {"evap", "dgh"}, "a chunk was filtered out"

    def test_boosts_do_not_mutate_the_input(self) -> None:
        """The fused list is reused for debugging output."""
        fused = reciprocal_rank_fusion([[row("a", content_stream="diagnostic_article")]])
        before = fused[0]["fused_score"]
        apply_boosts(fused, self.understanding())
        assert fused[0]["fused_score"] == before


class TestBoostBeforeTruncation:
    """The ordering bug the build plan calls out by name."""

    def test_a_boost_can_promote_from_outside_the_top_k(self) -> None:
        """v1 boosted after truncating to top-8, where this is impossible.

        A diagnostic article ranked 20th by fusion must be able to reach the
        returned list. If boosts ever move after truncation, this fails.
        """
        ranking = [row(f"c{i}") for i in range(20)]
        ranking.append(row("late", content_stream="diagnostic_article", fault_codes=["FC07"]))
        fused = reciprocal_rank_fusion([ranking])
        boosted = apply_boosts(
            fused, Understanding(query="q", product_family="DGH", fault_codes=["FC07"])
        )
        top_8 = [r["chunk_id"] for r in boosted[:8]]
        assert "late" in top_8, "boost applied too late to promote anything"


class TestCodeIndex:
    """Step 1: exact lookup ahead of retrieval."""

    @pytest.fixture
    def index(self) -> CodeIndex:
        """A code table covering one code in two families.

        Returns:
            The index.
        """
        return CodeIndex(
            [
                FaultCode(
                    code="E4", code_key="E04", meaning="VRF compressor", product_family="VRF"
                ),
                FaultCode(code="E4", code_key="E04", meaning="RC compressor", product_family="RC"),
                FaultCode(
                    code="7", code_key="FC07", meaning="Ignition failure", product_family="DGH"
                ),
                FaultCode(
                    code="7", code_key="FC07", meaning="Supply motor error", product_family="EVAP"
                ),
                # The corpus really contains rows like this: a code whose family
                # could not be resolved, whose "meaning" is the code restated.
                FaultCode(
                    code="7", code_key="FC07", meaning="FAULT CODE 7", product_family="UNKNOWN"
                ),
            ]
        )

    def test_family_match_wins(self, index: CodeIndex) -> None:
        """E:04 means different things on different products."""
        pins = index.lookup(["E04"], "VRF")
        assert len(pins) == 1
        assert pins[0].row.product_family == "VRF"
        assert not pins[0].cross_family

    def test_no_family_match_is_returned_but_flagged(self, index: CodeIndex) -> None:
        """The real case: "ducted heater throwing E:04".

        DGH prints FC codes, so there is no DGH E04 at all. Pinning a VRF
        compressor fault unflagged would be a confident wrong answer; returning
        nothing would hide that the code exists elsewhere.
        """
        pins = index.lookup(["E04"], "DGH")
        assert pins
        assert all(p.cross_family for p in pins)

    def test_no_family_named_returns_every_meaning_as_ambiguous(self, index: CodeIndex) -> None:
        """The shape installers actually type: a code and nothing else.

        With no family named there is nothing to contradict, so nothing is
        ``cross_family`` -- but neither is any one row the answer. Every meaning
        comes back marked ``ambiguous`` so the generator enumerates instead of
        picking.
        """
        pins = index.lookup(["E04"], "UNKNOWN")
        assert len(pins) == 2
        assert all(p.ambiguous for p in pins)
        assert not any(p.cross_family for p in pins)

    def test_an_unknown_family_row_does_not_shadow_the_real_ones(self, index: CodeIndex) -> None:
        """``UNKNOWN`` is not a family, and matching on it hid the real answer.

        A bare ``fc7`` used to pin exactly one row -- the one whose meaning is
        the string "FAULT CODE 7" -- because it "matched" the unresolved family,
        and the DGH ignition failure the installer was standing in front of
        never reached the model at all.
        """
        families = {p.row.product_family for p in index.lookup(["FC07"], "UNKNOWN")}
        assert families == {"DGH", "EVAP", "UNKNOWN"}

    def test_a_named_family_still_narrows(self, index: CodeIndex) -> None:
        """Enumerating for someone who did say which unit would be noise."""
        pins = index.lookup(["FC07"], "DGH")
        assert len(pins) == 1
        assert pins[0].row.product_family == "DGH"
        assert not pins[0].ambiguous

    def test_an_unknown_code_returns_nothing(self, index: CodeIndex) -> None:
        """Absence must be empty, not a guess."""
        assert index.lookup(["E99"], "DGH") == []

    def test_len_counts_distinct_keys(self, index: CodeIndex) -> None:
        """Families sharing a code count once."""
        assert len(index) == 2


class FakeStore:
    """A store returning fixed results for both channels."""

    def __init__(self, dense: list[dict[str, Any]], bm25: list[dict[str, Any]]) -> None:
        self._dense = dense
        self._bm25 = bm25
        self.dense_calls: list[int] = []
        self.bm25_calls: list[str] = []
        self.wheres: list[str | None] = []

    def search_dense(
        self, vector: Any, top_k: int, where: str | None = None
    ) -> list[dict[str, Any]]:
        """Return the canned dense results."""
        self.dense_calls.append(top_k)
        self.wheres.append(where)
        return self._dense[:top_k]

    def search_bm25(self, query: str, top_k: int, where: str | None = None) -> list[dict[str, Any]]:
        """Return the canned BM25 results."""
        self.bm25_calls.append(query)
        self.wheres.append(where)
        return self._bm25[:top_k]


class FakeEmbedder:
    """An embedder returning a fixed vector."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per text."""
        return [[0.0, 1.0] for _ in texts]


class BrokenEmbedder:
    """An embedder that simulates an outbound embedding failure."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Fail like the OpenAI embedding client when the network is unavailable."""
        raise RuntimeError("Embedding failed after 5 attempts: Connection error.")


class TestCascade:
    """The whole pipeline, with both channels faked."""

    def build(self, **overrides: Any) -> dict[str, Any]:
        """Run the cascade over canned results.

        Args:
            **overrides: Passed to :func:`retrieve`.

        Returns:
            The retrieval result.
        """
        store = FakeStore(
            dense=[row("d1"), row("shared"), row("d2")],
            bm25=[row("b1"), row("shared"), row("b2")],
        )
        kwargs: dict[str, Any] = {
            "store": store,
            "embedder": FakeEmbedder(),
            "code_index": CodeIndex([]),
            "use_llm": False,
        }
        kwargs.update(overrides)
        return retrieve("TQ heater fault code FC7", **kwargs)

    def test_cascade_returns_ranked_chunks(self) -> None:
        """End to end, with agreement winning."""
        result = self.build()
        assert result["results"]
        assert result["results"][0]["chunk_id"] == "shared"

    def test_counts_report_each_stage(self) -> None:
        """So a bad result can be attributed to a channel."""
        counts = self.build()["counts"]
        assert counts["dense"] == 3
        assert counts["bm25"] == 3
        assert counts["fused"] == 5
        assert counts["returned"] == 5

    def test_top_k_is_honoured(self) -> None:
        """The caller's limit, not the configured one."""
        assert len(self.build(top_k=2)["results"]) == 2

    def test_understanding_is_returned(self) -> None:
        """The API needs the inferred family for its response."""
        parsed = self.build()["understanding"]
        assert parsed.product_family == "DGH"
        assert parsed.fault_codes == ["FC07"]

    def test_codes_are_pinned_ahead_of_retrieval(self) -> None:
        """Step 1 of the cascade."""
        index = CodeIndex(
            [FaultCode(code="7", code_key="FC07", meaning="Ignition", product_family="DGH")]
        )
        result = self.build(code_index=index)
        assert len(result["pinned_codes"]) == 1
        assert result["pinned_codes"][0].row.code_key == "FC07"

    def test_the_rewritten_query_reaches_bm25(self) -> None:
        """Both channels must see the same text, or fusion compares nothing."""
        store = FakeStore(dense=[row("a")], bm25=[row("a")])
        retrieve(
            "TQ FC7",
            store=store,
            embedder=FakeEmbedder(),
            code_index=CodeIndex([]),
            use_llm=False,
        )
        assert store.bm25_calls == ["TQ FC7"]

    def test_bm25_runs_when_query_embedding_fails(self) -> None:
        """Search should still work locally when dense embedding cannot call OpenAI."""
        store = FakeStore(dense=[row("dense")], bm25=[row("lexical")])
        result = retrieve(
            "TQ FC7",
            store=store,
            embedder=BrokenEmbedder(),
            code_index=CodeIndex([]),
            use_llm=False,
        )

        assert [r["chunk_id"] for r in result["results"]] == ["lexical"]
        assert store.dense_calls == []
        assert store.bm25_calls == ["TQ FC7"]
        assert result["counts"]["dense"] == 0
        assert result["counts"]["bm25"] == 1

    def test_a_pre_filter_reaches_both_channels(self) -> None:
        """A caller-supplied filter constrains the search, not its output.

        Filtering afterwards returns nothing whenever the matches sat below the
        truncation point -- a filter that only works when it was not needed.
        """
        store = FakeStore(dense=[row("a")], bm25=[row("a")])
        retrieve(
            "q",
            store=store,
            embedder=FakeEmbedder(),
            code_index=CodeIndex([]),
            use_llm=False,
            where="product_family = 'VRF'",
        )
        assert store.wheres == ["product_family = 'VRF'", "product_family = 'VRF'"]

    def test_an_explicit_product_hint_overrides_inference(self) -> None:
        """The caller stated it, so it beats a guess -- still only boosted."""
        result = retrieve(
            "the heater is faulty",
            store=FakeStore(dense=[row("a")], bm25=[row("a")]),
            embedder=FakeEmbedder(),
            code_index=CodeIndex([]),
            use_llm=False,
            product_hint="EVAP",
        )
        assert result["understanding"].product_family == "EVAP"

    def test_a_store_failure_is_wrapped(self) -> None:
        """Callers catch one exception type, not the SDK's."""

        class Broken(FakeStore):
            def search_dense(self, vector: Any, top_k: int) -> list[dict[str, Any]]:
                raise ValueError("index corrupt")

        with pytest.raises(RetrievalError, match="Retrieval failed"):
            retrieve(
                "q",
                store=Broken([], []),
                embedder=FakeEmbedder(),
                code_index=CodeIndex([]),
                use_llm=False,
            )

    def test_a_custom_reranker_is_used(self) -> None:
        """The reranker is injectable so Cohere can be swapped in or out."""

        def only_first(query: str, candidates: list[dict[str, Any]], top_k: int) -> list:
            return candidates[:1]

        assert len(self.build(reranker=only_first)["results"]) == 1


def test_understanding_flows_into_boosts() -> None:
    """The two halves of the stage must agree on field names.

    A rename on one side would silently disable every boost, which no other
    test would catch because each half still works alone.
    """
    parsed = understand_deterministic("TQ heater fault code FC7")
    fused = reciprocal_rank_fusion(
        [[row("hit", model_series=["TQ"], fault_codes=["FC07"]), row("miss")]]
    )
    boosted = apply_boosts(fused, parsed)
    assert boosted[0]["chunk_id"] == "hit"
    assert {"product_family", "model_series", "fault_code"} <= set(boosted[0]["boosts"])
