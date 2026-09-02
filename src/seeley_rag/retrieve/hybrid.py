"""Stage 5 -- the retrieval cascade.

build-plan.md section 7.2.

1. **Code lookup** -- a fault code in the query hits the code table exactly and
   is pinned into context.
2. **Dense** -- top 30.
3. **BM25** -- top 30. Catches model numbers, part numbers, code strings.
4. **RRF fusion** -- ``score = sum(1 / (60 + rank_i))``. Parameter-free, works.
5. **Boosts, applied here -- before truncation.** Product-family match, and
   ``content_stream == "diagnostic_article"`` x 1.2.
6. **Rerank to top 5-8.**

⚠ **Step 5 happens before step 6, and that ordering is the whole point.** v1 of
the plan applied the stream boost after reranking to top-8, where it can only
reorder what already survived and can never promote the chunk that should have
been there. Boost the fused scores over all ~60 candidates, *then* truncate.

Why both channels are needed, measured on this corpus rather than assumed. Asked
"Braemar evaporative cooler water pump not priming", dense search returns the
correct Braemar evaporative manuals while BM25 returns *Coolerado* documents --
a different product line that happens to share cooling vocabulary. Asked for VRF
``E4``, BM25 finds the right VRF fault table while dense drifts to reverse
cycle. Neither channel alone is trustworthy on product family; fusion plus the
family boost is what fixes it.
"""

from __future__ import annotations

import collections
import functools
from typing import Any, Iterable, NamedTuple, Sequence

from seeley_rag.chunk.base import FaultCode, read_codes
from seeley_rag.exceptions import SeeleyRagError
from seeley_rag.logging_conf import get_logger
from seeley_rag.parse.base import UNKNOWN_FAMILY
from seeley_rag.retrieve.query import Understanding, understand
from seeley_rag.settings import get_settings

log = get_logger(__name__)


class RetrievalError(SeeleyRagError):
    """Retrieval could not run -- usually a missing index or code table."""


# ---------------------------------------------------------------------------
# Process-wide handles
# ---------------------------------------------------------------------------
#
# Opening the LanceDB table costs ~4.8s against the 16,189-row index, while the
# searches themselves are 30-80ms. Rebuilding the handle per query made a
# 150ms cascade take 7.4s. These are cached for the process, like
# ``get_settings()``; tests and the API inject their own.


@functools.lru_cache(maxsize=1)
def default_store() -> Any:
    """Return the process-wide vector store.

    Returns:
        The configured :class:`~seeley_rag.index.store.LanceDBStore`.
    """
    from seeley_rag.index.store import open_store

    return open_store()


@functools.lru_cache(maxsize=1)
def default_embedder() -> Any:
    """Return the process-wide embedder.

    Returns:
        The configured :class:`~seeley_rag.index.embedder.Embedder`.
    """
    from seeley_rag.index.embedder import Embedder

    return Embedder()


@functools.lru_cache(maxsize=1)
def default_code_index() -> CodeIndex:
    """Return the process-wide fault-code table.

    Returns:
        The loaded :class:`CodeIndex`.
    """
    return CodeIndex()


# ---------------------------------------------------------------------------
# Step 4 -- fusion
# ---------------------------------------------------------------------------


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[dict[str, Any]]], k: int = 60
) -> list[dict[str, Any]]:
    """Fuse several ranked lists with RRF.

    ``score = sum(1 / (k + rank))`` over the lists a chunk appears in, ranks
    1-based. Parameter-free in the sense that matters: it uses only *rank*, so
    it is immune to dense search returning a cosine distance while BM25 returns
    an unbounded relevance score. Normalising those two onto a common scale is
    the thing that quietly goes wrong; RRF sidesteps it.

    Args:
        rankings: Ranked result lists, best first.
        k: The RRF constant. 60 is the standard value and is not tuned here.

    Returns:
        One fused list, descending ``fused_score``, each row carrying the
        per-channel ranks it came from.

    Raises:
        RetrievalError: If a result row has no ``chunk_id`` to fuse on.
    """
    fused: dict[str, dict[str, Any]] = {}
    contributions: dict[str, dict[str, int]] = collections.defaultdict(dict)

    for channel_index, ranking in enumerate(rankings):
        channel = f"channel_{channel_index}"
        for rank, row in enumerate(ranking, start=1):
            chunk_id = row.get("chunk_id")
            if not chunk_id:
                raise RetrievalError(
                    "A retrieval result has no chunk_id; results cannot be fused without one."
                )
            if chunk_id not in fused:
                fused[chunk_id] = dict(row)
                fused[chunk_id]["fused_score"] = 0.0
            fused[chunk_id]["fused_score"] += 1.0 / (k + rank)
            contributions[chunk_id][channel] = rank

    for chunk_id, row in fused.items():
        row["ranks"] = contributions[chunk_id]

    return sorted(fused.values(), key=lambda r: r["fused_score"], reverse=True)


# ---------------------------------------------------------------------------
# Step 5 -- boosts, applied to the fused list before truncation
# ---------------------------------------------------------------------------


def apply_boosts(
    candidates: Sequence[dict[str, Any]], understanding: Understanding
) -> list[dict[str, Any]]:
    """Multiply fused scores by the metadata boosts, then re-sort.

    Every boost is a multiplier and none is a filter. A chunk from the wrong
    product family loses rank; it does not leave the list. That is deliberate --
    section 7.1: if the classifier guesses wrong and we filtered, the installer
    gets nothing and concludes the system is broken.

    Args:
        candidates: Fused candidates.
        understanding: The parsed query.

    Returns:
        A new list, descending ``boosted_score``, each row recording which
        boosts fired so a result can be explained.
    """
    settings = get_settings().retrieve
    wanted_codes = set(understanding.fault_codes)
    wanted_series = {s.upper() for s in understanding.model_series}
    boosted: list[dict[str, Any]] = []

    for row in candidates:
        score = float(row.get("fused_score", 0.0))
        reasons: list[str] = []

        if row.get("content_stream") == "diagnostic_article":
            score *= settings.diagnostic_article_boost
            reasons.append("diagnostic_article")

        if (
            understanding.product_family != "UNKNOWN"
            and row.get("product_family") == understanding.product_family
        ):
            score *= settings.product_family_boost
            reasons.append("product_family")

        if wanted_series and wanted_series & {s.upper() for s in row.get("model_series") or []}:
            score *= settings.model_series_boost
            reasons.append("model_series")

        if wanted_codes and wanted_codes & set(row.get("fault_codes") or []):
            score *= settings.code_match_boost
            reasons.append("fault_code")

        out = dict(row)
        out["boosted_score"] = score
        out["boosts"] = reasons
        boosted.append(out)

    return sorted(boosted, key=lambda r: r["boosted_score"], reverse=True)


# ---------------------------------------------------------------------------
# Step 1 -- the fault-code table
# ---------------------------------------------------------------------------


class PinnedCode(NamedTuple):
    """A fault-code row pinned into context, with its provenance.

    Attributes:
        row: The code row.
        cross_family: True when the code exists, but not for the product family
            the query resolved to. The answer must say so rather than present it
            as the code's meaning here.
        ambiguous: True when the query named a code but no product at all, so
            every family's meaning is a candidate and none of them is *the*
            answer. Distinct from ``cross_family``, which presupposes a family
            was named: telling a model a code "does not appear in the product
            family this question is about" when the question named no product
            is incoherent, and it ignores it.
    """

    row: FaultCode
    cross_family: bool
    ambiguous: bool = False


class CodeIndex:
    """The exact-lookup fault-code table, loaded once.

    build-plan section 5.3: a detected code hits this table *before* retrieval
    runs, and its chunk is pinned into context. Exact lookup beats semantic
    search at exact-lookup problems.

    Args:
        rows: Code rows. Defaults to reading ``data/02_processed/codes.jsonl``.
    """

    def __init__(self, rows: Iterable[FaultCode] | None = None) -> None:
        self._by_key: dict[str, list[FaultCode]] = collections.defaultdict(list)
        for row in rows if rows is not None else read_codes():
            self._by_key[row.code_key].append(row)

    def lookup(self, code_keys: Sequence[str], product_family: str = "UNKNOWN") -> list[PinnedCode]:
        """Return code rows for the given keys, preferring the query's family.

        ``E:04`` means one thing on a gas heater and another on a VRF unit, so a
        family match wins where one exists.

        ⚠ ``UNKNOWN`` is not a family. Some code rows carry it, and matching
        on it made a bare ``fc7`` pin exactly one row -- the one whose meaning
        is the string "FAULT CODE 7" -- as an authoritative family match, while
        hiding the DGH ignition-failure and EVAP motor-error meanings that
        answer the question. A query that names no product gets every meaning,
        marked ``ambiguous``.

        Where a family is named and none of its rows match, the rows are still
        returned, but flagged ``cross_family``. That distinction is load-bearing. Asked "the ducted
        heater is throwing E:04", the family resolves to DGH correctly and the
        code table has no DGH ``E04`` at all -- because DGH prints ``FC`` codes.
        Pinning a VRF compressor fault as though it answered the question would
        be a confident wrong answer with a citation attached. Returning nothing
        would hide that the code exists elsewhere. So it is returned, marked, and
        the caller can say "E:04 is not a gas-heating code; on VRF it means...".

        Args:
            code_keys: Normalised code keys from the query.
            product_family: The inferred family.

        Returns:
            Matching rows, family matches first, each marked.
        """
        named = product_family and product_family != UNKNOWN_FAMILY
        found: list[PinnedCode] = []
        for key in code_keys:
            rows = self._by_key.get(key, [])
            if not named:
                # No product named. Every family's meaning is a candidate and
                # none is the answer, so all of them are pinned as ambiguous.
                found.extend(PinnedCode(row=r, cross_family=False, ambiguous=True) for r in rows)
                continue
            matching = [r for r in rows if r.product_family == product_family]
            if matching:
                found.extend(PinnedCode(row=r, cross_family=False) for r in matching)
            else:
                found.extend(PinnedCode(row=r, cross_family=True) for r in rows)
        return found

    def __len__(self) -> int:
        """Number of distinct code keys held."""
        return len(self._by_key)


# ---------------------------------------------------------------------------
# The cascade
# ---------------------------------------------------------------------------


def retrieve(
    query: str,
    top_k: int | None = None,
    store: Any | None = None,
    embedder: Any | None = None,
    code_index: CodeIndex | None = None,
    understanding: Understanding | None = None,
    reranker: Any | None = None,
    use_llm: bool | None = None,
    product_hint: str | None = None,
    where: str | None = None,
) -> dict[str, Any]:
    """Run the full cascade.

    Args:
        query: The installer's question.
        top_k: Chunks to return. Defaults to configured ``rerank_top_k``.
        store: Vector store. Defaults to the configured LanceDB index.
        embedder: Embedder for the query vector. Defaults to a new one.
        code_index: Fault-code table. Defaults to reading ``codes.jsonl``.
        understanding: Pre-parsed query, to skip re-parsing.
        reranker: Callable ``(query, candidates, top_k) -> list``. Defaults to
            the configured reranker.
        use_llm: Whether query understanding may call an LLM. Defaults to
            ``retrieve.use_query_llm`` -- which is off. It defaulted to True
            here once, which silently bypassed that config flag and added
            seconds to every direct call.
        product_hint: Product family supplied by the caller. Overrides the
            inferred one -- the caller stated it, so it beats a guess. Still
            only boosted, never filtered (build-plan section 7.1).
        where: SQL pre-filter applied inside both channels. This IS a hard
            filter, and only reaches here from a caller who typed one -- an
            *inferred* family is never turned into one.

    Returns:
        ``{"query_id"-less" result}``: the understanding, pinned code rows, the
        ranked chunks, and per-stage counts for debugging.

    Raises:
        RetrievalError: If the index is missing or a channel fails.
    """
    settings = get_settings().retrieve
    limit = top_k or settings.rerank_top_k

    parsed = understanding or understand(query, use_llm=use_llm)
    if product_hint:
        # An explicit hint outranks inference, but it is still only boosted: a
        # caller passing the wrong family should cost rank, not results.
        parsed = parsed.model_copy(update={"product_family": product_hint})

    store = store if store is not None else default_store()
    embedder = embedder if embedder is not None else default_embedder()

    # Step 1 -- exact code lookup, ahead of retrieval.
    pinned: list[PinnedCode] = []
    if parsed.fault_codes:
        index = code_index if code_index is not None else default_code_index()
        pinned = index.lookup(parsed.fault_codes, parsed.product_family)[
            : settings.max_pinned_codes
        ]

    # Steps 2 and 3 -- both channels, over the rewritten query.
    search_text = parsed.search_text
    dense_error: str | None = None
    try:
        vector = embedder.embed_texts([search_text])[0]
    except Exception as exc:  # noqa: BLE001 - SDK and cache layers raise many types
        # Dense search is only one retrieval channel. If the query embedding
        # cannot be produced (most often because the local test server has no
        # outbound access), BM25 should still run so Search and Docs remain
        # usable. Generation can still fail later if it needs the same network.
        dense = []
        dense_error = str(exc)
        log.warning("dense_embedding_failed_bm25_fallback", extra={"error": dense_error})
    else:
        try:
            dense = store.search_dense(vector, settings.dense_top_k, where=where)
        except Exception as exc:  # noqa: BLE001 - store raises backend-specific types
            raise RetrievalError(f"Retrieval failed: {exc}") from exc

    try:
        bm25 = store.search_bm25(search_text, settings.bm25_top_k, where=where)
    except Exception as exc:  # noqa: BLE001 - store and SDK raise many types
        if dense_error:
            raise RetrievalError(
                f"Retrieval failed: dense embedding failed ({dense_error}); BM25 failed: {exc}"
            ) from exc
        raise RetrievalError(f"Retrieval failed: {exc}") from exc

    # Step 4 -- fuse, then step 5 -- boost the whole fused list...
    fused = reciprocal_rank_fusion(
        [ranking for ranking in (dense, bm25) if ranking], k=settings.rrf_k
    )
    boosted = apply_boosts(fused, parsed)

    # ...and only now truncate for the reranker.
    if reranker is None:
        from seeley_rag.retrieve.rerank import rerank as default_rerank

        reranker = default_rerank
    ranked = reranker(search_text, boosted, limit)

    result = {
        "understanding": parsed,
        "pinned_codes": pinned,
        "results": ranked,
        "counts": {
            "dense": len(dense),
            "bm25": len(bm25),
            "fused": len(fused),
            "returned": len(ranked),
            "pinned": len(pinned),
        },
    }
    log.info(
        "retrieved",
        extra={
            "family": parsed.product_family,
            "codes": parsed.fault_codes,
            **result["counts"],
        },
    )
    return result


def search(query: str, top_k: int | None = None, **kwargs: Any) -> list[dict[str, Any]]:
    """Run the cascade and return just the ranked chunks.

    Convenience for ``/search`` and for callers that do not need the
    understanding or the pinned codes.

    Args:
        query: The installer's question.
        top_k: Chunks to return.
        **kwargs: Passed through to :func:`retrieve`.

    Returns:
        The ranked chunks.
    """
    return retrieve(query, top_k=top_k, **kwargs)["results"]
