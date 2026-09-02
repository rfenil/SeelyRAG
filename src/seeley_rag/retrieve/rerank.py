"""Stage 5 -- reranking.

build-plan.md section 7.2, step 6.

Cohere ``rerank-v3.5`` where a key is available. The plan flags obtaining that
key as a Day 0 item because the listwise LLM fallback costs roughly $0.016 per
query in rerank input alone and about doubles the per-query figure in section 12.

Three backends, chosen in this order:

* ``cohere`` -- with a ``COHERE_API_KEY``. Preferred.
* ``llm`` -- the plan's own listwise fallback, opt-in via
  ``retrieve.use_llm_rerank`` because of the cost warning above. It runs on
  whichever provider :mod:`seeley_rag.llm` is configured for, so the OpenAI key
  already in use for embeddings is enough.
* ``identity`` -- returns the boosted fusion order unchanged, and says so.

⚠ **The identity backend does not pretend to rerank.** The obvious cheap
substitutes were rejected: lexical overlap is a worse BM25, and re-scoring with
the same embedding model is the dense channel again -- both already represented
in the fusion. Either would move results around without adding information,
which is worse than not reranking, because it looks like a quality step and is
not one.

Every result carries ``rerank_backend``, and :func:`rerank_backend` reports the
active path, so an eval can attribute its numbers rather than silently crediting
a ranking nothing reranked.
"""

from __future__ import annotations

import importlib.util
from typing import Any, Callable, Sequence

from seeley_rag.logging_conf import get_logger
from seeley_rag.settings import get_settings

log = get_logger(__name__)

#: Cohere's reranker truncates inputs near 4,000 tokens. Chunks are capped well
#: below that by Stage 3, but a merged multi-page table can approach it, so the
#: text sent is bounded rather than assumed safe.
MAX_DOCUMENT_CHARS = 12_000


#: Instruction for the listwise reranker. Asks for an ordering, not a rewrite:
#: the model may reorder evidence, never author it.
LLM_RERANK_PROMPT = (
    "You rank retrieved passages from Seeley International HVAC service manuals "
    "by how well each answers an installer's question.\n\n"
    "You are given a question and numbered passages. Return ONLY JSON:\n"
    '  {"order": [<passage numbers, best first>]}\n\n'
    "Rank by whether the passage contains the specific answer -- a fault "
    "code meaning, a procedure step, an exact figure -- not by how much it "
    "talks about the topic. A passage about the right product that answers "
    "the question beats one about a different product using the same words."
    "\n\n"
    "Each header carries the product family, the page, and a tag where the "
    "passage is a DIAGNOSTIC ARTICLE -- installer-written fault-finding prose "
    "about a real fault, which for a symptom question is usually the most "
    "directly useful source in the corpus. Prefer it where it answers the "
    "question. Do not prefer it where it does not."
    "\n\n"
    "Include only passages that are relevant. Do not invent passage numbers."
)


def cohere_installed() -> bool:
    """Whether the Cohere SDK can be imported.

    A seam, so a test can describe a machine with the package present without
    the suite depending on an optional extra being installed.

    Returns:
        True when ``import cohere`` would succeed.
    """
    return importlib.util.find_spec("cohere") is not None


def cohere_available() -> bool:
    """Whether Cohere reranking can actually run: a key *and* the SDK.

    ``cohere`` is declared in the ``downstream`` extra, not in
    ``requirements.txt``, so a venv can hold the key without the package. Keying
    off the key alone made ``/health`` and the CLI report ``cohere`` while every
    query fell through :func:`cohere_rerank`'s except clause to identity -- the
    rows stayed honest, the status line did not, and the difference is only
    visible in a warning log nobody reads.

    Returns:
        True when both are present.
    """
    return bool(get_settings().cohere_api_key) and cohere_installed()


def rerank_backend() -> str:
    """Return which reranking backend is active.

    Returns:
        ``"cohere"`` with a Cohere key and the SDK installed, ``"llm"`` when
        listwise reranking is enabled, else ``"identity"``.
    """
    if cohere_available():
        return "cohere"
    if get_settings().retrieve.use_llm_rerank:
        return "llm"
    return "identity"


def _passage_header(number: int, row: dict[str, Any]) -> str:
    """Render the header line for one listwise-rerank passage.

    The reranker sees a title and 600 characters of body, and nothing else --
    which means it is blind to the metadata the boosts already act on. Measured
    on the query log, that showed up as the reranker demoting a diagnostic
    article beneath training-slide pages on "TQ heater has no flame": the
    ``diagnostic_article`` boost had promoted it, and the reranker, unable to
    see why, undid the promotion. Restating the signal in the header lets the
    model weigh it instead of discarding it.

    Args:
        number: 0-based passage index, which is what the model orders by.
        row: A fused, boosted candidate.

    Returns:
        The header line.
    """
    facets = [str(row.get("product_family") or "UNKNOWN")]
    page = row.get("page_range") or row.get("page_label")
    if page:
        facets.append(f"p.{page}")
    if row.get("content_stream") == "diagnostic_article":
        facets.append("DIAGNOSTIC ARTICLE")
    return f"[{number}] {str(row.get('title', ''))[:80]} ({', '.join(facets)})"


def llm_rerank(
    query: str,
    candidates: Sequence[dict[str, Any]],
    top_k: int = 8,
    client: Any | None = None,
    model: str | None = None,
) -> list[dict[str, Any]]:
    """Rerank listwise with the configured LLM.

    build-plan section 7.2 step 6 names this as the fallback when no Cohere key
    exists, and warns it roughly doubles per-query cost -- so it is opt-in via
    ``retrieve.use_llm_rerank``.

    Unlike the identity backend this is a genuine reranking: the model reads the
    passages and the question together, which neither fused channel does.

    Ordering is all it may return. It never rewrites a passage, so nothing it
    produces can reach the answer as evidence; a hallucinated index is dropped,
    and any passage the model omits is appended in its existing order rather
    than lost.

    Args:
        query: The installer's question.
        candidates: Fused, boosted candidates.
        top_k: Results to keep.
        client: Injected SDK client, for tests.
        model: Override the reranking model. Defaults to
            ``retrieve.llm_rerank_model``, which falls back to the router
            model when unset.

    Returns:
        The top ``top_k`` candidates in the model's order.
    """
    from seeley_rag import llm

    rows = list(candidates)
    if not rows:
        return []

    pool = rows[: get_settings().retrieve.llm_rerank_candidates]
    listing = ("\n\n").join(
        _passage_header(i, row) + "\n" + " ".join(str(row.get("text", "")).split())[:600]
        for i, row in enumerate(pool)
    )

    try:
        payload = llm.complete_json(
            system=LLM_RERANK_PROMPT,
            user=f"Question: {query}\n\nPassages:\n{listing}",
            model=model or get_settings().retrieve.llm_rerank_model,
            client=client,
            max_tokens=400,
        )
        order = payload.get("order")
        if not isinstance(order, list):
            raise llm.LLMError(f"Reranker returned {type(order).__name__}, expected a list.")
    except Exception as exc:  # noqa: BLE001 - any SDK or parse failure is non-fatal
        log.warning("rerank_failed_falling_back", extra={"error": str(exc)})
        return identity_rerank(query, rows, top_k)

    ordered: list[dict[str, Any]] = []
    seen: set[int] = set()
    for position, index in enumerate(order):
        # Hallucinated or repeated indices are dropped rather than trusted.
        if not isinstance(index, int) or not 0 <= index < len(pool) or index in seen:
            continue
        seen.add(index)
        ranked = dict(pool[index])
        ranked["rerank_backend"] = "llm"
        ranked["rerank_score"] = 1.0 - position * 0.01
        ordered.append(ranked)

    # A passage the model simply ignored keeps its fused rank rather than
    # vanishing -- silence is not evidence of irrelevance.
    for index, row in enumerate(pool):
        if index not in seen:
            ranked = dict(row)
            ranked["rerank_backend"] = "llm"
            ranked["rerank_score"] = 0.0
            ordered.append(ranked)

    return ordered[:top_k]


def identity_rerank(
    query: str, candidates: Sequence[dict[str, Any]], top_k: int = 8
) -> list[dict[str, Any]]:
    """Return the boosted fusion order, truncated.

    Not a no-op in effect: the list it receives has already been fused across
    both channels and boosted on product family, model series, fault code and
    content stream. It is a reasonable ranking. It is simply not a *reranking*,
    and is labelled accordingly.

    Args:
        query: The installer's question. Unused; kept for interface parity.
        candidates: Fused, boosted candidates.
        top_k: Results to keep.

    Returns:
        The top ``top_k`` candidates, each marked with the backend used.
    """
    out: list[dict[str, Any]] = []
    for row in list(candidates)[:top_k]:
        ranked = dict(row)
        ranked["rerank_backend"] = "identity"
        ranked["rerank_score"] = ranked.get("boosted_score", ranked.get("fused_score", 0.0))
        out.append(ranked)
    return out


def cohere_rerank(
    query: str,
    candidates: Sequence[dict[str, Any]],
    top_k: int = 8,
    client: Any | None = None,
) -> list[dict[str, Any]]:
    """Rerank with Cohere ``rerank-v3.5``.

    A failure falls back to :func:`identity_rerank` rather than raising. A
    reranker being unavailable should cost result quality, not the answer --
    the installer still gets the fused, boosted list.

    Args:
        query: The installer's question.
        candidates: Fused, boosted candidates.
        top_k: Results to keep.
        client: Injected Cohere client, for tests.

    Returns:
        The top ``top_k`` candidates in the reranker's order.
    """
    rows = list(candidates)
    if not rows:
        return []

    settings = get_settings()
    try:
        if client is None:
            import cohere

            client = cohere.Client(api_key=settings.cohere_api_key)

        documents = [str(row.get("text", ""))[:MAX_DOCUMENT_CHARS] for row in rows]
        response = client.rerank(
            model="rerank-v3.5", query=query, documents=documents, top_n=min(top_k, len(rows))
        )
        ordered: list[dict[str, Any]] = []
        for item in response.results:
            ranked = dict(rows[item.index])
            ranked["rerank_backend"] = "cohere"
            ranked["rerank_score"] = float(item.relevance_score)
            ordered.append(ranked)
        return ordered
    except Exception as exc:  # noqa: BLE001 - SDK raises many types
        log.warning("rerank_failed_falling_back", extra={"error": str(exc)})
        return identity_rerank(query, rows, top_k)


def rerank(
    query: str,
    candidates: Sequence[dict[str, Any]],
    top_k: int = 8,
    client: Any | None = None,
) -> list[dict[str, Any]]:
    """Rerank fused candidates using whichever backend is available.

    Args:
        query: The installer's question.
        candidates: Fused, boosted candidates.
        top_k: Results to keep.
        client: Injected Cohere client, for tests.

    Returns:
        The top ``top_k`` chunks.
    """
    backend = rerank_backend()
    if backend == "cohere":
        return cohere_rerank(query, candidates, top_k, client=client)
    if backend == "llm":
        return llm_rerank(query, candidates, top_k, client=client)
    return identity_rerank(query, candidates, top_k)


def get_reranker() -> Callable[[str, Sequence[dict[str, Any]], int], list[dict[str, Any]]]:
    """Return the active reranking callable.

    Returns:
        A callable with the :func:`rerank` signature.
    """
    return rerank
