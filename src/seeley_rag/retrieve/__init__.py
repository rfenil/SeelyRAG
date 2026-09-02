"""Stage 5 -- the retrieval cascade.

build-plan.md section 7.
"""

from __future__ import annotations

from seeley_rag.retrieve.hybrid import (
    CodeIndex,
    PinnedCode,
    RetrievalError,
    apply_boosts,
    reciprocal_rank_fusion,
    retrieve,
    search,
)
from seeley_rag.retrieve.query import Understanding, understand, understand_deterministic
from seeley_rag.retrieve.rerank import (
    cohere_rerank,
    get_reranker,
    identity_rerank,
    llm_rerank,
    rerank,
    rerank_backend,
)

__all__: list[str] = [
    "CodeIndex",
    "PinnedCode",
    "RetrievalError",
    "Understanding",
    "apply_boosts",
    "cohere_rerank",
    "get_reranker",
    "identity_rerank",
    "llm_rerank",
    "reciprocal_rank_fusion",
    "rerank",
    "rerank_backend",
    "retrieve",
    "search",
    "understand",
    "understand_deterministic",
]
