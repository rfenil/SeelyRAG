"""Stage 4 -- embedding and the LanceDB vector + full-text index.

build-plan.md section 6.
"""

from __future__ import annotations

from seeley_rag.index.build import (
    IndexPlan,
    build_index,
    embed_chunks,
    indexed_hashes,
    plan_build,
    search_smoke,
    verify_index,
)
from seeley_rag.index.embed_cache import EmbeddingCache, cache_key, default_cache, namespace
from seeley_rag.index.embedder import Embedder, EmbeddingError
from seeley_rag.index.store import CHUNK_COLUMNS, LanceDBStore, StoreError, VectorStore, open_store

__all__: list[str] = [
    "CHUNK_COLUMNS",
    "EmbeddingCache",
    "Embedder",
    "EmbeddingError",
    "IndexPlan",
    "LanceDBStore",
    "StoreError",
    "VectorStore",
    "build_index",
    "cache_key",
    "default_cache",
    "embed_chunks",
    "indexed_hashes",
    "namespace",
    "open_store",
    "plan_build",
    "search_smoke",
    "verify_index",
]
