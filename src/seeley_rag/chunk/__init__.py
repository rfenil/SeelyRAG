"""Stage 3 -- page-anchored chunking, atomic tables, and fault-code extraction.

build-plan.md section 5.
"""

from __future__ import annotations

from seeley_rag.chunk.base import (
    Chunk,
    FaultCode,
    JsonlWriter,
    chunk_hashes,
    content_hash,
    make_chunk_id,
    read_chunks,
    read_codes,
)
from seeley_rag.chunk.chunker import (
    build_breadcrumb,
    chunk_corpus,
    chunk_document,
    chunk_page,
    split_text,
)
from seeley_rag.chunk.codes import (
    annotate_chunks,
    build_code_table,
    codes_by_key,
    extract_codes,
    normalise_code,
)
from seeley_rag.chunk.tables import (
    is_continuation,
    merge_multipage_tables,
    render_table,
    split_oversized_table,
)
from seeley_rag.chunk.tokens import count_tokens, estimate_tokens, truncate_to_tokens

__all__: list[str] = [
    "Chunk",
    "FaultCode",
    "JsonlWriter",
    "annotate_chunks",
    "build_breadcrumb",
    "build_code_table",
    "chunk_corpus",
    "chunk_document",
    "chunk_hashes",
    "chunk_page",
    "codes_by_key",
    "content_hash",
    "count_tokens",
    "estimate_tokens",
    "extract_codes",
    "is_continuation",
    "make_chunk_id",
    "merge_multipage_tables",
    "normalise_code",
    "read_chunks",
    "read_codes",
    "render_table",
    "split_oversized_table",
    "split_text",
    "truncate_to_tokens",
]
