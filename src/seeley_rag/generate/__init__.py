"""Stage 6 -- grounded answer synthesis.

build-plan.md section 8.
"""

from __future__ import annotations

from seeley_rag.generate.answer import (
    answer,
    assemble,
    build_citation,
    cited_numbers,
    log_query,
    new_query_id,
)
from seeley_rag.generate.prompts import (
    build_context,
    build_user_message,
    render_passage,
    render_pinned_code,
    system_prompt,
)

__all__: list[str] = [
    "answer",
    "assemble",
    "build_citation",
    "build_context",
    "build_user_message",
    "cited_numbers",
    "log_query",
    "new_query_id",
    "render_passage",
    "render_pinned_code",
    "system_prompt",
]
