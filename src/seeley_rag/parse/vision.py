"""Stage 2 -- vision transcription and captioning. STUB.

build-plan.md section 4.2, tiers B and C.

Tier B (scanned, no text layer) gets a full transcription. Tier C
(diagram-heavy) keeps its extracted text and gains a one-line caption, so the
page is findable by "TQ wiring diagram".

**Tier C is a vision call too.** In illustrated service manuals it can be 20-30%
of pages -- plausibly more volume than Tier B. Budget it explicitly from the
triage numbers rather than from a fixed estimate; it is the only line in the
cost model that can move by 3x.

Responses cache under ``data/cache/llm/``, keyed by page-image hash, so a
re-parse does not re-spend the vision budget.
"""

from __future__ import annotations

from pathlib import Path

#: Tier B. Preserve tables exactly -- every fault code and its description.
TRANSCRIBE_PROMPT = (
    "Transcribe this service-manual page to markdown. Preserve all tables "
    "exactly, including every fault/error code and its description. Describe any "
    "wiring diagram or exploded parts view in one sentence prefixed [DIAGRAM]. "
    "Output only the transcription."
)


def transcribe_page(image_path: Path) -> str:
    """Transcribe a scanned page to markdown (Tier B).

    Args:
        image_path: Rendered page PNG.

    Returns:
        The transcription.

    Raises:
        NotImplementedError: Always -- this is a stub.
    """
    raise NotImplementedError(
        "Stage 2 -- not yet implemented; see _context/01-plan/build-plan.md section 4.2"
    )


def caption_diagram(image_path: Path) -> str:
    """Caption a diagram-heavy page in one line (Tier C).

    Args:
        image_path: Rendered page PNG.

    Returns:
        A one-line caption prefixed ``[DIAGRAM]``.

    Raises:
        NotImplementedError: Always -- this is a stub.
    """
    raise NotImplementedError(
        "Stage 2 -- not yet implemented; see _context/01-plan/build-plan.md section 4.2"
    )
