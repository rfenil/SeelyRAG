"""Every stubbed module must import cleanly and fail honestly.

A stub that raises ``ImportError`` at collection time breaks the whole suite; a
stub that silently returns ``None`` is worse, because a later stage will build on
the nothing it returned. Both failure modes are checked here.
"""

from __future__ import annotations

import importlib

import pytest

#: Every module that is deliberately not implemented yet.
#:
#: parse.pdf, parse.html and parse.pagelabels were implemented in Stage 2, the
#: chunk in Stage 3, index in Stage 4, retrieve in Stage 5, generate in
#: Stage 6 and the API in Stage 7; each has its own tests. parse.vision is still stubbed: pages
#: needing it are recorded with ``needs_vision=True`` so it can be added
#: without re-parsing, and the index is incremental so folding them in later
#: will not re-embed the corpus.
STUB_MODULES = [
    "seeley_rag.parse.vision",
]

#: ``(module, callable, args)`` for each stub entry point.
STUB_CALLS = [
    ("seeley_rag.parse.vision", "transcribe_page", ("x.png",)),
    ("seeley_rag.parse.vision", "caption_diagram", ("x.png",)),
]


@pytest.mark.parametrize("module_name", STUB_MODULES)
def test_stub_module_imports(module_name: str) -> None:
    """Stubs must import cleanly so lint, typing and collection all work."""
    assert importlib.import_module(module_name) is not None


@pytest.mark.parametrize(("module_name", "func_name", "args"), STUB_CALLS)
def test_stub_raises_not_implemented(
    module_name: str, func_name: str, args: tuple[object, ...]
) -> None:
    """Calling a stub fails loudly and points at the specifying section."""
    func = getattr(importlib.import_module(module_name), func_name)
    with pytest.raises(NotImplementedError, match="build-plan.md"):
        func(*args)


def test_api_schemas_are_usable() -> None:
    """The API request/response models are real, even though the app is a stub.

    They are the integration contract with .NET, so they are worth having
    concrete before the server exists.
    """
    from seeley_rag.api.schemas import AskRequest, AskResponse, Citation

    request = AskRequest(query="TQ heater showing FC7, what do I check?")
    assert request.top_k == 8

    response = AskResponse(
        query_id="q_01",
        answer="Check the flame sensor [1]",
        citations=[Citation(n=1, title="TQ Service Guide", page_label="42")],
    )
    # /ask must return a query_id, because /feedback takes one.
    assert response.query_id == "q_01"
    assert response.citations[0].page_label == "42"


def test_parse_package_exposes_its_implemented_modules() -> None:
    """Stage 0 triage and the Stage 2 parsers are all real and re-exported."""
    from seeley_rag.parse import (
        has_table_signal,
        ingest_articles,
        parse_pdf,
        resolve_label,
        triage_corpus,
        triage_pdf,
    )

    for func in (
        triage_pdf,
        triage_corpus,
        parse_pdf,
        has_table_signal,
        resolve_label,
        ingest_articles,
    ):
        assert callable(func)


def test_vision_is_the_only_parse_stub_left() -> None:
    """The one deferred piece of Stage 2, and it must fail loudly.

    Pages needing it carry needs_vision=True in pages.jsonl, so the outstanding
    work is queued and countable rather than silently missing.
    """
    from seeley_rag.parse import vision

    with pytest.raises(NotImplementedError, match="build-plan.md"):
        vision.transcribe_page("page.png")
    with pytest.raises(NotImplementedError, match="build-plan.md"):
        vision.caption_diagram("page.png")
