"""Stage 7 -- FastAPI application.

build-plan.md section 9. The REST contract is the integration seam to .NET
later::

    POST /ask       {query, product_hint?, top_k?, stream?} -> answer + citations + query_id
    POST /search    {query, filters}                        -> raw chunks (debug)
    POST /feedback  {query_id, rating, comment}             -> ack
    GET  /pages/{doc_id}/{page_index}.png
    GET  /docs      -> corpus inventory
    GET  /health

``/ask`` returns ``query_id`` -- v1 of the plan had ``/feedback`` accept one that
nothing produced. Here it is minted in Stage 6, written to the query log with
the chunk IDs and the answer, and returned to the caller, so feedback can be
joined back to exactly what was retrieved and said.

Three things worth knowing before changing this file:

* **The store is opened at startup, not per request.** Opening the LanceDB table
  costs ~4.8s against 16,189 rows while the searches are 30-80ms (ADR 0007). A
  lazily-opened store makes the first request after every deploy look broken.
* **``/pages`` serves from a content-addressed tree and must not be a file-read
  primitive.** Every path is rebuilt from validated components and confirmed to
  resolve inside the image root -- see :func:`page_image_path`.
* **``stream`` is accepted and refused, not silently ignored.** Returning a
  whole response to a caller that asked for a stream looks like it worked.
"""

from __future__ import annotations

import json
import re
from contextlib import asynccontextmanager
from importlib.resources import files
from pathlib import Path
from typing import Any, AsyncIterator, Iterator

from seeley_rag.api.schemas import (
    AskRequest,
    AskResponse,
    DocsResponse,
    DocumentSummary,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
)
from seeley_rag.exceptions import SeeleyRagError
from seeley_rag.logging_conf import configure_logging, get_logger
from seeley_rag.page_images import page_image_url
from seeley_rag.settings import get_settings

log = get_logger(__name__)

#: A document id is a SHA-256 hex digest, or ``article:{numeric id}``. Anything
#: else cannot name a document in this corpus, so it is rejected before touching
#: the filesystem.
_DOC_ID_RE = re.compile(r"^(?:[0-9a-f]{64}|article:\d+)$")

#: Page images are written zero-padded four wide by Stage 2.
_IMAGE_NAME = "{index:04d}.png"

#: Filter values are lexicon identifiers, never free text.
_SAFE_VALUE_RE = re.compile(r"^[A-Za-z0-9_]{1,64}$")


def page_image_path(doc_id: str, page_index: int, root: Path | None = None) -> Path:
    """Resolve a page image path, or refuse.

    ``/pages/{doc_id}/{page_index}.png`` takes both components from the URL, so
    this is the one place in the project where a caller influences a filesystem
    read. It is written to be boring: the id must match a known shape, the index
    must be a non-negative integer, the name is rebuilt rather than taken, and
    the result is confirmed to resolve inside the image root. A traversal
    attempt fails the shape check long before any of that matters, and the
    containment check catches anything the shape check did not anticipate.

    Args:
        doc_id: Document SHA-256, or ``article:{id}``.
        page_index: 0-based page index.
        root: Image root. Defaults to ``data/01_interim/page_images``.

    Returns:
        The image path.

    Raises:
        ValueError: If the id or index is malformed, or the resolved path would
            escape the image root.
    """
    from seeley_rag.paths import PAGE_IMAGES_DIR

    if not _DOC_ID_RE.match(doc_id):
        raise ValueError(f"Not a document id: {doc_id!r}")
    if page_index < 0:
        raise ValueError(f"Page index must be non-negative, got {page_index}.")

    base = (root or PAGE_IMAGES_DIR).resolve()
    # `article:123` is not a legal directory name on Windows; Stage 2 stores
    # article pages under the id alone.
    directory = doc_id.split(":", 1)[1] if doc_id.startswith("article:") else doc_id
    candidate = (base / directory / _IMAGE_NAME.format(index=page_index)).resolve()
    if base not in candidate.parents:
        raise ValueError(f"Resolved path escapes the image root: {candidate}")
    return candidate


def to_hit(row: dict[str, Any]) -> SearchHit:
    """Convert a retrieval row into an API hit.

    Args:
        row: A ranked chunk from the cascade.

    Returns:
        The hit.
    """
    doc_id = str(row.get("doc_id") or "")
    page_index = row.get("page_index")
    page_url = page_image_url(doc_id, page_index) if isinstance(page_index, int) else None

    return SearchHit(
        chunk_id=str(row.get("chunk_id") or ""),
        doc_id=doc_id,
        title=str(row.get("title") or ""),
        page_label=row.get("page_range") or row.get("page_label"),
        page_image=row.get("page_image"),
        page_url=page_url,
        article_url=row.get("article_url"),
        product_family=str(row.get("product_family") or "UNKNOWN"),
        kind=str(row.get("kind") or "prose"),
        score=float(row.get("rerank_score") or row.get("boosted_score") or 0.0),
        boosts=list(row.get("boosts") or []),
        rerank_backend=str(row.get("rerank_backend") or "identity"),
        text=str(row.get("text") or ""),
    )


def build_predicate(request: SearchRequest) -> str | None:
    """Build the SQL pre-filter for a search request.

    Values are constrained to a safe character set rather than quoted and
    hoped for: this string is concatenated into a query, and the fields are
    lexicon-controlled identifiers (``DGH``, ``service_guide``), so anything
    outside that shape is a caller error, not a value to escape.

    Args:
        request: The search request.

    Returns:
        A predicate, or ``None`` when no filters were supplied.

    Raises:
        ValueError: If a filter value is not a plain identifier.
    """
    clauses: list[str] = []
    for field, value in (
        ("product_family", request.product_family),
        ("doc_type", request.doc_type),
    ):
        if value is None:
            continue
        if not _SAFE_VALUE_RE.match(value):
            raise ValueError(f"{field} must be a plain identifier, got {value!r}")
        clauses.append(f"{field} = '{value}'")

    if request.is_table is not None:
        clauses.append(f"is_table = {str(request.is_table).lower()}")

    return " AND ".join(clauses) if clauses else None


def read_feedback(path: Path | None = None) -> Iterator[dict[str, Any]]:
    """Stream the feedback log.

    Args:
        path: Source. Defaults to ``data/reports/feedback.jsonl``.

    Yields:
        Each feedback record.
    """
    from seeley_rag.paths import FEEDBACK_LOG_PATH

    resolved = path or FEEDBACK_LOG_PATH
    if not resolved.exists():
        return
    with resolved.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                try:
                    yield json.loads(stripped)
                except json.JSONDecodeError:
                    continue


def build_inventory(store: Any) -> DocsResponse:
    """Summarise the indexed corpus, one row per document.

    Args:
        store: The vector store.

    Returns:
        The inventory.
    """
    if not store.exists():
        return DocsResponse()

    table = store.table
    rows = (
        table.search()
        .select(
            [
                "doc_id",
                "title",
                "product_family",
                "doc_type",
                "category",
                "article_url",
                "page_index",
            ]
        )
        .limit(table.count_rows())
        .to_list()
    )

    documents: dict[str, DocumentSummary] = {}
    pages: dict[str, set[Any]] = {}
    for row in rows:
        doc_id = row.get("doc_id") or ""
        summary = documents.get(doc_id)
        if summary is None:
            summary = DocumentSummary(
                doc_id=doc_id,
                title=str(row.get("title") or ""),
                product_family=str(row.get("product_family") or "UNKNOWN"),
                doc_type=str(row.get("doc_type") or "unknown"),
                category=str(row.get("category") or ""),
                article_url=row.get("article_url"),
            )
            documents[doc_id] = summary
            pages[doc_id] = set()
        summary.chunks += 1
        pages[doc_id].add(row.get("page_index"))

    for doc_id, summary in documents.items():
        summary.pages = len(pages[doc_id])

    ordered = sorted(documents.values(), key=lambda d: (d.product_family, d.title))
    return DocsResponse(
        documents=ordered,
        total_documents=len(ordered),
        total_chunks=len(rows),
    )


def create_app(store: Any | None = None) -> Any:
    """Build the FastAPI application.

    Args:
        store: An injected vector store, for tests. Defaults to the configured
            LanceDB index, opened once at startup.

    Returns:
        The configured app.
    """
    from fastapi import FastAPI
    from seeley_rag.retrieve.hybrid import CodeIndex, default_code_index, default_store

    configure_logging()
    settings = get_settings()

    def get_store() -> Any:
        """Return the vector store, opened once for the process."""
        return store if store is not None else default_store()

    @asynccontextmanager
    async def lifespan(_: Any) -> AsyncIterator[None]:
        """Open the index before the first request.

        Opening the table costs ~4.8s against 16,189 rows. Paying it at startup
        rather than inside the first request is the difference between a slow
        deploy and an endpoint that appears broken.

        A missing index is a degraded service, not a failed boot: ``/health``
        must still come up to say what is wrong.
        """
        try:
            log.info("api_ready", extra={"indexed_chunks": get_store().count()})
        except SeeleyRagError as exc:
            log.warning("api_started_without_index", extra={"error": str(exc)})
        yield

    app = FastAPI(
        title="Seeley Installer Assistant",
        version="0.1.0",
        description=(
            "Answers HVAC installer questions from Seeley International's help centre, "
            "cited to the exact manual page."
        ),
        lifespan=lifespan,
    )

    def get_codes() -> CodeIndex:
        """Return the fault-code lookup table."""
        try:
            return default_code_index()
        except SeeleyRagError:
            return CodeIndex([])

    _register_query_routes(app, get_store, get_codes, settings)
    _register_search_routes(app, get_store, get_codes)
    _register_content_routes(app, get_store)
    _register_frontend(app)
    return app


def _register_frontend(app: Any) -> None:
    """Serve the local browser test UI."""
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    static_dir = Path(str(files("seeley_rag.api") / "static"))
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", response_class=FileResponse)
    def frontend() -> Any:
        """Return the browser test UI."""
        return FileResponse(static_dir / "index.html", media_type="text/html")


def _register_query_routes(app: Any, get_store: Any, get_codes: Any, settings: Any) -> None:
    """Register the endpoints that answer questions.

    Args:
        app: The FastAPI application.
        get_store: Accessor for the vector store.
        get_codes: Accessor for the fault-code table.
        settings: Resolved settings.
    """
    from fastapi import HTTPException

    from seeley_rag.generate.answer import generate_answer_response
    from seeley_rag.llm import active_provider, is_configured
    from seeley_rag.retrieve.rerank import rerank_backend

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Report whether the service can actually answer."""
        try:
            active = get_store()
            present = active.exists()
            rows = active.count() if present else 0
        except SeeleyRagError:
            present, rows = False, 0

        configured = is_configured()
        return HealthResponse(
            status="ok" if (present and rows and configured) else "degraded",
            index_present=present,
            indexed_chunks=rows,
            fault_codes=len(get_codes()),
            llm_provider=active_provider(),
            llm_configured=configured,
            rerank_backend=rerank_backend(),
            embedding_model=settings.index.embedding_model,
        )

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        """Answer a question with citations.

        Raises:
            HTTPException: 400 for an empty query, 501 for streaming, 503 when
                the index or provider is unavailable.
        """
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="query must not be empty")
        if request.stream:
            # Returning a whole response to a caller expecting a stream looks
            # like it worked. Refuse until it is actually implemented.
            raise HTTPException(
                status_code=501, detail="streaming is not implemented; set stream=false"
            )
        if not is_configured():
            raise HTTPException(
                status_code=503,
                detail=f"no API key for the configured provider ({active_provider()})",
            )

        try:
            return generate_answer_response(
                request.query,
                top_k=request.top_k,
                store=get_store(),
                code_index=get_codes(),
                product_hint=request.product_hint,
            )
        except SeeleyRagError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc


def _register_search_routes(app: Any, get_store: Any, get_codes: Any) -> None:
    """Register retrieval-only and feedback endpoints.

    Args:
        app: The FastAPI application.
        get_store: Accessor for the vector store.
        get_codes: Accessor for the fault-code table.
    """
    from fastapi import HTTPException

    from seeley_rag.retrieve.hybrid import retrieve

    @app.post("/search", response_model=SearchResponse)
    def search(request: SearchRequest) -> SearchResponse:
        """Return ranked chunks without generating an answer.

        Raises:
            HTTPException: 400 for an empty query, 503 when retrieval fails.
        """
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="query must not be empty")
        # Filters constrain the search rather than trimming its output. They
        # are pushed into both retrieval channels as a pre-filter: applied
        # afterwards to an already-truncated top-k, an explicit filter returns
        # nothing whenever the matches sat below the cut -- a filter that works
        # only when it was not needed. Measured: "fault code" filtered to VRF
        # gave 0 hits post-filtered and 3 pre-filtered.
        try:
            result = retrieve(
                request.query,
                top_k=request.top_k,
                store=get_store(),
                code_index=get_codes(),
                where=build_predicate(request),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except SeeleyRagError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        parsed = result["understanding"]
        hits = [to_hit(row) for row in result["results"]]
        return SearchResponse(
            query=request.query,
            product_family=parsed.product_family,
            model_series=parsed.model_series,
            fault_codes=parsed.fault_codes,
            intent=parsed.intent,
            hits=hits,
        )

    @app.post("/feedback", response_model=FeedbackResponse)
    def feedback(request: FeedbackRequest) -> FeedbackResponse:
        """Record feedback against a ``query_id`` from ``/ask``."""
        # Imported here rather than at registration so the path is read when the
        # request runs, which is what lets tests redirect the data tree.
        from seeley_rag.generate.answer import log_query
        from seeley_rag.paths import FEEDBACK_LOG_PATH

        log_query(
            {
                "query_id": request.query_id,
                "rating": request.rating,
                "comment": request.comment,
            },
            FEEDBACK_LOG_PATH,
        )
        log.info("feedback", extra={"query_id": request.query_id, "rating": request.rating})
        return FeedbackResponse(query_id=request.query_id)


def _register_content_routes(app: Any, get_store: Any) -> None:
    """Register the endpoints that serve corpus content.

    Args:
        app: The FastAPI application.
        get_store: Accessor for the vector store.
    """
    from fastapi import HTTPException
    from fastapi.responses import FileResponse

    @app.get("/docs-inventory", response_model=DocsResponse)
    def docs() -> DocsResponse:
        """Return the corpus inventory.

        Mounted at ``/docs-inventory`` rather than the plan's ``/docs``: FastAPI
        serves its Swagger UI at ``/docs``, and taking that path would remove
        the API's own documentation from an integration seam whose whole purpose
        is being integrated against.
        """
        try:
            return build_inventory(get_store())
        except SeeleyRagError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    # Annotated `Any`, not `Response`: `from __future__ import annotations` makes
    # every annotation a string, and FastAPI cannot resolve "Response" when it
    # builds the OpenAPI schema -- which silently breaks /openapi.json, the one
    # artefact the .NET integration is generated from.
    @app.get("/pages/{doc_id}/{page_index}.png", response_class=FileResponse)
    def page_image(doc_id: str, page_index: int) -> Any:
        """Serve a rendered page image.

        Raises:
            HTTPException: 400 for a malformed id, 404 when the image is absent.
        """
        try:
            path = page_image_path(doc_id, page_index)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not path.is_file():
            raise HTTPException(status_code=404, detail="page image not found")
        return FileResponse(path, media_type="image/png")


def get_app() -> Any:
    """Build the application for an ASGI server.

    ``uvicorn seeley_rag.api.main:get_app --factory``

    Returns:
        The configured app.
    """
    return create_app()
