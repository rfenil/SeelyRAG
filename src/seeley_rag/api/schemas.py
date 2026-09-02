"""Stage 7 -- request and response models.

build-plan.md section 9. This is the integration seam to .NET later, so the
shapes here are a contract: additive changes only, once anything consumes them.

Named ``schemas.py`` rather than the build plan's ``models.py`` so it cannot be
confused with ``config/models.yaml``, which is the product lexicon.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """A question from an installer.

    Attributes:
        query: The question.
        product_hint: Optional product family hint.
        top_k: Chunks to retrieve.
        stream: Whether to stream the answer.
    """

    query: str
    product_hint: str | None = None
    top_k: int = 8
    stream: bool = False


class Citation(BaseModel):
    """One citation, resolving to a page image and the source article.

    Attributes:
        n: Citation number as it appears inline in the answer.
        title: Document or article title.
        page_label: The **printed** page number (build-plan section 4.5).
        doc_url: The attachment URL.
        article_url: The Freshdesk article the user can open to verify.
        page_image: Rendered page image path.
        page_url: API URL for the rendered page image.
        snippet: The supporting text.
    """

    n: int
    title: str
    page_label: str | None = None
    doc_url: str | None = None
    article_url: str | None = None
    page_image: str | None = None
    page_url: str | None = None
    snippet: str = ""


class AskResponse(BaseModel):
    """A cited answer.

    Attributes:
        query_id: Correlation ID; ``/feedback`` takes this.
        answer: The generated answer with inline ``[n]`` markers.
        citations: Citations resolving those markers.
        confidence: Model-reported confidence.
        product_family: Inferred product family.
        latency_ms: End-to-end latency.
    """

    query_id: str
    answer: str
    citations: list[Citation] = []
    confidence: str = "unknown"
    product_family: str | None = None
    latency_ms: int = 0


class SearchRequest(BaseModel):
    """A retrieval-only request, for debugging what the cascade returns.

    Attributes:
        query: The question or search text.
        product_family: Restrict to one family. ⚠ A **hard** filter, unlike the
            soft boost retrieval applies to an *inferred* family -- this one was
            asked for explicitly, so honouring it literally is correct.
        doc_type: Restrict to one document type, e.g. ``service_guide``.
        is_table: Restrict to table chunks, or exclude them.
        top_k: Chunks to return.
    """

    query: str
    product_family: str | None = None
    doc_type: str | None = None
    is_table: bool | None = None
    top_k: int = Field(default=8, ge=1, le=50)


class SearchHit(BaseModel):
    """One retrieved chunk, with the scoring that produced it.

    Attributes:
        chunk_id: The chunk's deterministic id.
        doc_id: Owning document.
        title: Document or article title.
        page_label: The printed page number.
        page_image: Rendered page image path.
        page_url: API URL for the rendered page image.
        article_url: The article to open to verify.
        product_family: Resolved family.
        kind: ``prose`` or ``table``.
        score: The reranked score.
        boosts: Which metadata boosts fired.
        rerank_backend: Which backend ordered this -- ``cohere``, ``llm`` or
            ``identity``. Present so a caller can tell a reranked list from an
            unreranked one rather than assuming.
        text: The chunk text.
    """

    chunk_id: str
    doc_id: str = ""
    title: str = ""
    page_label: str | None = None
    page_image: str | None = None
    page_url: str | None = None
    article_url: str | None = None
    product_family: str = "UNKNOWN"
    kind: str = "prose"
    score: float = 0.0
    boosts: list[str] = []
    rerank_backend: str = "identity"
    text: str = ""


class SearchResponse(BaseModel):
    """Retrieval results plus what the query was understood to mean.

    Attributes:
        query: The query as received.
        product_family: Inferred family.
        model_series: Model codes found in the query.
        fault_codes: Normalised codes found in the query.
        intent: What the installer appears to want.
        hits: The ranked chunks.
    """

    query: str
    product_family: str = "UNKNOWN"
    model_series: list[str] = []
    fault_codes: list[str] = []
    intent: str = "general"
    hits: list[SearchHit] = []


class FeedbackRequest(BaseModel):
    """Feedback on one answer.

    Attributes:
        query_id: The ``query_id`` ``/ask`` returned. Required -- build-plan
            section 9 flags that v1's ``/feedback`` took one nothing produced.
        rating: ``up`` or ``down``.
        comment: Optional free text.
    """

    query_id: str
    rating: str = Field(pattern="^(up|down)$")
    comment: str | None = None


class FeedbackResponse(BaseModel):
    """Acknowledgement of recorded feedback.

    Attributes:
        query_id: Echoed back.
        recorded: Whether it was written.
    """

    query_id: str
    recorded: bool = True


class DocumentSummary(BaseModel):
    """One document in the corpus inventory.

    Attributes:
        doc_id: Document SHA-256, or ``article:{id}``.
        title: Document or article title.
        product_family: Resolved family.
        doc_type: Resolved document type.
        category: Solution category.
        chunks: Indexed chunks from this document.
        pages: Distinct pages indexed.
        article_url: The article to open.
    """

    doc_id: str
    title: str = ""
    product_family: str = "UNKNOWN"
    doc_type: str = "unknown"
    category: str = ""
    chunks: int = 0
    pages: int = 0
    article_url: str | None = None


class DocsResponse(BaseModel):
    """The corpus inventory.

    Attributes:
        documents: One entry per document.
        total_documents: How many documents are indexed.
        total_chunks: How many chunks are indexed.
    """

    documents: list[DocumentSummary] = []
    total_documents: int = 0
    total_chunks: int = 0


class HealthResponse(BaseModel):
    """Whether the service can actually answer.

    Reports each dependency separately rather than one boolean: an index that is
    present but empty, and a missing API key, are different outages with
    different fixes.

    Attributes:
        status: ``ok`` when every dependency is usable, else ``degraded``.
        index_present: Whether the vector index exists.
        indexed_chunks: Rows in the index.
        fault_codes: Rows in the fault-code lookup table.
        llm_provider: The configured generation provider.
        llm_configured: Whether that provider has a key.
        rerank_backend: Active reranking backend.
        embedding_model: Model the index was built with.
    """

    status: str = "ok"
    index_present: bool = False
    indexed_chunks: int = 0
    fault_codes: int = 0
    llm_provider: str = "openai"
    llm_configured: bool = False
    rerank_backend: str = "identity"
    embedding_model: str = ""
