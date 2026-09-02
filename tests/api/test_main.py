"""Stage 7 API.

build-plan section 9. This is the integration seam to .NET later, so the shapes
here are a contract and the tests read like one.

Nothing touches the network or a model: the store is a fake, and ``/ask`` is
exercised through an injected client. ``TestClient`` speaks ASGI in-process.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi", reason="Stage 7 tests need fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from seeley_rag.api.main import build_predicate, create_app, page_image_path, to_hit  # noqa: E402
from seeley_rag.api.schemas import SearchRequest  # noqa: E402


def row(chunk_id: str, **overrides: Any) -> dict[str, Any]:
    """Build a retrieval result row.

    Args:
        chunk_id: Identifier.
        **overrides: Fields to replace.

    Returns:
        The row.
    """
    base: dict[str, Any] = {
        "chunk_id": chunk_id,
        "doc_id": "d" * 64,
        "text": "Breadcrumb > Path\n\nSet the flame sensor gap to 4-6 mm.",
        "title": "TQ Service Guide",
        "page_label": "42",
        "page_image": "images/p42.png",
        "article_url": "https://example.invalid/article",
        "source_url": "https://example.invalid/doc",
        "product_family": "DGH",
        "doc_type": "service_guide",
        "kind": "prose",
        "is_table": False,
        "model_series": ["TQ"],
        "fault_codes": ["FC07"],
        "content_stream": "pdf",
        "page_index": 41,
        "category": "Ducted Gas Heating",
    }
    base.update(overrides)
    return base


class FakeTable:
    """A LanceDB-shaped table over fixed rows."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def count_rows(self) -> int:
        """Row count."""
        return len(self._rows)

    def search(self, *args: Any, **kwargs: Any) -> FakeTable:
        """Begin a query."""
        return self

    def select(self, columns: list[str]) -> FakeTable:
        """Project columns; ignored, the fake returns whole rows."""
        return self

    def limit(self, n: int) -> FakeTable:
        """Limit; ignored."""
        return self

    def to_list(self) -> list[dict[str, Any]]:
        """Materialise."""
        return self._rows


class FakeStore:
    """A vector store over fixed rows, honouring the `where` pre-filter."""

    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self._rows = rows if rows is not None else [row("c1"), row("c2")]
        self.wheres: list[str | None] = []

    def exists(self) -> bool:
        """The table is present."""
        return True

    def count(self) -> int:
        """Row count."""
        return len(self._rows)

    @property
    def table(self) -> FakeTable:
        """The underlying table."""
        return FakeTable(self._rows)

    def _filtered(self, where: str | None) -> list[dict[str, Any]]:
        """Apply a very small subset of SQL, enough to prove it was passed on."""
        self.wheres.append(where)
        if not where:
            return self._rows
        rows = self._rows
        for clause in where.split(" AND "):
            field, _, value = clause.partition(" = ")
            wanted = value.strip("'")
            if wanted in {"true", "false"}:
                rows = [r for r in rows if str(r.get(field.strip())).lower() == wanted]
            else:
                rows = [r for r in rows if r.get(field.strip()) == wanted]
        return rows

    def search_dense(
        self, vector: Any, top_k: int, where: str | None = None
    ) -> list[dict[str, Any]]:
        """Dense channel."""
        return self._filtered(where)[:top_k]

    def search_bm25(self, query: str, top_k: int, where: str | None = None) -> list[dict[str, Any]]:
        """BM25 channel."""
        return self._filtered(where)[:top_k]


class FakeEmbedder:
    """Returns a fixed vector."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """One vector per text."""
        return [[0.0, 1.0] for _ in texts]


@pytest.fixture
def store() -> FakeStore:
    """A store with one DGH and one EVAP chunk.

    Returns:
        The store.
    """
    return FakeStore(
        [
            row("c1"),
            row(
                "c2",
                product_family="EVAP",
                doc_type="installation",
                title="Braemar Cooler Manual",
                is_table=True,
                kind="table",
            ),
        ]
    )


@pytest.fixture
def client(store: FakeStore, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """A test client over an app wired to the fake store.

    Args:
        store: The fake store.
        monkeypatch: pytest's patcher.

    Returns:
        The client.
    """
    from seeley_rag.retrieve import hybrid

    monkeypatch.setattr(hybrid, "default_embedder", lambda: FakeEmbedder())
    monkeypatch.setattr(hybrid, "default_code_index", lambda: hybrid.CodeIndex([]))
    return TestClient(create_app(store=store))


class TestHealth:
    """Each dependency reported separately, because each fails differently."""

    def test_frontend_is_served_at_root(self, client: TestClient) -> None:
        """The browser test UI lives beside the API."""
        response = client.get("/")
        assert response.status_code == 200
        assert "Seeley Installer Assistant" in response.text

    def test_health_reports_the_index(self, client: TestClient) -> None:
        """An empty index and a missing key are different outages."""
        body = client.get("/health").json()
        assert body["index_present"] is True
        assert body["indexed_chunks"] == 2
        assert body["embedding_model"] == "text-embedding-3-large"

    def test_health_names_the_provider_and_backend(self, client: TestClient) -> None:
        """A caller reading numbers needs to know what produced them."""
        body = client.get("/health").json()
        assert body["llm_provider"] in {"openai", "anthropic"}
        assert body["rerank_backend"] in {"cohere", "llm", "identity"}

    def test_health_is_degraded_without_an_index(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The service must come up to say what is wrong."""

        class Empty(FakeStore):
            def exists(self) -> bool:
                return False

            def count(self) -> int:
                return 0

        with TestClient(create_app(store=Empty([]))) as empty:
            assert empty.get("/health").json()["status"] == "degraded"


class TestSearch:
    """Retrieval without generation -- the debug endpoint."""

    def test_search_returns_hits_and_understanding(self, client: TestClient) -> None:
        """Both halves: what was found, and what the query was taken to mean."""
        body = client.post("/search", json={"query": "TQ heater FC7", "top_k": 5}).json()
        assert body["product_family"] == "DGH"
        assert body["fault_codes"] == ["FC07"]
        assert body["hits"]

    def test_hits_carry_the_rerank_backend(self, client: TestClient) -> None:
        """A caller must be able to tell a reranked list from an unreranked one."""
        body = client.post("/search", json={"query": "TQ FC7"}).json()
        assert body["hits"][0]["rerank_backend"] in {"cohere", "llm", "identity"}

    def test_an_empty_query_is_rejected(self, client: TestClient) -> None:
        """Whitespace is not a question."""
        assert client.post("/search", json={"query": "   "}).status_code == 400

    def test_top_k_is_bounded_by_the_schema(self, client: TestClient) -> None:
        """An unbounded top_k is a denial-of-service knob."""
        assert client.post("/search", json={"query": "x", "top_k": 5000}).status_code == 422


class TestFilters:
    """Filters constrain the search; they do not trim its output."""

    def test_family_filter_reaches_the_store(self, client: TestClient, store: FakeStore) -> None:
        """Pushed down as a pre-filter, not applied to the results.

        Post-filtering a top-k list returns nothing whenever the matches sit
        below the cut -- measured on the real index, "fault code" filtered to
        VRF gave 0 hits post-filtered and 3 pre-filtered.
        """
        body = client.post("/search", json={"query": "cooler", "product_family": "EVAP"}).json()
        assert all(hit["product_family"] == "EVAP" for hit in body["hits"])
        assert any("product_family = 'EVAP'" in (w or "") for w in store.wheres)

    def test_doc_type_filter(self, client: TestClient) -> None:
        """The second lexicon-controlled field."""
        body = client.post("/search", json={"query": "x", "doc_type": "installation"}).json()
        assert body["hits"]
        assert all(hit["product_family"] == "EVAP" for hit in body["hits"])

    def test_table_filter(self, client: TestClient) -> None:
        """Fault-code tables are different evidence from prose about them."""
        body = client.post("/search", json={"query": "x", "is_table": True}).json()
        assert body["hits"]
        assert all(hit["kind"] == "table" for hit in body["hits"])

    def test_no_filters_sends_no_predicate(self, client: TestClient, store: FakeStore) -> None:
        """An unfiltered search must not pay for a predicate."""
        client.post("/search", json={"query": "x"})
        assert store.wheres[-1] is None

    def test_a_filter_value_that_is_not_an_identifier_is_rejected(self, client: TestClient) -> None:
        """The predicate is concatenated into a query, so the shape is enforced.

        These fields are lexicon identifiers; anything else is a caller error.
        """
        response = client.post(
            "/search", json={"query": "x", "product_family": "DGH'; DROP TABLE chunks--"}
        )
        assert response.status_code == 400
        assert "plain identifier" in response.json()["detail"]

    @pytest.mark.parametrize(
        ("request_kwargs", "expected"),
        [
            ({}, None),
            ({"product_family": "DGH"}, "product_family = 'DGH'"),
            ({"is_table": True}, "is_table = true"),
            ({"is_table": False}, "is_table = false"),
            (
                {"product_family": "DGH", "doc_type": "service_guide"},
                "product_family = 'DGH' AND doc_type = 'service_guide'",
            ),
        ],
    )
    def test_predicate_shapes(self, request_kwargs: dict[str, Any], expected: str | None) -> None:
        """The generated SQL, asserted literally."""
        assert build_predicate(SearchRequest(query="x", **request_kwargs)) == expected


class TestAsk:
    """The endpoint the installer actually hits."""

    def test_ask_returns_a_query_id(
        self,
        temp_data_root: Path,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """`/feedback` takes one, and v1 of the plan had nothing producing it."""
        # `answer.py` does `from seeley_rag import llm` and calls
        # `llm.complete_json`, so patching the llm module reaches it. Importing
        # `seeley_rag.generate.answer` by name does not: the package re-exports
        # the *function* `answer`, which shadows the submodule under
        # `import a.b.c as x` attribute resolution.
        from seeley_rag import llm as llm_module

        monkeypatch.setattr(
            llm_module,
            "complete_json",
            lambda **kwargs: {"answer": "Set the gap to 4-6 mm [1].", "confidence": "high"},
        )
        body = client.post("/ask", json={"query": "TQ FC7", "top_k": 2}).json()
        assert body["query_id"].startswith("q_")
        assert body["citations"]
        assert "4-6 mm" in body["answer"]

    def test_streaming_is_refused_not_ignored(self, client: TestClient) -> None:
        """Returning a whole response to a stream request looks like it worked."""
        response = client.post("/ask", json={"query": "x", "stream": True})
        assert response.status_code == 501
        assert "streaming" in response.json()["detail"]

    def test_an_empty_query_is_rejected(self, client: TestClient) -> None:
        """Whitespace is not a question."""
        assert client.post("/ask", json={"query": " "}).status_code == 400


class TestFeedback:
    """Joined back to the answer by query_id."""

    def test_feedback_is_recorded(self, temp_data_root: Path, client: TestClient) -> None:
        """Written as JSONL beside the query log, not into it."""
        response = client.post(
            "/feedback", json={"query_id": "q_abc", "rating": "down", "comment": "wrong page"}
        )
        assert response.status_code == 200
        assert response.json() == {"query_id": "q_abc", "recorded": True}

        written = (temp_data_root / "reports" / "feedback.jsonl").read_text(encoding="utf-8")
        record = json.loads(written.strip())
        assert record["query_id"] == "q_abc"
        assert record["rating"] == "down"

    def test_an_invalid_rating_is_rejected(self, client: TestClient) -> None:
        """Two values, so the eval can count them."""
        assert (
            client.post("/feedback", json={"query_id": "q_1", "rating": "sideways"}).status_code
            == 422
        )

    def test_query_id_is_required(self, client: TestClient) -> None:
        """Feedback that cannot be joined to an answer is not feedback."""
        assert client.post("/feedback", json={"rating": "up"}).status_code == 422


class TestPageImages:
    """The one place a caller influences a filesystem read."""

    def test_a_valid_path_resolves(self, tmp_path: Path) -> None:
        """The happy path, with the name rebuilt rather than taken."""
        doc = "a" * 64
        (tmp_path / doc).mkdir(parents=True)
        (tmp_path / doc / "0041.png").write_bytes(b"png")
        assert page_image_path(doc, 41, root=tmp_path).name == "0041.png"

    def test_article_ids_drop_the_prefix(self, tmp_path: Path) -> None:
        """`article:123` is not a legal directory name on Windows."""
        assert page_image_path("article:123", 0, root=tmp_path).parent.name == "123"

    @pytest.mark.parametrize(
        "doc_id",
        ["../../etc/passwd", "..", "a/b", "not-a-hash", "", "a" * 63, "A" * 64],
    )
    def test_malformed_ids_are_refused(self, doc_id: str, tmp_path: Path) -> None:
        """Rejected on shape, before any path is built."""
        with pytest.raises(ValueError):
            page_image_path(doc_id, 0, root=tmp_path)

    def test_a_negative_index_is_refused(self, tmp_path: Path) -> None:
        """Pages are 0-based and there is no page -1."""
        with pytest.raises(ValueError):
            page_image_path("a" * 64, -1, root=tmp_path)

    def test_a_missing_image_is_404(self, client: TestClient) -> None:
        """A well-formed id for a page that was never rendered."""
        assert client.get(f"/pages/{'a' * 64}/0.png").status_code == 404

    def test_a_malformed_id_is_400(self, client: TestClient) -> None:
        """Distinguished from "not found", because the caller can fix it."""
        assert client.get("/pages/notahash/0.png").status_code == 400


class TestInventory:
    """The corpus inventory, one row per document."""

    def test_documents_are_grouped(self, client: TestClient) -> None:
        """Chunks roll up to documents, and pages are counted distinctly."""
        body = client.get("/docs-inventory").json()
        assert body["total_chunks"] == 2
        assert body["total_documents"] >= 1
        assert all("product_family" in doc for doc in body["documents"])

    def test_swagger_ui_is_still_at_docs(self, client: TestClient) -> None:
        """The inventory is at /docs-inventory precisely so this keeps working.

        Taking /docs would remove the API's own documentation from an
        integration seam whose whole purpose is being integrated against.
        """
        assert client.get("/docs").status_code == 200
        assert client.get("/openapi.json").status_code == 200


def test_hit_conversion_prefers_a_page_range() -> None:
    """A merged multi-page table must report the span it covers."""
    assert to_hit(row("c1", page_range="42-44")).page_label == "42-44"


def test_hit_conversion_adds_a_page_url() -> None:
    """The frontend needs an API URL, not a repo-relative image path."""
    hit = to_hit(row("c1", page_index=41))
    assert hit.page_url == f"/pages/{'d' * 64}/41.png"


def test_hit_conversion_uses_remote_page_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hosted demos serve page images directly from object storage."""
    from seeley_rag.settings import get_settings

    monkeypatch.setenv("PAGE_IMAGE_BASE_URL", "https://assets.example.test/pages")
    get_settings.cache_clear()
    try:
        hit = to_hit(row("c1", page_index=41))
        assert hit.page_url == f"https://assets.example.test/pages/{'d' * 64}/0041.png"
    finally:
        get_settings.cache_clear()
