"""Stage 3 data model and JSONL I/O.

build-plan.md sections 5 and 6.

:class:`Chunk` is the ``chunks.jsonl`` row schema and, field for field, the
LanceDB table schema of build-plan section 6 -- one shape carried from chunking
through indexing, so the store never has to be told what a chunk looks like.

Named ``base.py`` to mirror :mod:`seeley_rag.parse.base`, which plays the same
role for Stage 2.

Two fields exist purely to make re-indexing cheap, and they are what let vision
transcriptions be folded in later without a rebuild:

* ``chunk_id`` is **deterministic** -- derived from document, page and ordinal,
  never from a counter or a UUID. Re-chunking the same page produces the same
  id, so the store can upsert by id instead of being dropped and refilled.
* ``content_hash`` is ``sha256`` of the exact text that will be embedded,
  breadcrumb prefix included. It is the embedding cache key (section 6) and the
  change detector: an equal hash means the vector on disk is still correct, so a
  re-index after a partial change embeds only what actually moved.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator, Literal

from pydantic import BaseModel, ConfigDict, Field

from seeley_rag.exceptions import ParseError
from seeley_rag.parse.base import ContentStream, LabelSource, Tier

#: What a chunk's body was built from.
#:
#: ``prose`` -- running text from a page.
#: ``table`` -- a detected table, kept atomic (build-plan section 5.1, rule 3).
ChunkKind = Literal["prose", "table"]


def content_hash(text: str) -> str:
    """Hash the exact text that will be embedded.

    Args:
        text: Final chunk text, breadcrumb prefix included.

    Returns:
        Hex SHA-256. Doubles as the embedding cache key.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_chunk_id(doc_id: str, page_index: int | None, ordinal: int) -> str:
    """Build a deterministic chunk id.

    Determinism is the whole basis of incremental indexing: the same page
    chunked twice must yield the same ids, or an upsert becomes a duplicate.

    Args:
        doc_id: Document SHA-256, or ``article:{id}``.
        page_index: 0-based page index; ``None`` for articles.
        ordinal: 0-based position of this chunk within the page.

    Returns:
        e.g. ``a1b2c3...:p41:c00`` or ``article:47001247475:pna:c00``.
    """
    page_part = "pna" if page_index is None else f"p{page_index}"
    return f"{doc_id}:{page_part}:c{ordinal:02d}"


class Chunk(BaseModel):
    """One indexable unit. This is the ``chunks.jsonl`` row schema.

    Attributes:
        chunk_id: Deterministic id -- see :func:`make_chunk_id`.
        doc_id: Owning document.
        text: The **final** text, breadcrumb prefix included. This is what gets
            embedded and what the generator quotes from, so nothing may be
            prepended downstream without changing ``content_hash`` too.
        content_hash: SHA-256 of ``text``; embedding cache key and change
            detector.
        token_count: Measured with the embedding model's own tokeniser.
        kind: Prose or table.
        page_index: 0-based index within the document. Internal only.
        page_label: The **printed** page number -- what a citation displays.
        label_source: How the label was obtained. ``index`` means it is a guess;
            the eval must be able to exclude those rather than score against
            them (build-plan section 4.5).
        page_range: ``"42-44"`` when a table was merged across pages and the
            labels support saying so. The chunk stays anchored to the first page
            for citation.
        page_span: Pages the chunk's content covers -- 1 for everything except a
            merged table. Accurate even when ``page_range`` is ``None`` because
            the labels were guesses, so the eval can exclude a multi-page chunk
            from single-page scoring rather than mark it wrong.
        page_image: Repo-relative path to the rendered page PNG. Populated so
            the generator can surface a diagram alongside the answer.
        fault_codes: Codes found in this chunk, normalised. Retrieval pins
            chunks whose codes match the query.
        source_article_ids: Every article linking the owning document.
        product_family: From the lexicon. Soft-boosted at retrieval, never
            hard-filtered on an inferred value.
        model_series: Model codes from the title.
        doc_type: From the folder name.
        title: Document or article title.
        source_url: Attachment URL for PDFs, article URL for articles.
        article_url: The article to open for verification.
        category: Solution category.
        folder: Folder name.
        tier: Parsing tier of the source page.
        content_stream: Which ingestion path produced the source page.
        needs_vision: Whether the source page still awaits transcription. Kept
            on the chunk so a later vision pass can find exactly the rows to
            replace without re-reading ``pages.jsonl``.
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    chunk_id: str
    doc_id: str
    text: str
    content_hash: str = ""
    token_count: int = 0
    kind: ChunkKind = "prose"

    page_index: int | None = None
    page_label: str | None = None
    label_source: LabelSource = "none"
    page_range: str | None = None
    page_span: int = 1
    page_image: str | None = None

    fault_codes: list[str] = Field(default_factory=list)
    source_article_ids: list[str] = Field(default_factory=list)

    product_family: str = "UNKNOWN"
    model_series: list[str] = Field(default_factory=list)
    doc_type: str = "unknown"
    title: str = ""
    source_url: str = ""
    article_url: str | None = None
    category: str = ""
    folder: str = ""
    tier: Tier = "plain_text"
    content_stream: ContentStream = "pdf"
    needs_vision: bool = False

    @property
    def is_table(self) -> bool:
        """Whether this chunk is an atomic table."""
        return self.kind == "table"

    def finalise(self, token_count: int | None = None) -> Chunk:
        """Stamp ``content_hash`` and ``token_count`` from the current text.

        Called once, after ``text`` is complete. Doing it here rather than at
        construction keeps the invariant in one place: the hash always describes
        the text that will actually be embedded.

        Args:
            token_count: Pre-computed count, so text the caller has already
                measured is not tokenised twice.

        Returns:
            ``self``, mutated.
        """
        from seeley_rag.chunk.tokens import count_tokens

        self.content_hash = content_hash(self.text)
        self.token_count = count_tokens(self.text) if token_count is None else token_count
        return self


class FaultCode(BaseModel):
    """One row of the exact-lookup fault-code table.

    build-plan section 5.3. Vector search is bad at codes -- ``E:04`` and
    ``E:05`` are neighbours in embedding space and opposites in meaning -- so
    codes get a lookup table that a query hits *before* retrieval runs.

    Attributes:
        code: The code as printed, e.g. ``E:04``.
        code_key: Normalised for lookup: upper-cased, separators removed, so
            ``E:04``, ``E 04`` and ``E-04`` all resolve to ``E04``.
        meaning: The code's meaning, taken verbatim from the source row or
            sentence. Never paraphrased -- a reworded fault description is a
            wrong answer with a citation attached.
        evidence: The surrounding text the meaning was read from, kept for
            audit.
        product_family: Which product this code belongs to. The same string
            means different things across families, so lookups are scoped.
        model_series: Model codes of the source document.
        doc_id: Owning document.
        chunk_id: The chunk to pin into context on a hit.
        page_label: Printed page for the citation.
        title: Document title for the citation.
        source_url: Attachment or article URL.
        article_url: The article to open for verification.
        in_table: Whether the code was found inside a detected table. Table
            evidence is far stronger than a prose mention.
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    code: str
    code_key: str
    meaning: str = ""
    evidence: str = ""
    product_family: str = "UNKNOWN"
    model_series: list[str] = Field(default_factory=list)
    doc_id: str = ""
    chunk_id: str = ""
    page_label: str | None = None
    title: str = ""
    source_url: str = ""
    article_url: str | None = None
    in_table: bool = False


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------


class JsonlWriter:
    """Append-safe JSONL writer, flushed per row.

    Mirrors :class:`seeley_rag.parse.base.PagesWriter`. Chunking the full corpus
    is a minutes-long run; buffered rows lost to a kill would have to be redone.

    Args:
        path: Destination file.
        overwrite: Truncate instead of appending.

    Attributes:
        count: Rows written.
    """

    def __init__(self, path: Path, overwrite: bool = True) -> None:
        self.path = path
        self.overwrite = overwrite
        self.count = 0
        self._handle: Any = None

    def __enter__(self) -> JsonlWriter:
        """Open the file for writing."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open(
            "w" if self.overwrite else "a", encoding="utf-8", newline="\n"
        )
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the file."""
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def write(self, record: BaseModel) -> None:
        """Append one record.

        Args:
            record: A pydantic model to serialise.

        Raises:
            ParseError: If the writer is not open.
        """
        if self._handle is None:
            raise ParseError(f"{type(self).__name__} must be used as a context manager.")
        self._handle.write(json.dumps(record.model_dump(mode="json"), ensure_ascii=False) + "\n")
        self._handle.flush()
        self.count += 1

    def write_all(self, records: list[Any]) -> int:
        """Append several records.

        Args:
            records: Models to serialise.

        Returns:
            How many were written.
        """
        for record in records:
            self.write(record)
        return len(records)


def _read_jsonl(path: Path, model: type[BaseModel], stage_hint: str) -> Iterator[Any]:
    """Stream a JSONL file into pydantic models.

    Args:
        path: Source file.
        model: Model to validate each row against.
        stage_hint: Command to suggest when the file is missing.

    Yields:
        One validated model per row.

    Raises:
        ParseError: If the file is missing or a row is invalid.
    """
    if not path.exists():
        raise ParseError(f"No file at {path}. Run `{stage_hint}` first.")
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield model.model_validate_json(stripped)
            except Exception as exc:  # noqa: BLE001 - pydantic and json both raise
                raise ParseError(f"{path}:{number} is not a valid row: {exc}") from exc


def read_chunks(path: Path | None = None) -> Iterator[Chunk]:
    """Stream ``chunks.jsonl``.

    Args:
        path: Source. Defaults to ``data/02_processed/chunks.jsonl``.

    Yields:
        Each chunk in file order.

    Raises:
        ParseError: If the file is missing or a row is invalid.
    """
    from seeley_rag.paths import CHUNKS_PATH

    yield from _read_jsonl(path or CHUNKS_PATH, Chunk, "python scripts/04_index.py")


def read_codes(path: Path | None = None) -> Iterator[FaultCode]:
    """Stream ``codes.jsonl``.

    Args:
        path: Source. Defaults to ``data/02_processed/codes.jsonl``.

    Yields:
        Each fault-code row in file order.

    Raises:
        ParseError: If the file is missing or a row is invalid.
    """
    from seeley_rag.paths import CODES_PATH

    yield from _read_jsonl(path or CODES_PATH, FaultCode, "python scripts/04_index.py")


def chunk_hashes(path: Path | None = None) -> dict[str, str]:
    """Return ``chunk_id -> content_hash`` for chunks already on disk.

    This is the incremental-index primitive. Comparing a fresh chunking against
    this map partitions the corpus into unchanged, changed and new, so a
    re-index -- including one that folds in vision transcriptions for the 3,459
    pages awaiting them -- embeds only what actually moved.

    Args:
        path: Source. Defaults to ``data/02_processed/chunks.jsonl``.

    Returns:
        The mapping. Empty when the file does not exist.
    """
    from seeley_rag.paths import CHUNKS_PATH

    resolved = path or CHUNKS_PATH
    if not resolved.exists():
        return {}
    hashes: dict[str, str] = {}
    with resolved.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
                hashes[row["chunk_id"]] = row["content_hash"]
            except (json.JSONDecodeError, KeyError):
                continue
    return hashes
