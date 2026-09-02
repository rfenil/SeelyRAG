"""Stage 6 -- answer synthesis.

build-plan.md section 8, over the retrieval cascade of section 7.

Every citation resolves to a page image **and** a link back to the source
Freshdesk article, so an installer verifies in two taps. That is what earns
trust.

``/ask`` must return ``query_id``; it is logged alongside the query, the
retrieved chunk IDs and the answer. The first week of real queries is worth more
than any synthetic eval.

⚠ **The prompt's rules are checked here, not trusted.**

A system prompt is a request, not a guarantee, and this one is guarding gas and
mains electrical work. Three things are therefore verified in code after the
model has spoken:

* **Citation numbers must resolve.** A ``[9]`` when eight passages were supplied
  is dropped, not rendered -- a citation that resolves to nothing is worse than
  no citation, because it looks verified.
* **Only cited passages become citations.** Listing all eight sources under an
  answer that used two implies corroboration that does not exist.
* **An uncited answer is marked down.** If the model returns claims with no
  citation at all, confidence is forced to ``low``, because the one property
  that makes this system trustworthy is missing.

None of that can make a wrong answer right. It makes a wrong answer *visible*,
which is the most a generation stage can honestly offer.
"""

from __future__ import annotations

import json
import re
import time
import unicodedata
import uuid
from pathlib import Path
from typing import Any, Sequence

from seeley_rag import llm
from seeley_rag.api.schemas import AskResponse, Citation
from seeley_rag.exceptions import ConfigurationError
from seeley_rag.generate.prompts import build_user_message, system_prompt
from seeley_rag.logging_conf import get_logger
from seeley_rag.page_images import page_image_url
from seeley_rag.settings import get_settings

log = get_logger(__name__)

#: Inline citation markers, e.g. ``[1]`` or ``[1, 2]`` or ``[1][2]``.
_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

#: Output cap for one answer. Generous: a fault-finding procedure runs long, and
#: truncating a procedure mid-step is worse than a long answer.
MAX_ANSWER_TOKENS = 2_000

_ASCII_REPLACEMENTS = str.maketrans(
    {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2212": "-",
        "\u00a0": " ",
        "\u202f": " ",
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u2022": "-",
        "\u00b7": "|",
        "\u2192": "->",
        "\u00d7": "x",
    }
)


def new_query_id() -> str:
    """Mint a correlation id for one question.

    ``/feedback`` takes one of these, so it has to exist before the answer is
    returned rather than being derived from it later.

    Returns:
        e.g. ``q_4f3c2a1b8e9d4c7a``.
    """
    return f"q_{uuid.uuid4().hex[:16]}"


def cited_numbers(answer_text: str) -> list[int]:
    """Return the citation numbers referenced in an answer, in order.

    Args:
        answer_text: The generated answer.

    Returns:
        Distinct 1-based numbers, in order of first appearance.
    """
    found: list[int] = []
    for match in _CITATION_RE.finditer(answer_text):
        for part in match.group(1).split(","):
            try:
                number = int(part.strip())
            except ValueError:
                continue
            if number not in found:
                found.append(number)
    return found


def plain_ascii(text: str) -> str:
    """Return display text with Unicode punctuation normalised to ASCII.

    Model output and scraped article titles can contain smart punctuation,
    non-breaking hyphens and emoji. They are noisy in the field UI and in logs,
    so generated answers are normalised before they reach the API response.
    """
    translated = text.translate(_ASCII_REPLACEMENTS)
    normalised = unicodedata.normalize("NFKD", translated)
    encoded = normalised.encode("ascii", "ignore").decode("ascii")
    encoded = re.sub(r"[ \t]+\n", "\n", encoded)
    encoded = re.sub(r"\n{3,}", "\n\n", encoded)
    return encoded.strip()


def normalise_sections(answer_text: str) -> str:
    """Move a misplaced ``Answer:`` heading to the top of the answer.

    The prompt asks for ``Answer:`` as the first line, followed by the sentence
    that answers the question. Models reliably write the sentence first and then
    emit ``Answer:`` above the bullet list -- so the heading ends up labelling
    the values rather than the answer, and the reader meets an unlabelled
    paragraph followed by a section called "Answer" that is not one.

    Four rounds of rewording the prompt did not fix it, which is the point ADR
    0009 already makes: a system prompt is a request. This is the check.

    Args:
        answer_text: The generated answer.

    Returns:
        The answer with the heading first, or unchanged when there is nothing
        to move.
    """
    lines = answer_text.split("\n")
    for index, line in enumerate(lines):
        if line.strip().lower() != "answer:":
            continue
        if index == 0:
            return answer_text
        if not any(before.strip() for before in lines[:index]):
            # Only blank lines above it: drop them rather than reorder.
            return "\n".join(lines[index:])
        del lines[index]
        moved = "\n".join(["Answer:", *lines])
        return re.sub(r"\n{3,}", "\n\n", moved).strip()
    return answer_text


def strip_inline_citations(answer_text: str) -> str:
    """Remove inline citation markers from text shown to the user.

    The model still has to include markers because they are how ``assemble``
    determines which source cards belong with the answer. After those cards are
    built, the markers are just visual noise in the answer body.
    """
    cleaned = _CITATION_RE.sub("", answer_text)
    cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n[ \t]+", "\n", cleaned)
    return cleaned.strip()


def _strip_unresolvable(answer_text: str, valid: set[int]) -> tuple[str, list[int]]:
    """Remove citation markers that do not resolve to a supplied passage.

    A marker pointing at nothing looks like verification and is not, so it is
    removed from the prose rather than left for the reader to chase.

    Args:
        answer_text: The generated answer.
        valid: Citation numbers that were actually supplied.

    Returns:
        The answer with unresolvable markers removed, and the numbers dropped.
    """
    dropped: list[int] = []

    def replace(match: re.Match[str]) -> str:
        numbers = [int(p.strip()) for p in match.group(1).split(",") if p.strip().isdigit()]
        keep = [n for n in numbers if n in valid]
        dropped.extend(n for n in numbers if n not in valid)
        if not keep:
            return ""
        return "[" + ", ".join(str(n) for n in keep) + "]"

    cleaned = _CITATION_RE.sub(replace, answer_text)
    # Removing a marker can leave " ." or a double space behind.
    cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip(), dropped


def build_citation(number: int, chunk: dict[str, Any]) -> Citation:
    """Turn a retrieved chunk into a citation.

    Args:
        number: The inline marker this resolves.
        chunk: The retrieved chunk row.

    Returns:
        A citation carrying the page image and the article link.
    """
    body = str(chunk.get("text") or "")
    # Drop the breadcrumb prefix from the snippet: it is metadata the citation
    # already shows in structured form, and it would crowd out the evidence.
    if "\n\n" in body:
        body = body.split("\n\n", 1)[1]
    snippet = plain_ascii(" ".join(body.split()))[:300]

    doc_id = str(chunk.get("doc_id") or "")
    page_index = chunk.get("page_index")
    page_url = page_image_url(doc_id, page_index) if isinstance(page_index, int) else None

    return Citation(
        n=number,
        title=plain_ascii(str(chunk.get("title") or "Untitled")),
        page_label=chunk.get("page_range") or chunk.get("page_label"),
        doc_url=chunk.get("source_url"),
        article_url=chunk.get("article_url"),
        page_image=chunk.get("page_image"),
        page_url=page_url,
        snippet=snippet,
    )


def log_query(record: dict[str, Any], path: Path | None = None) -> None:
    """Append one query record to the JSONL query log.

    build-plan section 9: the first week of real queries is worth more than any
    synthetic eval, so this is written from the first answer rather than added
    when someone thinks to.

    Args:
        record: The record to write.
        path: Destination. Defaults to ``data/reports/queries.jsonl``.
    """
    from seeley_rag.paths import QUERY_LOG_PATH

    resolved = path or QUERY_LOG_PATH
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        with resolved.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        # Never fail an answer because the log could not be written.
        log.warning("query_log_write_failed", extra={"error": str(exc)})


def _unanswered(
    query_id: str,
    reason: str,
    product_family: str | None,
    latency_ms: int,
) -> AskResponse:
    """Build the response for a question the corpus cannot answer.

    Saying so is a correct answer. Section 8 requires it explicitly, because the
    alternative -- a fluent answer from the model's general HVAC knowledge -- is
    indistinguishable from a real one to the installer reading it.

    Args:
        query_id: Correlation id.
        reason: What to tell the user.
        product_family: Inferred family, if any.
        latency_ms: Elapsed time.

    Returns:
        A cited-nothing response.
    """
    # Strip inline markers. A declined answer presents no sources, so leaving
    # "[1], [2]" in the prose would point at a list that is deliberately empty
    # -- and imply the passages supported something they did not.
    cleaned = _CITATION_RE.sub("", reason)
    cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
    cleaned = plain_ascii(cleaned)

    return AskResponse(
        query_id=query_id,
        answer=cleaned or reason,
        citations=[],
        confidence="low",
        product_family=product_family,
        latency_ms=latency_ms,
    )


def answer(
    query: str,
    chunks: Sequence[dict[str, Any]] | None = None,
    pinned: Sequence[Any] = (),
    product_family: str | None = None,
    eval_id: str | None = None,
    client: Any | None = None,
    model: str | None = None,
    top_k: int | None = None,
    log_path: Path | None = None,
    **retrieve_kwargs: Any,
) -> AskResponse:
    """Generate a cited answer from retrieved context.

    Args:
        query: The installer's question.
        chunks: Pre-retrieved chunks. When ``None``, the Stage 5 cascade runs.
        pinned: Pinned fault-code rows, when chunks are supplied directly.
        product_family: Inferred family, when chunks are supplied directly.
        eval_id: SME evaluation case id, when this answer was generated by Stage 8.
        client: Injected SDK client, for tests.
        model: Override the configured generation model.
        top_k: Chunks to retrieve, when retrieving here.
        log_path: Override the query-log destination.
        **retrieve_kwargs: Passed through to the cascade.

    Returns:
        The cited answer, with a ``query_id`` that ``/feedback`` can take.
    """
    started = time.monotonic()
    query_id = new_query_id()

    if chunks is None:
        from seeley_rag.retrieve.hybrid import retrieve

        result = retrieve(query, top_k=top_k, **retrieve_kwargs)
        chunks = result["results"]
        pinned = result["pinned_codes"]
        understanding = result["understanding"]
        product_family = understanding.product_family
    chunks = list(chunks)

    def elapsed() -> int:
        """Milliseconds since the request began."""
        return int((time.monotonic() - started) * 1000)

    if not chunks and not pinned:
        response = _unanswered(
            query_id,
            "Nothing in the Seeley help centre matched that question. Try naming the "
            "product family or model (for example 'TQ ducted gas heater') or the fault "
            "code shown on the controller.",
            product_family,
            elapsed(),
        )
        _record(query_id, query, chunks, response, log_path, eval_id)
        return response

    try:
        payload = llm.complete_json(
            system=system_prompt(),
            user=build_user_message(query, chunks, pinned),
            model=model or get_settings().generate.model,
            client=client,
            max_tokens=MAX_ANSWER_TOKENS,
        )
    except (llm.LLMError, ConfigurationError) as exc:
        log.warning("generation_failed", extra={"query_id": query_id, "error": str(exc)})
        response = _unanswered(
            query_id,
            f"The answer could not be generated: {exc}",
            product_family,
            elapsed(),
        )
        _record(query_id, query, chunks, response, log_path, eval_id)
        return response

    response = assemble(
        query_id=query_id,
        payload=payload,
        chunks=chunks,
        product_family=product_family,
        latency_ms=elapsed(),
    )
    _record(query_id, query, chunks, response, log_path, eval_id)
    return response


def assemble(
    query_id: str,
    payload: dict[str, Any],
    chunks: Sequence[dict[str, Any]],
    product_family: str | None,
    latency_ms: int,
) -> AskResponse:
    """Turn a model payload into a validated response.

    Separated from :func:`answer` so the checks can be tested without a model:
    these are the guarantees the prompt asks for and cannot enforce.

    Args:
        query_id: Correlation id.
        payload: The parsed model output.
        chunks: The passages that were supplied, in citation order.
        product_family: Inferred family.
        latency_ms: Elapsed time.

    Returns:
        The response, with citations resolved and confidence adjusted.
    """
    text = normalise_sections(plain_ascii(str(payload.get("answer") or "")))
    confidence = str(payload.get("confidence") or "unknown").lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "unknown"

    if payload.get("answered") is False or not text:
        missing = plain_ascii(str(payload.get("missing") or ""))
        reason = text or (
            f"The retrieved manuals do not cover that. {missing}".strip()
            if missing
            else "The retrieved manuals do not answer that question."
        )
        return _unanswered(query_id, reason, product_family, latency_ms)

    valid = set(range(1, len(chunks) + 1))
    text, dropped = _strip_unresolvable(text, valid)
    if dropped:
        # The model cited a passage that was never supplied. Not fatal, but the
        # answer is less grounded than it claimed to be.
        log.warning(
            "citation_out_of_range", extra={"query_id": query_id, "dropped": sorted(set(dropped))}
        )

    # Sorted by marker, not by order of appearance: the reader scans the source
    # list looking for the number they just read in the prose.
    used = sorted(cited_numbers(text))
    citations = [build_citation(n, chunks[n - 1]) for n in used]
    display_text = strip_inline_citations(text)

    if not citations:
        # Section 8 requires a citation on every factual claim. An answer with
        # none is either ungrounded or trivial; either way it is not what this
        # system promises, so it is not presented as confident.
        confidence = "low"
        log.warning("answer_had_no_citations", extra={"query_id": query_id})
    elif dropped:
        confidence = "low" if confidence == "high" else confidence

    return AskResponse(
        query_id=query_id,
        answer=display_text,
        citations=citations,
        confidence=confidence,
        product_family=product_family,
        latency_ms=latency_ms,
    )


def _record(
    query_id: str,
    query: str,
    chunks: Sequence[dict[str, Any]],
    response: AskResponse,
    log_path: Path | None,
    eval_id: str | None = None,
) -> None:
    """Write one query record and emit a structured log line.

    Args:
        query_id: Correlation id.
        query: The question.
        chunks: The chunks retrieved.
        response: The response returned.
        log_path: Override the query-log destination.
        eval_id: SME evaluation case id, when present.
    """
    record = {
        "query_id": query_id,
        "query": query,
        "chunk_ids": [c.get("chunk_id") for c in chunks],
        "answer": response.answer,
        "citations": [c.n for c in response.citations],
        "confidence": response.confidence,
        "product_family": response.product_family,
        "latency_ms": response.latency_ms,
    }
    if eval_id:
        record["eval_id"] = eval_id
    log_query(record, log_path)
    log.info(
        "answered",
        extra={
            "query_id": query_id,
            "citations": len(response.citations),
            "confidence": response.confidence,
            "latency_ms": response.latency_ms,
        },
    )


def generate_answer_response(
    query: str,
    top_k: int | None = None,
    product_hint: str | None = None,
    **kwargs: Any,
) -> AskResponse:
    """Answer a question for the API layer.

    A thin alias so ``api.main`` imports one clearly-named entry point rather
    than a function called ``answer`` that shadows the noun everywhere it is
    used.

    Args:
        query: The installer's question.
        top_k: Passages to retrieve.
        product_hint: Product family supplied by the caller.
        **kwargs: Passed through to :func:`answer`.

    Returns:
        The cited answer.
    """
    return answer(query, top_k=top_k, product_hint=product_hint, **kwargs)
