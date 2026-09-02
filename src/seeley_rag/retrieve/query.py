"""Stage 5 -- query understanding.

build-plan.md section 7.1, which specifies one Haiku call returning::

    {"product_family": "DGH", "model_series": ["TQ"], "fault_codes": ["E:04"],
     "intent": "fault_diagnosis", "wants_diagram": false,
     "rewritten_query": "TQ series gas ducted heater E:04 flame sensing fault"}

**Soft-boost the inferred product family; do not hard-filter.** If an installer
says "the ducted heater is throwing E:04" and the classifier guesses wrong, a
hard filter returns nothing and the system looks broken. Hard-filter only when
the user names a model explicitly -- :attr:`Understanding.model_explicit`
records which case this is.

⚠ **This is deterministic first, LLM second, and that is a deliberate departure
from the plan.**

Three of the five fields need no model at all. ``config/models.yaml`` already
resolves product family and model codes at near-perfect accuracy for zero cost
(the same lexicon Stage 2 used to label 13,156 pages), and
``chunk/codes.py`` already extracts fault codes with filters tuned against the
real corpus. A regex finds ``E:04`` more reliably than a language model does,
and it cannot hallucinate ``E:05``.

So the deterministic pass runs always, and an LLM -- when a key is configured --
is asked only to *add* what regexes are bad at: intent, whether a diagram is
wanted, and a rewritten query. It may never overwrite a deterministically
extracted fault code, because that is the one field where a confident wrong
answer is worst.

The practical consequence is that the whole cascade runs today with no Anthropic
key, at ~0ms instead of ~200ms per query, and gets better rather than different
when a key is added.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from seeley_rag import llm
from seeley_rag.chunk.codes import extract_codes, normalise_code
from seeley_rag.exceptions import ConfigurationError
from seeley_rag.logging_conf import get_logger
from seeley_rag.parse.base import UNKNOWN_FAMILY, resolve_model_series
from seeley_rag.settings import get_models_lexicon, get_settings

log = get_logger(__name__)

#: What the installer is trying to do. Drives nothing structural yet; recorded
#: so the eval can slice accuracy by question type.
Intent = Literal[
    "fault_diagnosis",
    "installation",
    "specification",
    "parts",
    "maintenance",
    "general",
]

#: Words that put a query in each intent. First match in this order wins, most
#: specific first -- "why is the fault code showing during installation" is a
#: fault question, not an installation question.
INTENT_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "fault_diagnosis",
        (
            "fault",
            "error",
            "code",
            "not working",
            "won't",
            "wont",
            "will not",
            "failure",
            "fail",
            "broken",
            "lockout",
            "lock out",
            "flashing",
            "alarm",
            "diagnos",
            "troubleshoot",
            "problem",
            "issue",
            "stopped",
            "no heat",
            "not heating",
            "not cooling",
            "leak",
        ),
    ),
    (
        "parts",
        ("part number", "spare", "replacement", "exploded", "part no", "order a", "kit"),
    ),
    (
        "installation",
        ("install", "commission", "mount", "duct size", "clearance", "wiring diagram", "connect"),
    ),
    (
        "specification",
        (
            "pressure",
            "voltage",
            "torque",
            "rating",
            "capacity",
            "specification",
            "spec",
            "kpa",
            "kw",
            "amp",
            "setting",
            "how many",
            "what size",
        ),
    ),
    (
        "maintenance",
        ("service", "clean", "filter", "maintenance", "annual", "descale"),
    ),
)

#: Phrases meaning "show me the picture". The generator surfaces the page image
#: when this is set (build-plan section 8).
DIAGRAM_MARKERS: tuple[str, ...] = (
    "diagram",
    "wiring",
    "schematic",
    "exploded",
    "drawing",
    "layout",
    "picture",
    "image",
    "where is",
    "which terminal",
    "what does it look like",
)

#: A model code named explicitly enough to hard-filter on: the token appears
#: with a digit attached, as installers write them ("TQ5", "MCMX", "CQ4").
_EXPLICIT_MODEL_RE = re.compile(r"\b[A-Z]{2,5}\d{1,2}[A-Z]?\b")


class Understanding(BaseModel):
    """What a query is asking for.

    Attributes:
        query: The original text, untouched.
        product_family: Inferred family, or ``UNKNOWN``. **Soft-boosted only.**
        model_series: Model codes named in the query.
        model_explicit: Whether a model code was named explicitly enough to
            justify a hard filter. This is the single distinction section 7.1
            draws, so it is recorded rather than inferred downstream.
        fault_codes: Normalised codes found in the query.
        intent: What the installer is trying to do.
        wants_diagram: Whether the answer should surface a page image.
        rewritten_query: Expanded query for dense retrieval. Falls back to the
            original when no LLM is configured.
        source: ``deterministic`` or ``deterministic+llm``, so a result can be
            attributed.
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    query: str
    product_family: str = UNKNOWN_FAMILY
    model_series: list[str] = Field(default_factory=list)
    model_explicit: bool = False
    fault_codes: list[str] = Field(default_factory=list)
    intent: Intent = "general"
    wants_diagram: bool = False
    rewritten_query: str = ""
    source: str = "deterministic"

    @property
    def search_text(self) -> str:
        """The text to embed and to send to BM25."""
        return self.rewritten_query or self.query


def _resolve_family_from_query(query: str) -> str:
    """Infer a product family from free text.

    The lexicon's ``resolve_product_family`` expects category and folder names
    from the portal, which a query does not have. This matches the same aliases
    and model codes against the query itself, longest-pattern-wins for the same
    reason: "VRF reverse cycle" contains RC's "reverse cycle", and first-match
    ordering mislabelled 1,622 pages in Stage 2 before it was fixed.

    Args:
        query: The installer's question.

    Returns:
        A family key, or :data:`~seeley_rag.parse.base.UNKNOWN_FAMILY`.
    """
    families: dict[str, Any] = get_models_lexicon().get("families", {})
    lowered = query.lower()

    best_key, best_len = UNKNOWN_FAMILY, 0
    for key, spec in families.items():
        for pattern in list(spec.get("category_patterns", [])) + list(spec.get("aliases", [])):
            candidate = pattern.lower()
            if candidate in lowered and len(candidate) > best_len:
                best_key, best_len = key, len(candidate)
    if best_key != UNKNOWN_FAMILY:
        return best_key

    # Model codes are the weakest signal and are matched as whole tokens only.
    tokens = {t.strip(".,()/-").upper() for t in query.replace("/", " ").split()}
    for key, spec in families.items():
        for code in spec.get("model_codes", []):
            if code.upper() in tokens and len(code) > best_len:
                best_key, best_len = key, len(code)
    return best_key


def _detect_intent(query: str) -> Intent:
    """Classify what the installer is trying to do.

    Args:
        query: The installer's question.

    Returns:
        The first matching intent, most specific first.
    """
    lowered = query.lower()
    for intent, markers in INTENT_MARKERS:
        if any(marker in lowered for marker in markers):
            return intent  # type: ignore[return-value]
    return "general"


def _wants_diagram(query: str) -> bool:
    """Whether the answer should surface a page image.

    Args:
        query: The installer's question.

    Returns:
        True when the query asks to be shown something.
    """
    lowered = query.lower()
    return any(marker in lowered for marker in DIAGRAM_MARKERS)


def _codes_in_query(query: str) -> list[str]:
    """Extract fault codes from a query.

    ``extract_codes`` requires fault vocabulary near a candidate, which suits
    manual prose but not a bare query like "E:04 on a TQ". The query is wrapped
    in a minimal fault context so the corpus-tuned filters still apply without
    rejecting the very thing being asked about.

    Args:
        query: The installer's question.

    Returns:
        Normalised codes, in order of appearance.
    """
    found = extract_codes(f"fault code: {query}")
    # "FC7" written bare is not matched by the lexicon's letter+digit pattern,
    # which expects one or two leading letters and no more; catch the explicit
    # FC form directly, since it is how DGH codes are actually written.
    for match in re.finditer(r"\bFC\s?(\d{1,2})\b", query, re.IGNORECASE):
        key = normalise_code(f"FC{match.group(1)}")
        if key and key not in found:
            found.append(key)
    return found


def _series_in_query(query: str) -> list[str]:
    """Extract model codes, including suffixed forms installers actually type.

    ``resolve_model_series`` matches whole tokens, which is right for document
    titles but misses "TQ5" -- the lexicon lists ``TQ``, and an installer writes
    the size on the end. A token whose *letter prefix* is a known code counts,
    which recovers "TQ5", "TQM6" and "CQ4D" without admitting arbitrary words:
    the prefix must match a code exactly and be followed by a digit.

    Args:
        query: The installer's question.

    Returns:
        Distinct model codes, lexicon order first.
    """
    found = list(resolve_model_series(query))

    known: set[str] = set()
    for spec in get_models_lexicon().get("families", {}).values():
        known.update(code.upper() for code in spec.get("model_codes", []))

    for token in re.findall(r"\b[A-Za-z]{2,6}\d{1,2}[A-Za-z]?\b", query):
        prefix = re.match(r"^[A-Za-z]+", token)
        if prefix and prefix.group(0).upper() in known:
            code = prefix.group(0).upper()
            if code not in {f.upper() for f in found}:
                found.append(code)
    return found


def understand_deterministic(query: str) -> Understanding:
    """Classify a query using the lexicon and the code patterns only.

    Costs nothing and takes microseconds. This is the floor the LLM path builds
    on, and the whole of it when no key is configured.

    Args:
        query: The installer's question.

    Returns:
        The parsed understanding.
    """
    series = _series_in_query(query)
    family = _resolve_family_from_query(query)
    if family == UNKNOWN_FAMILY and series:
        # A suffixed code like "TQ5" resolves the family that the alias and
        # whole-token passes both missed.
        family = _family_for_series(series)

    codes = _codes_in_query(query)
    intent = _detect_intent(query)
    # A named fault code IS a fault question, whatever words surround it.
    # "the ducted heater is throwing E:04" carries no marker vocabulary at all.
    if codes and intent == "general":
        intent = "fault_diagnosis"

    return Understanding(
        query=query,
        product_family=family,
        model_series=series,
        model_explicit=bool(_EXPLICIT_MODEL_RE.search(query.upper())) and bool(series),
        fault_codes=codes,
        intent=intent,
        wants_diagram=_wants_diagram(query),
        rewritten_query=query,
        source="deterministic",
    )


def _family_for_series(series: list[str]) -> str:
    """Return the family owning the first of these model codes.

    Args:
        series: Model codes found in the query.

    Returns:
        A family key, or :data:`~seeley_rag.parse.base.UNKNOWN_FAMILY`.
    """
    families: dict[str, Any] = get_models_lexicon().get("families", {})
    wanted = {code.upper() for code in series}
    for key, spec in families.items():
        if wanted & {code.upper() for code in spec.get("model_codes", [])}:
            return key
    return UNKNOWN_FAMILY


#: Instruction for the optional LLM pass. It is asked only for what regexes are
#: bad at, and told explicitly not to invent codes.
LLM_SYSTEM_PROMPT = """You expand HVAC service queries for a retrieval system \
over Seeley International installer manuals (gas ducted heating, evaporative \
cooling, reverse cycle, VRF).

Return ONLY a JSON object with these keys:
  "intent": one of fault_diagnosis, installation, specification, parts, maintenance, general
  "wants_diagram": true if the user is asking to be shown a diagram, wiring, or layout
  "rewritten_query": the query expanded with synonyms an installer would use, \
expanding the symptom into the words a manual would use. One sentence. Keep \
every code and model number EXACTLY as written.

Never invent a fault code or model number that is not in the query.

NEVER name a product family, product type or model the query did not name. \
Installers type bare codes: "fc7" is a gas-heater ignition failure AND an \
evaporative supply-motor error, and guessing one silently turns an ambiguous \
question into a confident answer about the wrong appliance. If the query names \
no product, expand the symptom only and leave the product open."""


def _llm_enrich(understanding: Understanding, client: Any | None = None) -> Understanding:
    """Ask a model for intent, diagram intent and a rewritten query.

    Deliberately narrow. The model may not touch ``fault_codes``,
    ``product_family`` or ``model_series``: those come from the lexicon and the
    corpus-tuned patterns, and a hallucinated ``E:05`` in place of ``E:04``
    would be pinned ahead of retrieval and cited as authoritative.

    A failure here is not an error. The deterministic understanding is already
    usable, so any exception logs and returns it unchanged.

    Provider-agnostic via :mod:`seeley_rag.llm` -- OpenAI by default, Anthropic
    when configured. See ADR 0008.

    Args:
        understanding: The deterministic result.
        client: An injected SDK client, for tests.

    Returns:
        The understanding, enriched where the call succeeded.
    """
    if client is None and not llm.is_configured():
        return understanding

    try:
        payload = llm.complete_json(
            system=LLM_SYSTEM_PROMPT,
            user=understanding.query,
            client=client,
            max_tokens=400,
        )
    except (llm.LLMError, ConfigurationError) as exc:
        log.warning("query_understanding_llm_failed", extra={"error": str(exc)})
        return understanding

    enriched = understanding.model_copy(deep=True)
    intent = payload.get("intent")
    if intent in {m[0] for m in INTENT_MARKERS} | {"general"}:
        enriched.intent = intent
    if isinstance(payload.get("wants_diagram"), bool):
        # Either signal is enough: the regex catches "wiring diagram", the model
        # catches "show me how it goes together".
        enriched.wants_diagram = enriched.wants_diagram or payload["wants_diagram"]
    rewritten = payload.get("rewritten_query")
    if isinstance(rewritten, str) and rewritten.strip():
        enriched.rewritten_query = rewritten.strip()
    enriched.source = f"deterministic+{llm.active_provider()}"
    return enriched


def understand(query: str, use_llm: bool | None = None, client: Any | None = None) -> Understanding:
    """Classify and rewrite a user query.

    Args:
        query: The installer's question.
        use_llm: Whether to attempt the LLM enrichment pass. Defaults to
            ``retrieve.use_query_llm``, which is off -- the deterministic pass
            is the designed floor and the rewrite costs seconds. Ignored when no
            key is configured and no client is injected.
        client: An injected chat client, for tests.

    Returns:
        The parsed query understanding.
    """
    if use_llm is None:
        use_llm = get_settings().retrieve.use_query_llm
    understanding = understand_deterministic(query)
    if use_llm or client is not None:
        understanding = _llm_enrich(understanding, client=client)
    log.info(
        "query_understood",
        extra={
            "family": understanding.product_family,
            "codes": understanding.fault_codes,
            "intent": understanding.intent,
            "source": understanding.source,
        },
    )
    return understanding
