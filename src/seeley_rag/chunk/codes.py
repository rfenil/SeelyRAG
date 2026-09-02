"""Stage 3 -- fault-code extraction.

build-plan.md section 5.3. The highest-value 90 minutes in the chunking stage.

Installers search by code, and vector search is *bad* at codes: ``E:04`` and
``E:05`` are near-identical in embedding space and catastrophically different in
meaning. So codes get their own exact-lookup table, swept from every chunk and
especially from table chunks. At query time a detected code hits this table
first and is pinned into context.

Don't make the embedding model do arithmetic.

Patterns live in ``config/models.yaml`` under ``fault_code_patterns``.
Output: ``data/02_processed/codes.jsonl``.

⚠ **The lexicon patterns are far too loose to use raw**, and a first pass over
the real corpus showed exactly how. Four filters exist because of what it
produced, and removing any one of them re-admits the junk it was added for:

1. ``\\bfault\\s+code\\s+(\\w+)\\b`` captures whatever word follows the phrase.
   The first pass yielded codes named ``access``, ``chart``, ``column``,
   ``definition``, ``displayed``, ``does``, ``history`` and ``Braemar``. A
   candidate must now look like a code -- see :data:`_CODE_TOKEN_RE`.
2. ``\\b[EFH][\\s:.-]?\\d{1,2}\\b`` matches "F 12" in a dimensions table and
   "H 10" in a part number, so a candidate is kept only when
   :func:`has_fault_context` finds fault vocabulary nearby.
3. Wide code tables are laid out ``code | meaning | code | meaning``. Joining
   every other cell in the row welded four codes' meanings into each one, so a
   code takes the cell **beside** it, not the whole row.
4. Contents pages match the phrase patterns and yield dotted leaders as the
   meaning ("FAULT CODE 08 EXAMPLE......"). Meanings that are empty or are
   table-of-contents filler are dropped.

Precision matters more than recall here: a missed code still reaches the
installer through ordinary hybrid retrieval, while a wrong one is *pinned ahead*
of it. That is why pure-letter codes such as ``PL`` or ``FP`` are deliberately
not admitted -- they are indistinguishable from ordinary words at this level,
and hybrid retrieval already finds them.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, NamedTuple

from seeley_rag.chunk.base import Chunk, FaultCode
from seeley_rag.logging_conf import get_logger
from seeley_rag.settings import get_models_lexicon

log = get_logger(__name__)

#: Words that mark text as being about faults rather than dimensions or parts.
#: A candidate code with none of these nearby is discarded.
FAULT_VOCABULARY = (
    "fault",
    "error",
    "code",
    "alarm",
    "diagnos",
    "flash",
    "lockout",
    "lock out",
    "trouble",
    "failure",
    "fail",
    "malfunction",
    "sensor",
    "thermistor",
    "shutdown",
    "reset",
    "protection",
)

#: How far either side of a candidate to look for fault vocabulary. One line of
#: a manual is roughly this wide, so it keeps the window to the code's own row
#: or sentence rather than drifting into an unrelated paragraph.
CONTEXT_WINDOW_CHARS = 160

#: What a fault code may look like: at most two letters, an optional separator,
#: one or two digits. A digit is **required** -- see filter 1 in the module
#: docstring.
_CODE_TOKEN_RE = re.compile(r"^[A-Za-z]{0,2}[\s:.\-]?\d{1,2}$")

#: A markdown table row emitted by :mod:`seeley_rag.chunk.tables`.
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")

#: A markdown header separator row, e.g. ``|---|---|``.
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|[\s\-|:]+\|\s*$")

#: Sentence splitter, for pulling a meaning out of prose.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

#: Contents-page filler: a dotted leader running to a page number.
_TOC_LEADER_RE = re.compile(r"\.{4,}")

#: A lower-case letter immediately followed by an upper-case one. Rare in real
#: prose, dense in text where a PDF's two columns have been read across rather
#: than down: "coHnignhe cptrinegss" is "High pressure" woven into its neighbour.
_MIDCAP_RE = re.compile(r"[a-z][A-Z]")

#: Five or more consonants in a row. Catches the other corruption in this
#: corpus: PDFs whose ToUnicode CMap is broken, which decode to a Caesar-shifted
#: alphabet -- "(QVXUHWKHPRWRUSRZHUFDEOH" is "Ensure the motor power cable".
_CONSONANT_RUN_RE = re.compile(r"[bcdfghjklmnpqrstvwxz]{5,}", re.IGNORECASE)

#: Mid-word capitals per 100 characters above which text is taken as
#: interleaved. Measured, not guessed: over the corpus's extracted code
#: meanings, every genuinely corrupt string scores above 1.5 and the highest-
#: scoring legitimate one -- a fault title carrying a part number and an
#: abbreviation -- scores 0.18. The gap between them is an order of magnitude.
MIDCAP_DENSITY_LIMIT = 1.5

#: ``E:04``-shaped: letters, an optional separator, one or two digits.
_LETTER_DIGIT_RE = re.compile(r"^([A-Za-z]{1,2})[\s:.\-]?(\d{1,2})$")

#: A bare number, as printed after "fault code".
_BARE_DIGITS_RE = re.compile(r"^(\d{1,2})$")

#: ``3 flashes``-shaped.
_FLASH_RE = re.compile(r"^(\d{1,2})\s+flash(?:es)?$", re.IGNORECASE)

#: Prefix given to a code printed as a bare number after "fault code".
#:
#: The DGH manuals print "Fault Code 08" while the article titles, and every
#: installer, say "FC8". They are the same code, so both normalise to ``FC08``
#: or a query for one silently misses the other. The printed form is kept
#: verbatim in :attr:`~seeley_rag.chunk.base.FaultCode.code`.
BARE_CODE_PREFIX = "FC"


class CodeHit(NamedTuple):
    """One raw code occurrence, before deduplication.

    Attributes:
        code: The code as printed.
        code_key: Normalised lookup key.
        meaning: Text explaining the code, verbatim from the source.
        evidence: The line or sentence it was read from.
        in_table: Whether it came from a table row.
    """

    code: str
    code_key: str
    meaning: str
    evidence: str
    in_table: bool


def _compiled_patterns() -> list[tuple[re.Pattern[str], bool]]:
    """Compile the lexicon's fault-code patterns.

    Returns:
        ``(pattern, is_phrase)`` pairs in lexicon order. ``is_phrase`` marks the
        "fault code X" / "error code X" forms, whose captured group may be a
        bare number -- which is only meaningful *because* the phrase named it as
        a code, and so normalises differently.
    """
    raw: list[str] = get_models_lexicon().get("fault_code_patterns", [])
    compiled: list[tuple[re.Pattern[str], bool]] = []
    for pattern in raw:
        regex = re.compile(pattern, re.IGNORECASE)
        compiled.append((regex, regex.groups > 0 and "code" in pattern))
    return compiled


def is_code_like(token: str) -> bool:
    """Whether a candidate token could be a fault code at all.

    Args:
        token: The captured text.

    Returns:
        True for ``E4``, ``FC53``, ``b5``, ``08``; False for ``access``,
        ``chart`` and every other ordinary word the phrase patterns capture.
    """
    stripped = token.strip()
    if _FLASH_RE.match(stripped):
        return True
    return bool(_CODE_TOKEN_RE.match(stripped))


def normalise_code(code: str, from_phrase: bool = False) -> str:
    """Normalise a code for exact lookup.

    ``E:04``, ``E 4``, ``e-04`` and ``E04`` all have to answer the same query,
    so separators are dropped, letters upper-cased and digits zero-padded to two
    places. Flash codes become ``FLASH3`` so they cannot collide with ``E:03``.
    A bare number captured from "fault code 8" becomes ``FC08`` -- see
    :data:`BARE_CODE_PREFIX`.

    Args:
        code: The code as printed.
        from_phrase: Whether it was captured from a "fault code X" phrase, which
            is what makes a bare number meaningful.

    Returns:
        The lookup key, e.g. ``E04``, ``FC08`` or ``FLASH3``. Empty when the
        token is not code-shaped.
    """
    stripped = code.strip()

    flash = _FLASH_RE.match(stripped)
    if flash:
        return f"FLASH{int(flash.group(1))}"

    letter_digit = _LETTER_DIGIT_RE.match(stripped)
    if letter_digit:
        return f"{letter_digit.group(1).upper()}{int(letter_digit.group(2)):02d}"

    bare = _BARE_DIGITS_RE.match(stripped)
    if bare:
        # A bare number is a code only because the phrase said so. Standing
        # alone it is a quantity, a page number or a dimension.
        return f"{BARE_CODE_PREFIX}{int(bare.group(1)):02d}" if from_phrase else ""

    return ""


def has_fault_context(text: str, start: int, end: int) -> bool:
    """Whether a candidate at ``[start:end]`` sits in fault vocabulary.

    Args:
        text: The text the candidate was found in.
        start: Match start offset.
        end: Match end offset.

    Returns:
        True when any word from :data:`FAULT_VOCABULARY` appears within
        :data:`CONTEXT_WINDOW_CHARS` either side.
    """
    window = text[max(0, start - CONTEXT_WINDOW_CHARS) : end + CONTEXT_WINDOW_CHARS].lower()
    return any(word in window for word in FAULT_VOCABULARY)


def looks_corrupt(text: str) -> bool:
    """Whether text shows the corpus's two PDF-extraction corruptions.

    Some manuals extract with their columns woven together character by
    character ("Full wFautlel rw partoetre pcrtiootnection" is "Full water
    protection" interleaved with itself), and a few decode through a broken
    CMap into a shifted alphabet. Both produce text that is worse than useless
    as a pinned fault-code meaning: it is unreadable, but it looks like content.

    This is a Stage 2 defect showing through, not a chunking one -- it reaches
    14.2% of chunks and 20.5% of tokens corpus-wide. Detecting it here keeps it
    out of the one place it would do most damage, the lookup table that gets
    pinned ahead of retrieval.

    Args:
        text: Text to judge.

    Returns:
        True when the text is very likely mis-extracted.
    """
    if not text:
        return False
    if _CONSONANT_RUN_RE.search(text):
        return True
    density = len(_MIDCAP_RE.findall(text)) / len(text) * 100
    return density >= MIDCAP_DENSITY_LIMIT


def is_usable_meaning(meaning: str) -> bool:
    """Whether a candidate meaning says anything.

    Rejecting here rather than at the row level is deliberate: a code whose
    meaning is corrupt in one manual is often printed cleanly in another, and
    :func:`build_code_table` keeps the best occurrence. Discarding the bad
    sighting lets the good one win instead of losing the code entirely.

    Args:
        meaning: The extracted meaning.

    Returns:
        False for empty text, contents-page dotted leaders, and mis-extracted
        text.
    """
    stripped = meaning.strip()
    if len(stripped) < 3:
        return False
    if _TOC_LEADER_RE.search(stripped):
        return False
    return not looks_corrupt(stripped)


def extract_codes(text: str) -> list[str]:
    """Sweep text for fault codes.

    Args:
        text: Page or chunk text.

    Returns:
        The distinct codes found, normalised, in order of first appearance.
        Candidates that are not code-shaped, or that have no fault vocabulary
        nearby, are discarded -- see the module docstring.
    """
    found: list[str] = []
    seen: set[str] = set()
    for pattern, is_phrase in _compiled_patterns():
        for match in pattern.finditer(text):
            if not has_fault_context(text, match.start(), match.end()):
                continue
            token = _matched_code(match)
            if not is_code_like(token):
                continue
            key = normalise_code(token, from_phrase=is_phrase)
            if key and key not in seen:
                seen.add(key)
                found.append(key)
    return found


def _matched_code(match: re.Match[str]) -> str:
    """Return the code from a match, preferring a capture group.

    ``fault\\s+code\\s+(\\w+)`` captures the code itself in group 1, while the
    bare ``E:04`` pattern has no groups and the whole match is the code.

    Args:
        match: A pattern match.

    Returns:
        The code text.
    """
    if match.groups() and match.group(1):
        return match.group(1)
    return match.group(0)


def _split_row(line: str) -> list[str]:
    """Split a markdown table row into its cells.

    Args:
        line: A rendered table row.

    Returns:
        Cell text, outer pipes removed.
    """
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _row_meaning(cells: list[str], code_cell_index: int) -> str:
    """Take the meaning from the cell beside the code.

    Wide fault tables repeat ``code | meaning`` two or three times across a
    single row, so joining every other cell welds unrelated codes together --
    which is what a first pass over the corpus did. The cell immediately after
    the code is its own description; the one before is the fallback for tables
    that put the description first.

    Taken verbatim either way. A reworded fault description is a wrong answer
    with a citation attached, so nothing here paraphrases.

    Args:
        cells: The row's cells.
        code_cell_index: Index of the cell holding the code.

    Returns:
        The adjacent cell's text, or ``""`` when both neighbours are empty.
    """
    after = cells[code_cell_index + 1] if code_cell_index + 1 < len(cells) else ""
    if after.strip():
        return after.strip()
    before = cells[code_cell_index - 1] if code_cell_index > 0 else ""
    return before.strip()


def _sentence_around(text: str, position: int) -> str:
    """Return the sentence containing ``position``.

    Args:
        text: The text searched.
        position: Offset of the match.

    Returns:
        The containing sentence, whitespace-normalised.
    """
    offset = 0
    for sentence in _SENTENCE_RE.split(text):
        if offset <= position < offset + len(sentence) + 1:
            return " ".join(sentence.split())
        offset += len(sentence) + 1
    return " ".join(text.split())


def sweep_text(text: str) -> list[CodeHit]:
    """Find every fault code in a chunk's text, with its meaning.

    Table rows are handled cell-wise so a code keeps the cell that explains it;
    prose falls back to the containing sentence.

    Args:
        text: Chunk text.

    Returns:
        Every accepted occurrence, in order.
    """
    patterns = _compiled_patterns()
    hits: list[CodeHit] = []

    for line in text.split("\n"):
        if _TABLE_SEPARATOR_RE.match(line):
            continue
        if _TABLE_ROW_RE.match(line):
            hits.extend(_sweep_table_row(line, patterns))
        else:
            hits.extend(_sweep_prose_line(line, patterns))
    return hits


def _sweep_table_row(line: str, patterns: list[tuple[re.Pattern[str], bool]]) -> list[CodeHit]:
    """Find codes in one rendered table row.

    Args:
        line: The row.
        patterns: Compiled patterns with their phrase flags.

    Returns:
        Accepted occurrences from this row.
    """
    cells = _split_row(line)
    evidence = " ".join(line.split())
    hits: list[CodeHit] = []
    seen: set[str] = set()

    for index, cell in enumerate(cells):
        for pattern, is_phrase in patterns:
            for match in pattern.finditer(cell):
                token = _matched_code(match)
                if not is_code_like(token):
                    continue
                if not has_fault_context(line, 0, len(line)):
                    continue
                key = normalise_code(token, from_phrase=is_phrase)
                if not key or key in seen:
                    continue
                meaning = _row_meaning(cells, index)
                if not is_usable_meaning(meaning):
                    continue
                seen.add(key)
                hits.append(
                    CodeHit(
                        code=token.strip(),
                        code_key=key,
                        meaning=meaning,
                        evidence=evidence,
                        in_table=True,
                    )
                )
    return hits


def _sweep_prose_line(line: str, patterns: list[tuple[re.Pattern[str], bool]]) -> list[CodeHit]:
    """Find codes in one line of prose.

    Args:
        line: The line.
        patterns: Compiled patterns with their phrase flags.

    Returns:
        Accepted occurrences from this line.
    """
    hits: list[CodeHit] = []
    seen: set[str] = set()

    for pattern, is_phrase in patterns:
        for match in pattern.finditer(line):
            token = _matched_code(match)
            if not is_code_like(token):
                continue
            if not has_fault_context(line, match.start(), match.end()):
                continue
            key = normalise_code(token, from_phrase=is_phrase)
            if not key or key in seen:
                continue
            sentence = _sentence_around(line, match.start())
            if not is_usable_meaning(sentence):
                continue
            seen.add(key)
            hits.append(
                CodeHit(
                    code=token.strip(),
                    code_key=key,
                    meaning=sentence,
                    evidence=sentence,
                    in_table=False,
                )
            )
    return hits


def _hit_rank(row: FaultCode) -> tuple[int, int]:
    """Rank a candidate row so the best evidence wins deduplication.

    Args:
        row: A candidate row.

    Returns:
        ``(in_table, meaning_length)`` -- table evidence beats prose, and among
        equals the fuller explanation wins.
    """
    return (1 if row.in_table else 0, len(row.meaning))


def build_code_table(chunks: Iterable[Chunk]) -> list[FaultCode]:
    """Build the fault-code lookup table.

    Deduplicated by ``(code_key, product_family)``, not by code alone: ``E:04``
    means one thing on a gas heater and another on a VRF unit, and collapsing
    them would answer a DGH question from a VRF manual -- build-plan section 13,
    risk 3.

    Args:
        chunks: Every chunk record.

    Returns:
        One row per code and product family, best evidence retained.
    """
    best: dict[tuple[str, str], FaultCode] = {}

    for chunk in chunks:
        for hit in sweep_text(chunk.text):
            if not hit.code_key:
                continue
            row = FaultCode(
                code=hit.code,
                code_key=hit.code_key,
                meaning=hit.meaning,
                evidence=hit.evidence,
                product_family=chunk.product_family,
                model_series=list(chunk.model_series),
                doc_id=chunk.doc_id,
                chunk_id=chunk.chunk_id,
                page_label=chunk.page_label,
                title=chunk.title,
                source_url=chunk.source_url,
                article_url=chunk.article_url,
                in_table=hit.in_table,
            )
            key = (row.code_key, row.product_family)
            existing = best.get(key)
            if existing is None or _hit_rank(row) > _hit_rank(existing):
                best[key] = row

    rows = sorted(best.values(), key=lambda r: (r.product_family, r.code_key))
    log.info("code_table_built", extra={"codes": len(rows)})
    return rows


def annotate_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Stamp each chunk with the fault codes it contains.

    Done in place and returned for chaining. Retrieval uses this to pin chunks
    whose codes match a query's, so the field has to travel on the chunk rather
    than living only in the lookup table.

    Args:
        chunks: Chunks to annotate.

    Returns:
        The same list, with ``fault_codes`` populated.
    """
    for chunk in chunks:
        codes: list[str] = []
        for hit in sweep_text(chunk.text):
            if hit.code_key and hit.code_key not in codes:
                codes.append(hit.code_key)
        chunk.fault_codes = codes
    return chunks


def codes_by_key(rows: Iterable[FaultCode]) -> dict[str, list[FaultCode]]:
    """Group code rows by lookup key.

    Args:
        rows: Fault-code rows.

    Returns:
        ``code_key -> rows``, one entry per product family that uses the code.
    """
    grouped: dict[str, list[FaultCode]] = {}
    for row in rows:
        grouped.setdefault(row.code_key, []).append(row)
    return grouped


def lexicon_patterns() -> list[str]:
    """Return the raw pattern strings, for reporting and tests.

    Returns:
        The patterns as written in ``config/models.yaml``.
    """
    patterns: list[Any] = get_models_lexicon().get("fault_code_patterns", [])
    return [str(p) for p in patterns]
