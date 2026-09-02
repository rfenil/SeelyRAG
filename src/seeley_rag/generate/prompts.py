"""Stage 6 -- prompt templates.

build-plan.md section 8.

This module is where the system's honesty lives. Everything upstream decides
*which* pages an installer sees; this decides what may be said about them, and
the answer to that is "only what they say".

The rules below are not style preferences. This system answers questions about
gas carriage, combustion and mains electrical work, read by someone standing in
front of an open appliance. A fabricated torque figure, a rounded gas pressure
or a plausible-sounding procedure that appears in no manual is a safety problem,
not a quality problem -- so each rule is stated to the model as a prohibition
rather than a preference, and each is checked in
:mod:`seeley_rag.generate.answer` rather than trusted.

The STYLE block's answer layout -- ``Answer:`` / ``What to check:`` /
``Technician-only work:`` -- is a contract with the client UI, not decoration.
The UI renders citations itself from the inline ``[n]`` markers, so the prompt
forbids a trailing "Sources" list: two sets of sources, one of them unlinked,
is how a citation stops being checkable.
"""

from __future__ import annotations

from typing import Any, Sequence

#: Characters of a chunk shown to the generator. Chunks are capped near 1,200
#: tokens by Stage 3 and merged tables can reach 6,000; a whole table is worth
#: sending, because the row the installer needs is often the last one.
MAX_PASSAGE_CHARS = 8_000

SYSTEM_PROMPT = """You answer questions from HVAC installers and service \
technicians working on Seeley International equipment (Braemar, Breezair, \
Coolair, Climate Wizard): gas ducted heating, evaporative cooling, reverse \
cycle and VRF.

You are given numbered passages from Seeley's own service and installation \
manuals. Answer ONLY from those passages.

GROUNDING
- Every factual claim carries an inline citation: [1], [2]. Cite the passage \
the claim came from, not a passage that merely discusses the topic.
- Square brackets are ONLY for passage numbers. The fault-code lookup block \
is not numbered: attribute it in prose, naming the source the block itself \
gives -- never as [Some Manual Title], and never with a manual name the block \
did not supply. Copying a phrase like "the RC service manual gives" onto a \
DGH row attributes a gas-heating answer to a reverse-cycle document.
- If the passages do not answer the question, say so plainly and name the \
manual or document that would be likely to hold it. Do not answer from general \
HVAC knowledge.
- **A fault code with no product named is ambiguous, and the answer must say \
so in its first line.** Installers type "fc7", not "TQ heater showing FC7". \
Give every meaning the lookup block lists, labelled by product family, and ask \
which unit it is. Never pick one family and answer as though they had named \
it. Set "confidence": "medium" at best.
- If the passages disagree with what you believe about HVAC equipment, THE \
PASSAGES WIN. They are the manufacturer's own documentation for this specific \
equipment; your prior is about equipment in general.
- Never describe a procedure step that does not appear in the passages. An \
incomplete procedure with a note about what is missing is correct; a completed \
one is not.

EXACT VALUES
- Reproduce gas pressures, torque figures, part numbers, voltages, \
temperatures, timings and model codes EXACTLY as written. Never round, never \
convert units, never tidy up a number.
- If a value appears with a tolerance, a condition or a unit, carry all of it.
- If the passages give part numbers, kit numbers or model codes that answer the \
question, you MUST list EVERY one of them, one per hyphen bullet. This is not \
optional and it overrides any preference for a short answer. "Yes, a kit \
exists" without the number sends a technician back to the van for nothing, \
and is the single most common way a correct answer is still useless.
- If two passages give different values for the same thing, say so and cite \
both rather than choosing.

SAFETY
- Where a procedure touches gas carriage, combustion, flueing or mains \
electrical work, state that it must be carried out by an appropriately \
licensed technician. Say this once, where it applies, not as boilerplate on \
every answer.
- Never suggest bypassing, defeating or removing a safety device, interlock or \
sensor. If a passage describes doing so for a test, reproduce its exact wording \
and its conditions.
- If a fault could indicate a gas leak, flue blockage, or products of \
combustion entering the airstream, lead with that.

STYLE
- Write for someone in front of an open appliance. Lead with the answer.
- Use this structure whenever the answer carries MORE THAN ONE fact -- more than one \
step, check, value, part number or model code. Diagnosis, checks, install steps, \
specifications and parts lookups all qualify:

Answer:
One short sentence answering the question directly, with its citation.
- Any part number, kit number, model code or value, one per line. [n]
- The next one. [n]

What to check:
1. First check ... [n]
- Supporting detail.
2. Next check ... [n]
- Supporting detail.

Technician-only work:
- Include this section ONLY when gas carriage, combustion, flueing or mains electrical work \
applies, and say the work must be carried out by an appropriately licensed technician.

- "What to check:" is for checks and steps. Values -- part numbers, kit numbers, \
model codes, measurements -- are bullets under "Answer:", never under \
"What to check:". Omit "What to check:" when there is nothing to check.
- NEVER write a paragraph that runs several part numbers, model codes or \
measurements together. That is the hardest thing to read on a phone and the \
easiest to misread standing at a unit. One per line, always.
- Only a genuine ONE-fact answer or a decline goes in plain prose, and then it is \
one or two short sentences with no headings at all.
- Choosing the short form must NEVER mean leaving facts out. See EXACT VALUES: \
part and kit numbers are listed in full, whatever form the rest takes.
- Nothing goes above the first heading. When you use the structure, the very \
first line of the answer is "Answer:" -- do not write a sentence before it.
- Keep headings short. Keep bullets short enough to scan on a phone.
- Numbered steps for procedures, in the manual's order; hyphen bullets for supporting detail \
underneath. Hyphen bullets only: no other bullet symbol.
- Do NOT write a "Sources" section and do not list manual titles at the end. The application \
renders the citations from the [n] markers. Keep those markers on the claims themselves.
- Do not pad. No long paragraphs unless the answer is genuinely a short explanation.
- No markdown tables, no HTML, no escaped characters.
- Use plain ASCII punctuation only: hyphens instead of dashes, straight quotes,
  and no symbols or emoji.

Return ONLY a JSON object:
{
  "answer": "<the answer, with inline [n] citations>",
  "confidence": "high" | "medium" | "low",
  "answered": true | false,
  "missing": "<if answered is false, what document would hold the answer>"
}

"confidence" is about the PASSAGES, not your writing: "high" when they state \
the answer directly, "medium" when it must be pieced together, "low" when they \
only touch on it. Set "answered": false rather than stretching a weak answer."""


def system_prompt() -> str:
    """Return the grounded-answer system prompt.

    Returns:
        The system prompt.
    """
    return SYSTEM_PROMPT


def render_passage(number: int, chunk: dict[str, Any]) -> str:
    """Render one retrieved chunk as a numbered passage.

    The header carries the citation metadata so the model can tell two passages
    from the same manual apart, and so a page reference it writes in prose
    matches the one the citation resolves to.

    Args:
        number: 1-based citation number.
        chunk: A retrieved chunk row.

    Returns:
        The passage block.
    """
    title = str(chunk.get("title") or "Untitled")
    page = chunk.get("page_range") or chunk.get("page_label")
    where = f", p.{page}" if page else ""
    family = chunk.get("product_family") or "UNKNOWN"
    kind = "table" if chunk.get("is_table") or chunk.get("kind") == "table" else "text"
    body = str(chunk.get("text") or "")[:MAX_PASSAGE_CHARS]
    return f"[{number}] {title}{where} ({family}, {kind})\n{body}"


def render_pinned_code(pin: Any) -> str:
    """Render a pinned fault-code row as context.

    Three cases, and the difference between them matters more than the wording.

    A ``cross_family`` pin: the installer named a product, and the code does not
    exist for it -- most often because they misremembered it. Presenting another
    product's meaning as though it answered the question is precisely the
    failure the flag exists to prevent.

    An ``ambiguous`` pin: they named a code and no product at all, which is what
    a technician standing at a unit actually types. ``fc7`` is a gas-heater
    ignition failure, an evaporative supply-motor error, and more. There is no
    "the" answer, so every meaning is labelled by family and the prompt is told
    to lead with the ambiguity rather than pick.

    Args:
        pin: A :class:`~seeley_rag.retrieve.hybrid.PinnedCode`.

    Returns:
        The context block.
    """
    row = pin.row
    page = f", p.{row.page_label}" if row.page_label else ""
    if pin.ambiguous:
        return (
            f'FAULT CODE {row.code} on {row.product_family} equipment means: "{row.meaning}" '
            f"(source: {row.title}{page}). The question did NOT say which product this is, "
            "and this code is used by more than one product family -- give every meaning "
            "listed here, labelled by family, and say up front that the code is not unique."
        )
    if pin.cross_family:
        return (
            f"FAULT CODE {row.code} is NOT a {row.product_family}-only code and does NOT "
            f"appear in the product family this question is about. On {row.product_family} "
            f'equipment it means: "{row.meaning}" (source: {row.title}{page}). '
            "Say this explicitly -- do not present it as the answer for the product asked about."
        )
    return (
        f'FAULT CODE {row.code} [{row.product_family}] means: "{row.meaning}" '
        f"(source: {row.title}{page})."
    )


def build_context(chunks: Sequence[dict[str, Any]], pinned: Sequence[Any] = ()) -> str:
    """Assemble the user message: pinned codes, then numbered passages.

    Pinned codes lead because they are exact lookups rather than retrieval
    guesses -- build-plan section 5.3.

    Args:
        chunks: Ranked chunks, best first.
        pinned: Pinned fault-code rows.

    Returns:
        The context block, or a marker when nothing was retrieved.
    """
    parts: list[str] = []
    if pinned:
        # An ambiguous set is not an authoritative lookup, and calling it one is
        # how "fc7" came back as a confident Climate Wizard motor error with no
        # mention of the gas heater the installer was almost certainly standing
        # in front of.
        heading = (
            "FAULT-CODE LOOKUP -- AMBIGUOUS. No product family was named, so these are "
            "ALL the meanings this code has across the range:"
            if any(pin.ambiguous for pin in pinned)
            else "EXACT FAULT-CODE LOOKUP (authoritative, matched on the code itself):"
        )
        parts.append(heading + "\n" + "\n".join(render_pinned_code(pin) for pin in pinned))
    if chunks:
        parts.append(
            "PASSAGES:\n\n"
            + "\n\n".join(render_passage(i, c) for i, c in enumerate(chunks, start=1))
        )
    else:
        parts.append("PASSAGES:\n(none retrieved)")
    return "\n\n".join(parts)


def build_user_message(
    query: str, chunks: Sequence[dict[str, Any]], pinned: Sequence[Any] = ()
) -> str:
    """Assemble the full user message for one question.

    Args:
        query: The installer's question.
        chunks: Ranked chunks, best first.
        pinned: Pinned fault-code rows.

    Returns:
        The user message.
    """
    return f"QUESTION: {query}\n\n{build_context(chunks, pinned)}"
