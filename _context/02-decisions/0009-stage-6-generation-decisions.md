# 0009. Stage 6 generation: verify the prompt's rules rather than trust them

## Status

Accepted.

## Context

Stage 6 turns retrieved passages into a cited answer. Build-plan §8 states the
requirements precisely — answer only from context, inline citations on every
factual claim, exact values verbatim, surface a page image where the answer
depends on one, and specific safety rules covering gas carriage, combustion and
mains electrical work.

Every one of those is expressed to the model as an instruction. The decision
this ADR records is what to do about the fact that instructions are requests.

## Decision 1: the prompt's guarantees are checked in code

`generate/answer.py::assemble` runs after the model has spoken and enforces
three things the prompt cannot:

| Check | Failure it prevents |
|---|---|
| Citation numbers must resolve to a supplied passage | `[9]` with eight passages renders as a verified claim pointing at nothing |
| Only cited passages become citations | Listing all eight sources under a two-source answer implies corroboration that does not exist |
| An uncited answer is forced to `confidence: low` | A confident, fluent, ungrounded answer is indistinguishable from a real one |

An out-of-range marker is stripped from the prose rather than left for a reader
to chase, and it also downgrades a `high` confidence — the answer was less
grounded than it claimed to be.

None of this makes a wrong answer right. It makes a wrong answer *visible*,
which is the most a generation stage can honestly offer. The alternative — ship
the prompt and assume compliance — is the arrangement where the first sign of a
problem is an installer acting on a fabricated torque figure.

`assemble` is deliberately separable from `answer` so these checks are tested
without a model in the loop.

## Decision 2: declining is a first-class answer

§8 requires saying so when the context does not hold the answer. Two supporting
choices:

- **No retrieval results means no model call.** Nothing to ground an answer in
  is nothing to ask about; spending a request to be told so is waste.
- **Citation markers are stripped from a decline.** The model tends to write
  "the passages only cover warranty terms [1], [2]" — accurate, but it points at
  a source list that is deliberately empty. The prose is cleaned so the two
  halves agree.

Verified against the corpus: "what is the warranty period on a Tesla Powerwall"
returns a decline, no citations, and names what the passages actually cover.

## Decision 3: cross-family fault codes get an explicit instruction

ADR 0007 introduced `PinnedCode.cross_family` for the case where a code exists
but not for the product asked about. Stage 6 is where that flag has to do
something, so `render_pinned_code` states it emphatically and tells the model
not to present it as the answer.

It works. Asked "the ducted heater is throwing E:04" the answer is that there is
no such code for Seeley ducted gas heating, that RC and VRF use `E4` for
compressor discharge temperature protection and those do not apply, and that the
nearest DGH code is FC4/DC4 — with its actual thresholds (70 °C during run,
re-ignition after cooling to 60 °C, lockout after 10 minutes).

A refinement came out of testing: the model initially attributed the pinned codes
as `[SDHV / SCHV Three Phase Ducted Inverter Installation Manual, p.27]`, using
square brackets for something that is not a numbered passage. The prompt now
reserves brackets for passage numbers and asks for prose attribution otherwise.

## Decision 4: `gpt-5` for generation, despite the latency

Measured on the same question, same passages:

| Model | Latency | Citations | Words |
|---|---|---|---|
| gpt-5 | 6.7s | 3 | 249 |
| gpt-5-mini | 6.3s | 7 | 374 |
| gpt-4.1 | 2.5s | 2 | 191 |

§8 targets ~2.8s end to end, which only `gpt-4.1` meets. `gpt-5` is nonetheless
the default: these answers concern gas pressures, combustion and mains
electrical work, and four seconds is a cheap price for the better reading of a
fault table. `generate.model` changes it in one line, and the measurement is
recorded here so the trade is an informed one rather than a default nobody
examined.

Observed range in practice is 3–8s, varying with answer length rather than with
reasoning effort — `reasoning_effort` is already `minimal` from ADR 0008.

## Decision 5: the query log is written from the first answer

§9 asks for `query_id`, the query, the retrieved chunk IDs and the answer to be
logged as JSONL, and notes the first week of real queries is worth more than any
synthetic eval.

That is implemented now, in Stage 6, rather than deferred to the API stage that
will expose it — a log added after a system starts being used has already missed
the queries worth having. Writes go to `data/reports/queries.jsonl` and a
failure to write is logged but never fails an answer.

## Known limitation: duplicate sources from one page

Two chunks from the same document page can both be cited, producing `[1]` and
`[8]` resolving to the same title and page. The API response keeps both, because
every marker in the prose must resolve; the CLI groups them onto one line.

The root fix is deduplicating at retrieval, which ADR 0007 deliberately
deferred: capping per-document hits would have dropped the very page carrying
the gas-pressure table in that ADR's own worked example. It should be revisited
when the eval can measure the trade rather than argue it.

## Alternatives considered

**A second model call to verify the answer against the passages.** A real
option, and the honest version of "check the model's work". Rejected for now: it
doubles latency and cost, and the deterministic checks above catch the failure
modes that are actually detectable without another judgement call. Worth
revisiting when the eval can show what the deterministic checks miss.

**Refusing to answer at all when confidence is low.** Rejected. An installer
with a partial answer and a visible confidence marker is better served than one
with nothing, provided the citations are honest — and they are checked to be.
