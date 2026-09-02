# 0007. Stage 5 retrieval: deterministic query understanding, and an honest reranker

## Status

Accepted.

## Context

Stage 5 turns a question into ranked chunks. The build plan (§7) specifies the
cascade precisely — code lookup, dense 30, BM25 30, RRF at k=60, boosts *before*
truncation, rerank to 5–8 — and that shape is implemented exactly as written.

Two of its components name services this project has no key for: Haiku for query
understanding (§7.1) and Cohere `rerank-v3.5` for reranking (§7.2 step 6). Only
an OpenAI key exists. The decisions below are about what to do with those two
holes, plus three things the real corpus forced.

## Decision 1: query understanding is deterministic first, LLM second

§7.1 specifies one Haiku call returning five fields. Three of them need no model
at all:

- `product_family` — `config/models.yaml` already resolves this, and it is the
  same lexicon that labelled 13,156 pages in Stage 2.
- `model_series` — the same lexicon.
- `fault_codes` — `chunk/codes.py` already extracts these with filters tuned
  against the real corpus in ADR 0005.

So the deterministic pass runs always and produces a usable `Understanding` in
microseconds. The LLM pass is wired, tested and optional, and is asked only for
what regexes are genuinely bad at: intent, diagram intent, and a rewritten query.

**It may never overwrite a deterministically extracted field.** A hallucinated
`E:05` in place of `E:04` would be pinned ahead of retrieval and cited as
authoritative — the single worst failure available in this system. There is a
test asserting the model's output cannot reach `fault_codes` or
`product_family`.

This is a departure from the plan, and the better arrangement regardless of
keys: a regex finds `E:04` more reliably than a language model does, costs
nothing, and cannot invent one. The consequence is that the cascade runs today
with no Anthropic key at ~0ms rather than ~200ms, and *improves* rather than
changes when a key is added.

### Consequence: suffixed model codes

The lexicon lists `TQ`; installers write `TQ5`. Whole-token matching — correct
for document titles — missed it, so "what is the manifold gas pressure for a
TQ5" resolved to `UNKNOWN` and received no boost at all.

`_series_in_query` accepts a token whose *letter prefix* is exactly a known code
and is followed by a digit. That recovers `TQ5`, `TQM6`, `MCMX3` without
admitting arbitrary words, and a test asserts "check within 24 hours" yields no
model codes.

## Decision 2: with no Cohere key, rerank is the identity backend, and says so

The tempting move is a cheap local substitute. Both obvious candidates were
rejected:

- **Lexical overlap** is a worse BM25, and BM25 is already one of the two fused
  channels.
- **Re-scoring with the same embedding model** is the dense channel again, also
  already fused.

Either would move results around without adding information, which is *worse*
than not reranking, because it looks like a quality step and is not one.

So `identity_rerank` returns the boosted fusion order — a genuinely reasonable
ranking, fused across two channels and boosted on four signals — and stamps
every row `rerank_backend="identity"`. `rerank_backend()` reports it and
`scripts/06_search.py` prints it.

The labelling is the decision, not an implementation detail. A silent identity
pass reported as reranking would inflate every quality number taken before a
Cohere key arrives, and nobody would know to discount them. Adding the key
upgrades the step without touching a caller; a Cohere failure at runtime falls
back to identity rather than losing the answer.

## Decision 3: cross-family code pins are returned and flagged

§5.3 pins a code-table hit into context ahead of retrieval. The corpus makes
that more delicate than it sounds.

Asked "the ducted heater is throwing E:04", the family resolves to DGH —
correctly — and the code table has **no DGH `E04` at all**, because DGH prints
`FC` codes. Three options:

| Behaviour | Failure |
|---|---|
| Pin the VRF/RC `E04` unflagged | A confident wrong answer with a citation attached |
| Pin nothing | Hides that the code exists, on other products |
| Pin, flagged `cross_family` | — |

`PinnedCode` carries `row` and `cross_family`, so the generator can say "E:04 is
not a gas-heating code; on VRF it means high discharge temperature protection".
The CLI prints `(NOT this product family)`.

This is the pinning analogue of §7.1's soft-boost rule: never assert something
the metadata contradicts, and never silence something the installer explicitly
named.

## Decision 4: retrieval handles are process-cached

Opening the LanceDB table costs **~4.8s** against 16,189 rows; the searches
themselves are 30–80ms. Building a fresh store, embedder and code index per call
made a ~150ms cascade take **7.4s**.

`default_store()`, `default_embedder()` and `default_code_index()` are
`lru_cache`d for the process, like `get_settings()`. Every one is injectable, so
tests and the API supply their own. Measured after: **81–102ms** per query.

## What the corpus showed about fusion

The plan asserts hybrid retrieval is needed. The index confirms it, and names
the failure mode:

| Query | Dense alone | BM25 alone | Fused + boosted |
|---|---|---|---|
| "Braemar evaporative cooler water pump not priming" | correct Braemar EVAP | **Coolerado** — a different product line | correct Braemar EVAP |
| "VRF outdoor unit E4 high discharge temperature" | drifts to RC | correct VRF fault table | correct VRF at #1 |
| "manifold gas pressure setting for a TQ5" | rank 12 | rank 24 | **rank 1** |

The last row is the clearest evidence for the design. The correct chunk — *TQ
DGH Gas Valve Identification and Gas Pressure settings*, p.4 — was **neither
channel's top hit**. Agreement across both plus the family and model boosts
promoted it to first. Concatenating the channels, or trusting either alone,
would have buried it.

It is also why the boosts must be applied before truncation. A chunk ranked 12th
and 24th cannot be promoted by anything applied to an already-truncated top-8.
`test_a_boost_can_promote_from_outside_the_top_k` fails if that ordering is ever
reversed — the exact bug §7.2 rule 5 records from v1 of the plan.

## Alternatives considered

**Hard-filtering on the inferred product family.** Rejected, per §7.1 and now
with evidence: "what is the manifold gas pressure setting for a TQ5" resolved to
`UNKNOWN` before the suffixed-code rule existed. Under a hard filter that query
would have returned nothing. Boosts degrade; filters break.

**Tuning the RRF constant.** Left at 60. It is the standard value, the fusion is
parameter-free by design, and there is no eval yet to tune against. Tuning k
before the eval exists would be fitting to anecdote.

**Deduplicating results by document.** Considered, and not added. Three of the
top four results for the gas-pressure query come from one document, which looks
redundant — but that document *is* the answer, and capping per-document hits
would have removed the p.5 table that carries the actual pressure figures. Worth
revisiting once the eval can measure it.
