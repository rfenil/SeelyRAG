# 0011. Reranking on, ambiguity surfaced, and the queries users actually type

## Status

Accepted. Supersedes nothing; extends ADR 0007's reranking section.

## Context

Both rerank backends have been implemented since Stage 5 and neither was on.
`rerank_backend()` returned `identity` — the boosted fusion order, truncated
and honestly labelled. `production-readiness.md` B-4 records this as the
cascade's missing last quality step.

B-4 also states the condition for turning one on:

> **Done when:** the eval from A-3 shows the improvement, measured rather than
> assumed. Do not enable it permanently on the strength of one query.

A-3 is the 60-question SME set. It does not exist, and it is not ours to
produce. Waiting for it means a built feature stays dark indefinitely; enabling
without it means enabling on the strength of the single TQ5 query in ADR 0008.
Neither is acceptable, so the question became: **what can be measured honestly
today, and is it enough to justify the flip?**

## Decision 1: measure movement and cost, and refuse to measure accuracy

`scripts/09_rerank_ab.py` runs the cascade once per query, captures the fused
and boosted candidate list, then ranks that same list with both backends. The
reranker is the only variable — running the cascade twice would let embedding
cache state and BM25 ties leak into the comparison.

It reports overlap@k, rank movement, whether the first result changed, and
latency. It deliberately does **not** score the two orderings for correctness.
The only available judge would be a model from the same family that produced
the ranking, which is marking its own homework, and a number produced that way
would be quoted later as though it meant something.

Input is the 17 distinct questions in `data/reports/queries.jsonl` — not a
question set, but questions someone actually asked this system, which is more
than the one query the alternative rested on.

**Measured, 2026-08-31, gpt-4.1-mini, top-8:**

| | |
|---|---|
| Median added latency | 0.92s (max 8.7s, a cold first call) |
| Lists changed at all | 11 of 17 |
| First result changed | 5 of 17 |
| Left completely alone | 6 of 17 |

The shape matters more than the totals. The queries it leaves untouched are the
ones the deterministic pass already settles — both Tesla Powerwall declines,
`obscure question`, and the exact TQ gas-pressure lookup. The movement is
concentrated on symptom descriptions carrying no fault code, which is precisely
where fusion is weakest.

Build-plan §7.2's warning that listwise reranking "roughly doubles per-query
cost" was written against a larger model. On the router-class model actually
used it does not hold, and the measured latency is a fraction of the 3–8s
generation it precedes.

## Decision 2: enable it, and record that this does not close B-4

`retrieve.use_llm_rerank: true`.

The evidence supports the flip: the backend only reorders — it can never author
or alter a passage, so its worst case is a worse ordering of the same evidence
— it is conservative where the deterministic signals are strong, it costs about
a second, and it is one config line to revert.

The evidence does **not** support calling B-4 done. Movement is not improvement.
A test asserts the flag's shipped value so that reverting it is a visible
decision rather than a drift, and the config comment carries the measurement
next to the flag.

## Decision 3: the reranker sees the signals the boosts act on

The listwise prompt sent a title and 600 characters of body. That makes the
reranker blind to the metadata `apply_boosts` acts on, so it can silently undo
a boost it cannot see the reason for. Observed, not theorised: on "TQ heater has
no flame, what do I check?" it demoted the FC7 diagnostic article — promoted by
`diagnostic_article_boost`, and the highest-value content in this corpus per the
brief's second corpus fact — beneath training-slide pages.

Each passage header now carries product family, page, and a `DIAGNOSTIC ARTICLE`
tag, and the prompt explains what that tag means. On re-measurement the article
recovered one place. It did not recover first place, and that is left standing
rather than tuned away: without ground truth there is nothing to say a
fault-finding training slide is the wrong answer to "no flame, what do I check".
The change is justified as *giving the model the information*, not as producing
a particular ranking.

## Decision 4: a reported backend must be one that can run

`rerank_backend()` returned `cohere` on the presence of `COHERE_API_KEY` alone.
`cohere` is declared in the `downstream` extra, not `requirements.txt`, so a
venv can hold the key without the package — and then `cohere_rerank`'s except
clause catches the `ImportError`, every query silently falls through to
identity, and `/health`, the CLI banner and the web UI all report `cohere`.

The rows stayed honest, because the fallback stamps them `identity`. The status
line did not, and the discrepancy was visible only in a warning log. That is the
exact failure ADR 0007's labelling exists to prevent, so `cohere_available()`
now requires the key **and** an importable SDK.

## Decision 5: the queries this was measured on are not the queries users type

Every question in `queries.jsonl` is well-formed, because whoever wrote them knew
the system. The end users are trade workers with one hand free. They type `fc7`.

Probing that register exposed a defect with nothing to do with reranking, and
worse than anything reranking could cause. Asked `fc7`, the system answered:

> FC7 means a motor error on Climate Wizard CW-H pre-2020 units [1]. On Q Series
> evaporative coolers, FC07 is a motor error [6]...

`confidence: high`, four evaporative meanings, and no mention of the gas-heater
ignition failure the installer was almost certainly standing in front of.

**Cause.** `CodeIndex.lookup` treated `UNKNOWN` as a matching product family.
Some rows in the corpus carry it, so a bare `fc7` "matched" exactly one row —
the one whose meaning is the string `FAULT CODE 7` — pinned it as an exact,
authoritative, family-matched answer, and never surfaced the DGH or EVAP
meanings at all. The `cross_family` flag could not help: its wording presupposes
a family *was* named, and telling a model a code "does not appear in the product
family this question is about" when the question named no product is incoherent,
so the model ignored it.

**Fix.** A third state, `PinnedCode.ambiguous`. A query naming a code and no
product pins every family's meaning, the context block is headed AMBIGUOUS
rather than authoritative, and the prompt must lead with the ambiguity, label
each meaning by family, ask which unit, and cap confidence at medium. Measured
after: the same `fc7` leads with "Ambiguous code", puts the DGH ignition failure
first, enumerates the rest by family, and returns `confidence: medium`.

`max_pinned_codes` went from 3 to 6, because FC02 alone has four meanings and the
enumeration was silently truncating.

Two further defects fell out of the same probe:

- **`--llm` was not a flag, it was an override.** `store_true` yields `False`
  when absent, so `scripts/07_ask.py` and `06_search.py` forced the rewrite off
  and made `retrieve.use_query_llm` unreachable from either. Now tri-state:
  absent means the config decides.
- **The prompt's prose-attribution example was copyable.** It read *attribute it
  in prose ("the RC service manual gives...")*, and the model duly attributed a
  DGH row to "the RC service manual".

## Decision 6: turn the query rewrite on, and forbid it naming a product

`retrieve.use_query_llm: true`.

The reason it was off — "the fields that matter come from the lexicon regardless"
— is true of a well-formed query and false of the ones users type. Measured
over `scripts/10_novice_queries.py`: `cooler smells` went from reverse-cycle
owners manuals to the right evaporative ones; `gas presure tq5` from a sales
guide to the gas-valve pressure pages; `no hot air` from mixed fan pages to DGH
diagnostics. Cost 1.5—1.8s, not the 1—4s first measured.

But the rewrite as written was told to name "the product type", so on `fc7` it
invented *gas ducted heating* and on `fc 7` *evaporative cooler* — the same
question, opposite guesses, each retrieved for confidently. That is laundering,
not expansion. It is now forbidden from naming any product family, type or model
the query did not, and ambiguity is left to the code lookup where the data is.

`scripts/10_novice_queries.py` is the permanent fixture: bare codes, symptom-only
questions, misspellings, spacing variants. It scores nothing — there is no ground
truth — but it flags the three shapes known to go wrong: a code with no family,
an unresolved family, and a top-k that is really one document.

## Consequences

- `/ask` and `/search` gain roughly a second, and every result and `/health`
  now reports `llm` rather than `identity`.
- Every query makes one additional LLM call. `retrieve.llm_rerank_model` names
  the model for it, defaulting to the router model — which is what it used
  implicitly before the field existed, and which was chosen for latency rather
  than judgement. Naming it makes that mismatch correctable and lets an eval
  attribute a rerank number to a model.
- `make rerank-ab` re-runs the comparison. It should be re-run when the SME set
  lands, when the corpus changes materially (vision, B-1), and before changing
  the rerank model.
- The honest summary for anyone reading a demo: **the answers are now ordered
  by a model that read them, and nobody has yet measured whether that ordering
  is more correct.**
