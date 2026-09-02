# 0008. OpenAI as the default LLM provider, behind a provider-agnostic layer

## Status

Accepted. Supersedes the model choices in build-plan §7.1 and §8, not the
requirements they carry.

## Context

The build plan names Claude throughout: Sonnet for generation (§8), Haiku for
the query router (§7.1). Cohere `rerank-v3.5` is named separately for reranking
(§7.2 step 6).

Those were written before any key existed. What this project actually has is an
OpenAI key — already paid for and in use for the 16,189 embeddings of Stage 4 —
and no Anthropic or Cohere key.

The question put to this decision was simply "why Anthropic? can we use OpenAI?"
The honest answer is that nothing in the architecture requires Anthropic, so
this ADR records what changed and what did not.

## Decision

**`openai` is the default provider, and `anthropic` stays fully wired.**

All model access goes through one new module, `seeley_rag.llm`, which dispatches
on `generate.provider`. Nothing above that layer imports a vendor SDK. Switching
back is two lines in `config/config.yaml` plus a key; the Anthropic path has its
own tests and is not left to rot.

### What OpenAI covers

One key now covers every remaining model need:

| Need | Stage | Model |
|---|---|---|
| Embeddings | 4 (done) | `text-embedding-3-large` |
| Query rewrite | 5 | `gpt-4.1-mini` |
| Listwise rerank | 5 | `gpt-4.1-mini` |
| Generation | 6 | `gpt-5` |
| Vision transcription | 2b (3,459 pages owed) | a multimodal gpt-5/4.1 |

That last row is the one that tips it. Stage 2b is still outstanding and needs a
vision model; on the plan's arrangement that would have been a third vendor
relationship. It is now the same key.

### What did not change

The §8 requirements are provider-independent and remain in force: answer only
from context, inline citations on every factual claim, exact values verbatim,
and the safety rules covering gas carriage, combustion and mains electrical
work. Those are prompt and evaluation obligations, not vendor properties, and no
model choice discharges them.

## Measured: reasoning models are the wrong default for a router

§7.1 budgets the query router at ~200ms. The gpt-5 family reasons unless told
not to, measured against this key:

| Model | Latency | Reasoning tokens |
|---|---|---|
| gpt-5-mini (default effort) | 8.3s | yes |
| gpt-5-nano (default effort) | 4.4s | yes |
| gpt-5-mini, `reasoning_effort="minimal"` | 2.1s | 0 |
| gpt-5-nano, `reasoning_effort="minimal"` | 1.5s | 0 |
| gpt-4.1-mini | 1.0s | 0 |
| gpt-4.1-nano | 0.6s | 0 |

Two consequences in the code:

- `llm.is_reasoning_model()` drives which parameters are sent. Reasoning models
  reject `max_tokens` in favour of `max_completion_tokens`, and take
  `reasoning_effort`; sending either to a gpt-4.1 model is an API error. Getting
  this wrong fails at runtime with a vendor message that says nothing about this
  project, so it is tested directly.
- The router defaults to `gpt-4.1-mini`, and `reasoning_effort` defaults to
  `minimal` for when a reasoning model is chosen anyway.

## Both LLM steps in Stage 5 are opt-in

Neither is on by default, and the reasons differ:

**Query rewrite** (`retrieve.use_query_llm`, default false). The deterministic
pass from ADR 0007 already supplies product family, model codes and fault codes
from the lexicon. The rewrite adds real value to dense retrieval but costs
~1–4s against a cascade that is otherwise ~90ms. Measured: 0.01s off, 3.5s on.

**Listwise rerank** (`retrieve.use_llm_rerank`, default false). §7.2 warns this
roughly doubles per-query cost, which is why Cohere is preferred. But it is the
plan's own named fallback, and with an OpenAI key it is now available rather
than theoretical — so it is implemented, tested, and off.

It is worth having. Asked "what is the manifold gas pressure setting for a TQ5":

| Rank | identity (boosted fusion) | LLM listwise |
|---|---|---|
| 1 | TQ TQA TQM installation manual, p.14 | **TQ DGH Gas Valve Identification and Gas Pressure settings, p.4** |
| 2 | TQD3 TQMD5 installation manual, p.8 | 2026 Braemar Heating Technical, p.7 |
| 5 | TQ DGH Gas Valve Identification, p.5 | TQ DGH Gas Valve Identification, p.5 |

The document that actually answers the question moved from 5th to 1st, above
generic installation manuals that merely mention gas. Cost: 6.6s.

The reranker may return an ordering and nothing else. It cannot rewrite a
passage, so nothing it produces reaches the answer as evidence; hallucinated or
repeated indices are dropped, and a passage it omits keeps its fused rank rather
than vanishing — silence is not evidence of irrelevance.

## Consequence: the ambient environment can supply keys

`pydantic-settings` reads the process environment, and developer shells
frequently export `ANTHROPIC_API_KEY`. On the machine this was built on, one
does.

That is convenient and a hazard: a test asserting "no key configured" passed or
failed depending on whose shell ran it, and a developer could unknowingly bill a
key they did not mean to use. Tests that depend on a key's absence now clear it
explicitly with `monkeypatch`. Worth knowing when reading a run's provider line.

## Alternatives considered

**Staying on Claude and asking for an Anthropic key.** Defensible — the plan
chose it, and a second provider is a hedge against one vendor's outage. Rejected
because it adds an account, a key and a billing relationship to a project that
already has a working one, and delivers nothing the OpenAI path does not.

**Using OpenAI but hard-coding it.** Rejected. The plan's choice deserves to
remain reachable, the abstraction cost is one small module, and Stage 6 has not
been written yet — pinning the provider before the generation prompts exist
would be the wrong order.

**Turning the LLM rerank on by default now that it is cheap to reach.** Rejected
for this build. §7.2's cost warning stands, the identity backend labels itself
honestly, and there is no eval yet to show the improvement generalises beyond
the one query above. Revisit when the eval can measure it.
