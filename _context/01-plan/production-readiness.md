# Production readiness — the path from demo to a system installers rely on

## What this document is

`build-plan.md` specifies the POC. **This specifies what turns that POC into a
production system**, written at the point the POC was finished and demonstrable,
so the reasoning is recorded while it is still fresh rather than reconstructed
later.

It assumes the demo is approved. Nothing here requires rewriting the pipeline —
every item is additive on what exists. That is stated first because it is the
most useful thing to know when picking this up: **you are extending a working
system, not rebuilding one.**

Read `README.md` in `_context/` for how this folder is organised, and the ADRs in
`02-decisions/` for why the existing code is shaped the way it is.

---

## Where the system actually stands

Verified at handoff, not estimated.

### What is built

| Stage | State |
|---|---|
| 0 · Triage | implemented |
| 1 · Acquire | implemented |
| 2 · Parse | implemented **except vision** |
| 3 · Chunk | implemented |
| 4 · Index | implemented |
| 5 · Retrieve | implemented |
| 6 · Generate | implemented |
| 7 · API | implemented |
| 8 · Evaluate | implemented (harness only — no question set) |

**695 tests**, lint clean. `parse/vision.py` is the only stub in the codebase.

### The corpus

- **1,011 articles, 544 unique documents (1.99 GB), 13,156 parsed pages** — all
  **25** categories, not just the two pilot ones
- **16,189 chunks, 6.99M tokens**, embedded for **$0.91**
- **136 fault codes** in the exact-lookup table
- Retrieval cascade: **~90ms**. Generation: **3–8s** (median 7.3s over 37 logged
  queries)

### What is measurably wrong

These four numbers are the honest reason the system is worse than a demo makes
it look. Each is measured, not estimated.

| Defect | Size | Consequence |
|---|---|---|
| Vision unbuilt | **3,459 pages (26.3%)** — 1,708 scanned, 1,751 diagram-heavy | That content is invisible to search entirely |
| PDF text corruption | **14.2% of chunks, 20.5% of tokens** | Caps retrieval quality regardless of embedding or reranking |
| Page labels guessed | **38.7%** — only 7,678 of 12,526 are citable | A citation can name a page number that is not the printed one |
| Nothing reranks | ~~`identity` backend~~ **listwise LLM, on since 2026-08-31** | Closed as a gap; still unmeasured for accuracy — see B-4 |

### What is unknown

**Accuracy.** There is no SME question set, so no gate in build-plan §10 has a
real number against it. Everything above is a known unknown; this is the
unknown one.

---

## Ordering

Items are grouped by what they unblock, not by effort. Within each group they
are ordered by dependency.

**Group A gates any external exposure.** Nothing in B or C matters if A fails.
**Group B is what makes the answers good enough to rely on.**
**Group C is what makes the service operable.**

A-1 through A-3 are largely other people's time. Start them first and build
Group B while they run — that sequencing is the single biggest schedule lever
here.

---

## Group A — before anyone outside the org touches it

### A-1 · Content rights (blocking, not technical)

**Why.** The manuals are Seeley International's intellectual property. This
system crawled a public help centre with no API key and no stated relationship,
and now stores and redistributes derived content. Public availability is not a
licence to ingest.

**Done when:** written permission covering ingestion, storage and redistribution
through a derived product. If Seeley is a client, get it in the engagement
terms. If not, get it before anything is exposed.

**Owner:** commercial, not engineering. **Cannot be worked around technically** —
this is the one item that could invalidate the work rather than merely limit it.

See build-plan §15.

### A-2 · PyMuPDF licence (blocking, legal-technical)

**Why.** `pymupdf~=1.24.9` in `requirements.txt` is **AGPL-3.0**. Fine for
internal POC use; serving externally without resolving it is a licence
violation.

**Two routes:**
1. Buy a commercial licence from Artifex.
2. Swap to `pypdfium2` + `pdfplumber`. This touches `parse/pdf.py` and
   `parse/triage.py` only, behind interfaces that already exist — but it
   **re-parses the corpus**, and the resulting text will differ, which changes
   `content_hash` and therefore re-embeds everything (~$1, hours of machine
   time). Cheap in money, not in calendar.

**Done when:** either a licence is held, or PyMuPDF is gone from
`requirements.txt` and the corpus re-parsed and re-indexed.

**Note:** route 2 is worth combining with **B-2** — both re-parse, and doing them
in one pass costs one re-index instead of two.

See build-plan §14 item 2.

### A-3 · The SME question set (blocking, and on the critical path)

**Why.** This is the only item that tells you whether the system works. Without
it, every quality decision below is guesswork, and no one can honestly answer
"how often is it right?" for a system answering gas and electrical questions.

**What to ask for.** The template and brief already exist at
`_context/04-eval/sme-question-template.yaml`. **60 questions**, mix per
build-plan §10:

- ~40% fault diagnosis
- ~25% installation / commissioning
- ~15% spec lookup (clearances, gas pressures, electrical)
- ~10% "show me the diagram"
- ~10% **unanswerable from the corpus**

⚠ The last 10% is the one people drop, and it is the one that measures whether
the system invents answers. Do not let it be trimmed.

⚠ `expected_page` must be the **printed** page number on the page itself, not
the PDF viewer's counter. The template says this at length because getting it
wrong makes the page-accuracy gate meaningless.

**Done when:** 60 questions land, `python scripts/05_eval.py --run` executes
them, and each gate in build-plan §10 has a real number.

**Expect the first run to fail some gates.** That is the point. Two failures are
already predictable and are *not* retrieval problems:

- **Page accuracy** is capped by B-3 — 38.7% of labels are guesses. The eval
  excludes non-citable labels rather than scoring against them; if too many
  cases land on guessed pages the metric reports `n/a`, which is what it does
  today.
- **p95 latency** currently reads 52s, but that is **one outlier** ("What
  clearances do I need around a TQ series indoor unit?"). Median is 7.3s.
  Investigate that query specifically before treating latency as systemic.

### A-4 · Authentication and rate limiting on the API

**Why.** Verified: the API has **no authentication, no rate limiting, no CORS
configuration and no middleware of any kind**. Anyone who can reach it can query
it, and every query spends OpenAI credit. That is a cost-exposure and
availability problem from the first minute of exposure.

**Minimum viable:**
- API-key or bearer auth as a FastAPI dependency
- Per-key rate limit and a daily spend cap
- CORS restricted to known origins
- Request-size limits (`top_k` is already bounded at 50 in the schema; `query`
  length is not)

**Where:** `api/main.py`. The route registrars are already split
(`_register_query_routes`, `_register_search_routes`,
`_register_content_routes`), so a dependency slots in without restructuring.

**Done when:** an unauthenticated request is refused, and a runaway client
cannot exceed a configured daily spend.

---

## Group B — before installers rely on the answers

### B-1 · Vision (`parse/vision.py`)

**The largest measurable gap, and entirely within our control.**

**Why.** 3,459 pages — 26.3% of the corpus — carry no usable text. 1,708 are
scanned, 1,751 are diagram-heavy. They are recorded with `needs_vision=True`, so
the work is queued and countable rather than silently missing.

**What exists already:** every page has a rendered PNG at 150 DPI
(`data/01_interim/page_images/`), the stub declares `transcribe_page` and
`caption_diagram`, and `seeley_rag.llm` is provider-agnostic — the OpenAI key
already in use covers multimodal calls, so **no new vendor is needed** (ADR
0008).

**Why it is cheap to fold in:** the index is incremental by construction (ADR
0006). `chunk_id` is deterministic and `content_hash` covers the embedded text,
so transcribing a page changes only that page's chunks. A no-op re-index is
**4.7s and zero API calls**; after vision it re-embeds only what actually moved.
This was designed for exactly this task.

**Cost:** 3,459 vision calls, one per page. Build-plan §12 estimated the vision
line for ~10k pages and warned it is the one estimate that can move by 3×.
**Price it at current rates before committing** — it is the largest single
spend in the project.

**Done when:** `needs_vision` is false corpus-wide, `scripts/04_index.py` shows
the new chunks, and `scripts/05_embed.py` re-embeds only them.

### B-2 · PDF extraction corruption

**Why.** 14.2% of chunks and 20.5% of tokens carry mis-extracted text. Two
distinct causes, both identified:

1. **Interleaved columns** — two text columns read across rather than down.
   "Full wFautlel rw partoetre pcrtiootnection" is "Full water protection" woven
   into itself.
2. **Broken ToUnicode CMaps** — a few PDFs decode to a shifted alphabet.
   "(QVXUHWKHPRWRUSRZHUFDEOH" is "Ensure the motor power cable".

**What exists:** `chunk/codes.py::looks_corrupt` already detects both and keeps
them out of the fault-code table (ADR 0005). The detector is calibrated —
mid-word capitals at ≥1.5 per 100 characters, or a five-consonant run — and the
gap between corrupt and legitimate text is an order of magnitude.

**What it does not do:** the corrupt text is still in the embedded chunks. This
is a **Stage 2** defect surfacing downstream, and it caps retrieval quality no
matter how good the embeddings or the reranker are.

**Approach:** use `looks_corrupt` to identify affected documents, then apply a
different extraction path to those specifically — a layout-aware extractor, or
the vision path from B-1 for the worst of them. **Combine with A-2 if the
PyMuPDF swap happens**, so the corpus is re-parsed once rather than twice.

**Done when:** the corrupt fraction is materially reduced and measured, using
the same detector so the before/after numbers are comparable.

### B-3 · Page-label recovery

**Why.** Only 61.3% of page labels are citable: 3,608 come from the PDF's own
label tree, 4,070 from a footer regex, and **4,848 (38.7%) are guessed as
`index + 1`**. Every page records `label_source`, so the eval can exclude
guesses rather than score against them — which is correct, and also means the
page-accuracy gate is capped before retrieval is involved.

**Why it matters beyond the metric:** a citation naming a page number that is
not the printed one is worse than one naming no page. Verification is the entire
trust mechanism (build-plan §8).

**Approach:** the guessed pages are concentrated in documents whose footers the
regex missed. Sample them by document, extend `parse/pagelabels.py` for the
footer styles that failed, and re-run `scripts/03_parse.py`. Improvement is
directly measurable as the `label_source` distribution shifts.

**Note:** vision (B-1) may recover labels on scanned pages as a side effect, so
sequence B-1 first and re-measure before sizing this.

### B-4 · Turn on reranking — **partly done**

**Status, 2026-08-31.** The listwise LLM backend is **on** (`retrieve.use_llm_rerank: true`,
ADR 0011). `scripts/09_rerank_ab.py` / `make rerank-ab` measures it against the
identity backend over the query log, ranking the *same* fused candidate list with
both so the reranker is the only variable. Over the 17 distinct logged questions:
median **+0.92s**, **11 of 17** lists changed, **5 of 17** changed their first
result, and the 6 the deterministic pass already settles were left untouched.

**What is still open is the reason this item is not closed:** that harness measures
movement and cost, and deliberately refuses to measure accuracy — the only
available judge is a model from the family that produced the ranking. **Movement
is the precondition for improvement, not improvement.** Re-run `make rerank-ab`
when the SME set lands; reverting is one line in `config/config.yaml`.

Two defects surfaced while measuring, both fixed (ADR 0011): the reranker saw
only a title and 600 characters, so it silently undid the boosts it could not
see — it demoted the FC7 diagnostic article beneath training slides on "TQ heater
has no flame" — and `rerank_backend()` reported `cohere` on the key alone, while
every query fell through to identity because the SDK lives in the `downstream`
extra.

**Why it mattered.** The cascade's last quality step was absent. Every result
honestly reported `rerank_backend: "identity"`, so it could not be mistaken for
a reranked list (ADR 0007) — but it was a real gap.

**Two options, both already implemented:**
- **Cohere `rerank-v3.5`** — build-plan §7.2's preference, and still the one to
  prefer. Needs a key **and** `pip install -e ".[downstream]"` — the SDK is not in
  `requirements.txt`. With both, it activates with no code change; with only the
  key, `rerank_backend()` now reports `identity` rather than claiming a backend
  that cannot run.
- **Listwise LLM rerank** — `retrieve.use_llm_rerank: true`, which is now the
  shipped default. Runs on the existing OpenAI key. §7.2's "roughly doubles
  per-query cost" was written against a larger model: on the router-class model
  actually used the median is **+0.92s**. `retrieve.llm_rerank_model` names it.

**Evidence it is worth it:** on "what is the manifold gas pressure setting for a
TQ5", the LLM reranker moved the document that actually answers the question
from **5th to 1st**, above generic installation manuals (ADR 0008).

**Done when:** the eval from A-3 shows the improvement, measured rather than
assumed. Seventeen logged queries are not that eval, which is why this item is
*partly* done rather than done.

### B-4b · Queries as users actually type them — **new, partly done**

**Why.** Every question in `queries.jsonl` is well-formed, because whoever wrote
them knew the system. Trade workers type `fc7`, `no hot air`, `braemer heater no
ignition`. Nothing in the repo tested that register, and the first probe of it
found a worse defect than anything in this document: a bare `fc7` returned four
Climate Wizard meanings at `confidence: high` and never mentioned the gas-heater
ignition failure, because `UNKNOWN` was being treated as a matching product
family in the code table. Fixed; see ADR 0011.

**What exists now.** `scripts/10_novice_queries.py` (`make novice`) — 18 queries
across bare codes, symptom-only questions, misspellings and spacing variants. It
scores nothing, because there is no ground truth; it flags three shapes known to
go wrong: a code with no product family, an unresolved family, and a top-k that
is really one document.

**What is still open.**

- The fixture has no expected answers. **The SME set (A-3) should be written in
  this register, not in the register of `queries.jsonl`** — ask for the questions
  as an installer would type them into a phone, not as a well-formed sentence.
  This is worth raising before the 60 questions are written, because it is
  cheap then and a rewrite afterwards.
- `family-unresolved` still fires on most symptom-only queries. The lexicon
  resolves brands and model codes, not symptoms; a symptom-to-family mapping
  ("no hot air" — heating, "not cooling" — cooling) would close much of it and is
  a lexicon edit, not a code change.
- Nothing handles a follow-up. The ambiguous answer ends by asking which unit,
  and there is no multi-turn memory to receive the reply (build-plan §14 item 5).

### B-5 · Human review of gas and electrical answers

**Why.** Build-plan §13 risk 9: a confident wrong answer on a gas or electrical
procedure is the lowest-probability, highest-consequence failure in this system.
No automated gate substitutes for a licensed technician reading the output.

**Done when:** a sample of answers across the fault-diagnosis and
installation categories has been reviewed by someone licensed, and the findings
are fed back into the system prompt or the eval set.

**Note:** Stage 6 already enforces citation resolution, drops hallucinated
markers, and downgrades uncited answers to `low` confidence (ADR 0009). Those
checks make a wrong answer *visible*; they cannot make it right.

---

## Group C — before it is someone's production dependency

### C-1 · Storage

LanceDB is embedded, single-node, on a local filesystem. Build-plan §6 always
named the production swap: **pgvector** to live inside the RosteredAI Postgres
estate, or **Qdrant** past single node.

The store sits behind the `VectorStore` protocol in `index/store.py`, and
nothing outside that module imports `lancedb` — the swap was deliberately kept
to one file. The embedding cache is independent of the store, so rebuilding into
a new backend costs **no API spend at all**.

### C-2 · Operations

- **Backup and restore** — `data/00_raw/` is write-once and re-acquirable only
  by a 25-minute polite crawl; the derived stages are reproducible but slow.
  Neither is currently backed up.
- **Monitoring and alerting** — structured JSON logs exist
  (`logging_conf.py`); nothing consumes them.
- **Deployment and rollback** — no pipeline, no container, no runbook.
- **Query log growth** — `data/reports/queries.jsonl` grows unbounded. It is
  valuable (it is the eval's input); it needs rotation and retention.
- **Health checks** — `/health` already reports each dependency separately
  (index presence, row count, code count, provider, key, rerank backend), so it
  is usable as a readiness probe as-is.

### C-3 · Full-corpus re-index memory

A full rebuild peaks at **~2.9 GB RSS** — `build_index` embeds everything before
writing any row, holding 16,189 × 3,072 floats at once. It completes, and
incremental re-runs never approach it because they embed almost nothing. Stream
batches into the store before the corpus grows substantially — vision (B-1) will
roughly increase the indexable text.

---

## Group D — quality upgrades, once the above is true

In build-plan §14's priority order, with what has changed since it was written:

1. **Contextual retrieval, done properly** — whole document in a cached prompt,
   chunks grouped per document, enrichment **before** embedding. §5.2 calls it
   the highest-value single upgrade. ~1 day, $20–60.
2. **Figure detection and cropping** — page images cover it for now.
3. **Incremental sync** — ⚠ **the plan says this needs an API key; it does
   not.** `03-research/portal-recon.md` §7 found the portal publishes a sitemap
   with 1,008 article URLs and per-article `lastmod`. That makes incremental
   sync possible today, on the public crawl. Still unimplemented, and now the
   cheapest item on this list.
4. **The other 18 categories** — ✅ **already done.** All 25 categories are
   crawled; the corpus is 1,011 articles, not the 326-article pilot.
5. UI · multi-turn memory · .NET/Ocelot integration · fine-tuned embeddings ·
   model/part knowledge graph.

---

## What deliberately does not need doing

Recorded so no one spends time re-litigating it:

- **The pipeline architecture.** Every item above is additive. No stage needs
  rewriting.
- **Embedding cost.** The full corpus is $0.91. Build-plan §12 says it: do not
  spend engineering hours optimising a rounding error.
- **The provider choice.** OpenAI covers embeddings, routing, reranking,
  generation and the outstanding vision work on one key (ADR 0008). Anthropic
  remains wired and tested behind `generate.provider` if that changes.
- **Retrieval-level deduplication.** Considered twice and deferred twice (ADRs
  0007, 0009): capping per-document hits would have dropped the very page
  carrying the gas-pressure table. Revisit only when the eval can measure the
  trade.
- **Tuning the RRF constant.** Left at 60. Tuning it before the eval exists is
  fitting to anecdote.

---

## A note on stale dependency pins

Three separate installs broke on version pins written against an older
interpreter — `tiktoken` (Stage 3), `lancedb`/`pandas`/`numpy` (Stage 4), and
`pydantic` (Stage 7). All are now lower bounds rather than `~=` pins (ADR 0006).

Production will want pins again, for reproducibility — but pin from a **resolved
lockfile against the target interpreter**, not by hand. Hand-written pins are
what caused all three failures.
