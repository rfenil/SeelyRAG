# CLAUDE.md — working reference for agent sessions

## What this is

A RAG system over Seeley International's public Freshdesk help centre, so HVAC
installers can ask fault-diagnosis and installation questions and get answers
cited to the exact manual page. Python 3.11+, FastAPI, delivered as a REST API.
The full specification is `_context/01-plan/build-plan.md` — read it before
changing anything structural.

## Four corpus facts that drive every design decision

1. **The knowledge is in the PDFs, not the HTML.** ~900 articles, but a typical
   article body is one sentence plus a link to a 2 MB service manual. The HTML is
   a card catalogue. Capturing article text without attachments captures nothing.
2. **A minority of articles are real content.** The diagnostic/fault-finding
   articles (80+ in DGH alone) have substantial bodies in installer language.
   Distinguish them from stubs; do not discard them together.
3. **No Freshdesk API key exists.** `/api/v2/*` returns 401. The public crawl is
   the only acquisition path (ADR 0002).
4. **The same PDF is attached to multiple articles.** 15–30% duplication.
   Deduplication is correctness, not optimisation.

## Pipeline stages

| Stage | Module | State |
|---|---|---|
| 0 · Triage | `parse/triage.py` | **implemented** |
| 1 · Acquire | `acquire/` | **implemented** |
| 2 · Parse | `parse/base.py`, `pdf.py`, `html.py`, `pagelabels.py` | **implemented** |
| 2b · Vision | `parse/vision.py` | stub (pages flagged `needs_vision`) |
| 3 · Chunk | `chunk/` | **implemented** |
| 4 · Index | `index/` | **implemented** |
| 5 · Retrieve | `retrieve/` | **implemented** |
| 6 · Generate | `generate/` | **implemented** |
| 7 · API | `api/` | **implemented** |
| 8 · Evaluate | `evaluate.py` | **implemented** (harness; no question set) |

Stubs import cleanly, pass lint, and raise `NotImplementedError` naming the
build-plan section that specifies them.

## Commands

```bash
make init          # venv + install + create data/ dirs
make robots        # Stage 0 gate: may we crawl? Run this first.
make triage        # PDF triage -> data/reports/triage_*.md
make acquire       # crawl the pilot categories -> manifest + PDFs
python scripts/03_parse.py                     # Stage 2 -> pages.jsonl + page images
python scripts/03_parse.py --refresh-metadata   # after editing config/models.yaml
python scripts/04_index.py                      # Stage 3 -> chunks.jsonl + codes.jsonl
python scripts/04_index.py --stats              # what it would produce, writing nothing
python scripts/05_embed.py --plan               # Stage 4: what it would embed and cost
python scripts/05_embed.py --smoke              # 200 chunks + real queries, ~1 cent
python scripts/05_embed.py                      # Stage 4 -> data/03_index/
python scripts/06_search.py --demo              # Stage 5: run the cascade
python scripts/06_search.py "TQ FC7" --explain  # per-channel ranks and boosts
python scripts/06_search.py --demo --llm-rerank # force listwise rerank (now the default)
python scripts/09_rerank_ab.py --plan           # B-4 A/B: what it would run, spends nothing
python scripts/09_rerank_ab.py                  # B-4 A/B: identity vs llm over the query log
python scripts/10_novice_queries.py             # how the cascade handles 'fc7', 'no hot air'
python scripts/07_ask.py "TQ heater FC7?"       # Stage 6 -> cited answer
python scripts/07_ask.py --demo --snippets      # incl. a question it should decline
python scripts/08_serve.py                      # Stage 7 -> REST API on :8000
make lint          # black --check, isort --check, flake8
make test          # pytest
make coverage      # pytest with an 80% floor on acquire/
make chunk         # Stage 3 (same as scripts/04_index.py)
make embed         # Stage 4 (same as scripts/05_embed.py)
make search        # Stage 5. Usage: make search ARGS='--demo'
make ask           # Stage 6. Usage: make ask ARGS='--demo'
make serve         # Stage 7. Usage: make serve ARGS='--reload'
make rerank-ab     # Stage 5 B-4 A/B. Usage: make rerank-ab ARGS='--plan'
make novice        # Stage 5. The queries trade workers actually type.
make clean         # remove derived stages; NEVER touches data/00_raw
```

`make` is not installed by default on Windows (`winget install ezwinports.make`).
Without it, run the recipe bodies directly — each is one or two plain commands.

Useful flags: `python scripts/02_acquire.py --limit 3 --dry-run`,
`--categories`, `--rps`, `--no-attachments`, `--progress-every`.

**Crawl everything:** `python scripts/02_acquire.py --categories` (no values
after the flag means all 25 categories).

**Resume is the default.** A run reads the existing manifest first and skips
articles already acquired (their pages are never fetched) and attachments
already on disk (`attachment_id -> sha256` from the manifest, since the hash is
only knowable after a download). `--overwrite` starts over; `--no-resume`
appends without skipping. A manifest truncated by a kill is repaired
automatically via `compact()`, which also drops duplicate rows. Rows are flushed
per-write so a kill loses at most the row in flight.

## Rules that are load-bearing

**Crawl etiquette is a correctness requirement, not politeness.** With no API
key there is no fallback channel, so a block ends the project.

- 1 req/sec, single-threaded. **Never parallelise the crawl.** The scraper and
  the attachment downloader share one `RateLimiter`, so 1 rps is a property of
  the run, not of each component. Do not give them separate limiters.
- Always cache. Every fetch is keyed by URL under `data/00_raw/html/`.
- **Stop immediately on 429 or 403.** Never retry into a block. Retries are for
  5xx and timeouts only.
- Honest User-Agent with a contact address.
- The robots gate runs before the first fetch of every run, and an *undetermined*
  verdict (500, network failure) blocks the crawl just as a disallow does.

**Data layout.**

- `data/00_raw/` is **write-once**. Nothing modifies or deletes a file under it.
- PDFs are content-addressed: `data/00_raw/pdf/{sha256}.pdf`. The path *is* the
  hash, so dedupe is free and re-downloads are idempotent.
- **Every path comes from `paths.py`.** No directory string literals elsewhere.
- `data/` is gitignored entirely.

**`_context/` holds everything about the project that is not part of it** —
brief, plan, ADRs, research, eval material. Nothing in `src/`, `tests/`,
`scripts/` or `config/` may import from it. No `.py` files live there.
`_context/scratch/` is gitignored working space.

## Conventions

- Pydantic v2 for all models and settings. `from __future__ import annotations`
  at the top of every module. Type hints on every signature. Google-style
  docstrings on every public function and class.
- **No `print()` in `src/`.** Logging is structured JSON via `logging_conf.py`.
  Scripts may print user-facing summaries only.
- No bare `except:`. Catch from the hierarchy in `exceptions.py`
  (`AcquisitionError`, `RobotsDisallowedError`, `RateLimitedError`,
  `ManifestError`, `ParseError`, `ConfigurationError`).
- black line length 100, isort profile black. flake8 has its own `.flake8`
  because it cannot read `pyproject.toml`.
- **No test may make a real network request.** `pytest_httpx` mocks HTTP and a
  `no_network` autouse fixture in `conftest.py` fails any test that gets past it.
- **Every non-obvious decision gets an ADR** in `_context/02-decisions/`. Never
  edit a decided ADR — supersede it.
- All file I/O pins `encoding="utf-8"`. Windows defaults to cp1252 and would
  corrupt article titles.

## Two places the live portal contradicts the build plan (ADR 0004)

Both would cause silent data loss, so do not "fix" them back toward the plan:

1. **Pagination is `/folders/{id}/page/{N}`, not `?page=N`.** The query-string
   form silently returns page 1, so a crawler using it captures 10 of 80
   articles and reports success.
2. **Article bodies now carry 1026 characters of shared safety boilerplate.**
   It is stripped (`strip_boilerplate`, markers in `config/config.yaml`) before
   the `<200 char` stub rule is applied. Unstripped, ~900 stubs are misclassified
   as content and the index fills with copies of one safety notice.

## Current state

**Done.** Stages 0, 1, 2 (except vision), 3, 4, 5, 6 and 7. **695 tests**,
91% coverage on `chunk/`, 89% on `index/`, 94% on `retrieve/`, 93% on
`generate/` and on `acquire/` + `parse/`, lint clean. **`parse/vision.py` is
the only stub left in the project.**
Crash-resume works in the crawl and the parse; chunking (20s) and re-indexing
(4.7s) are cheap enough to simply re-run.

Corpus on disk: **1,011 articles, 544 unique documents (1.99 GB), 13,156 parsed
pages** — 12,526 PDF pages from 540 documents plus 630 diagnostic articles.
2,956 tables detected, 12,526 page images rendered at 150 DPI. Verified against the live
portal — robots gate passes, 143 folders and 169 DGH articles enumerated, real
PDFs downloaded content-addressed, manifest validates clean.

**Measured over the whole corpus** (12,526 pages): 72.4% plain text, 14.0%
diagram-heavy, 13.6% scanned. **3,459 pages (26.3%) await vision** — close to the
plan's 35–45% estimate, not the 11.5% an early 2-manual sample suggested.

**Page labels: only 61.3% are citable.** 3,608 come from the PDF's label tree,
4,070 from the footer regex, and **4,848 (38.7%) are guessed as `index+1`**. Every
page records `label_source`, so the eval can exclude guesses rather than score
against them. This caps the page-accuracy gate — read it before blaming
retrieval.

**Stage 3 output: 16,189 chunks, 6.99M tokens** (mean 432) — 13,259 prose,
2,930 table chunks of which 19 are merged across pages. 394 near-empty pages and
524 sub-50-character fragments ("NOTES:", "SECTION 1", "This page has been left
blank intentionally") are dropped rather than indexed. Embedding the lot with
`text-embedding-3-large` costs **$0.91**, so chunk quality is worth engineering
time and embedding cost is not.

**Re-indexing is incremental by construction.** `chunk_id` is deterministic
(`{doc_id}:p{page_index}:c{ordinal}`) and every chunk carries
`content_hash = sha256(final text)`. Re-running reports unchanged / changed /
new / gone, and Stage 4 embeds only `changed + new`. This is what makes the
vision backfill an update rather than a rebuild: transcribing the 3,459 pending
pages changes only their own chunks.

**⚠ 14.2% of chunks (20.5% of tokens) carry PDF-extraction corruption.** Two
kinds: columns interleaved character-by-character ("Full wFautlel rw partoetre
pcrtiootnection" is "Full water protection" woven into itself), and a few PDFs
whose broken ToUnicode CMap decodes to a shifted alphabet. This is a **Stage 2**
defect, not a chunking one. `chunk/codes.py` detects it (`looks_corrupt`) and
keeps it out of the fault-code table, but it is still in the embedded chunk
text and will cap retrieval quality. Fixing it means a different extraction path
for the affected documents — not yet scoped.

**The fault-code table is 136 rows**, down from 294 before filtering. The
lexicon's four patterns are far too loose to use raw, and `codes.py` documents
each filter alongside the junk it removes: phrase patterns capturing ordinary
words (`access`, `chart`, `history`), `[EFH]\d{1,2}` matching duct dimensions,
whole-row joins welding four codes' meanings together, and contents-page dotted
leaders. Codes are keyed by `(code_key, product_family)` — `E:04` on a gas
heater is not `E:04` on a VRF unit. Bare numbers from "Fault Code 8" normalise
to `FC08` so they match the `FC8` installers actually say.

**Stage 4 is built and the index is live: 16,189 rows at 3,072-d**, with both
LanceDB's native Rust FTS index and an IVF-PQ vector index. The first full build
took 906s and 63 API requests for **$0.91**. Verified against real queries.

**Re-indexing costs nothing.** A no-op re-run is **4.7s and zero API calls** —
the plan partitions the corpus by `content_hash` and embeds only `changed + new`.
This is the property the vision backfill depends on: transcribing the 3,459
pending pages will re-embed only those chunks.

Embeddings are cached under `data/cache/embeddings/{model}-{dim}/` in 256
sharded JSON files, keyed by the hash of the exact text sent. The cache is
independent of the store, so dropping the LanceDB table and rebuilding is free.
The namespace carries model and width, so a 1024-d experiment cannot read 3072-d
vectors and switching back recovers the old cache.

**⚠ The first full build peaked at ~2.9 GB RSS.** `build_index` embeds every
chunk before writing any row, so all 16,189 vectors are held as Python lists at
once. It completed fine, and incremental re-runs never approach it because they
embed almost nothing — but a full rebuild of a substantially larger corpus should
stream batches to the store instead. Not urgent; worth knowing before the corpus
grows.

**What the first real queries show.** Retrieval works, and it already
demonstrates why Stage 5's cascade is necessary rather than optional. Asked
"Braemar evaporative cooler water pump not priming", dense search returns the
right Braemar EVAP manuals while BM25 returns *Coolerado* documents — a
different product line. Asked for VRF `E4`, BM25 finds the right VRF fault table
while dense drifts to RC. Neither channel alone is trustworthy on product
family; RRF fusion plus the product-family soft boost is what fixes it.

**Stage 5 is built. The cascade runs in ~90ms** over the 16,189-row index:
code lookup → dense 30 + BM25 30 → RRF(k=60) → boosts → truncate → rerank.

**Query understanding is deterministic first, LLM second** — a departure from
§7.1 worth keeping. The lexicon already resolves product family and model codes,
and `chunk/codes.py` already extracts fault codes with corpus-tuned filters; a
regex finds `E:04` more reliably than a model does and cannot hallucinate
`E:05`. The LLM path may only *add* intent, diagram-intent and a rewritten query
— it can never overwrite an extracted code or family, and a test asserts that.

`_series_in_query` resolves suffixed codes: the lexicon lists `TQ`, installers
write `TQ5`. Without it, "manifold pressure for a TQ5" got no family and no
boost at all.

**⚠ The two LLM steps in Stage 5 are configured separately.**

- `retrieve.use_query_llm` — **now ON** (2026-08-31, ADR 0011). It was off
  because "the fields that matter come from the lexicon regardless" — true of a
  well-formed query, false of the ones installers type. Cost 1.5–1.8s. CLI:
  `--llm` / `--no-llm`, both tri-state: absent means the config decides.
- `retrieve.use_llm_rerank` — was off for the same cost warning; **now on**,
  see below. Cohere is still preferred where a key exists. CLI: `--llm-rerank`.

**Reranking has three backends** — `cohere` (needs a key *and* the SDK, which
lives in the `downstream` extra), `llm` (the plan's own listwise fallback), and
`identity`. Every result carries `rerank_backend`, and the CLI prints the active
one. **The labelling is the point:** a silent identity pass reported as
reranking would inflate every eval number — which is why `rerank_backend()`
checks the Cohere SDK is importable rather than trusting the key alone, and why
the cheap substitutes were rejected (lexical overlap is a worse BM25; re-scoring
with the same embedding model is the dense channel again).

**⚠ `retrieve.use_llm_rerank` is now ON (2026-08-31, ADR 0011).** Measured over
the 17 distinct questions in the query log with `make rerank-ab`: median +0.92s,
11 of 17 lists changed, 5 of 17 changed their first result, and the 6 the
deterministic pass already settles were left untouched. §7.2's "roughly doubles
per-query cost" was written against a larger model and does not hold on the
router-class one actually used.

**This does not close production-readiness B-4.** That gate is accuracy, and
`scripts/09_rerank_ab.py` deliberately does not measure it — the only available
judge is a model from the family that produced the ranking. Movement is the
precondition for improvement, not improvement itself. Re-run `make rerank-ab`
when the SME set lands. Reverting is one line in `config/config.yaml`, and a
test asserts the shipped value so a revert is a decision rather than a drift.

The reranker also now sees what the boosts act on — each passage header carries
product family, page and a `DIAGNOSTIC ARTICLE` tag. Without it the reranker
demoted the FC7 diagnostic article beneath training slides on "TQ heater has no
flame": it was blind to the `diagnostic_article` boost that had promoted it.

The LLM reranker earns its cost when used. Asked "manifold gas pressure setting
for a TQ5", it moved *TQ DGH Gas Valve Identification and Gas Pressure settings*
p.4 from 5th to **1st**, above generic installation manuals that merely mention
gas. 6.6s.

**Fusion plus boosts demonstrably fixes what neither channel manages alone.**
Asked "Braemar evaporative cooler water pump not priming", BM25 alone returned
*Coolerado* documents — a different product line; the family boost now puts the
correct Braemar manuals on top. Asked "manifold gas pressure for a TQ5", the
winning chunk was rank **12 in dense and 24 in BM25** — neither channel's
favourite — promoted to first by agreement plus the family and model boosts.

**⚠ A fault code with no product named is the highest-risk query shape, and
it is what users type.** Asked `fc7`, the system used to answer with four
Climate Wizard meanings at `confidence: high` and never mention the gas-heater
ignition failure. `CodeIndex.lookup` was treating `UNKNOWN` as a matching
family, so a bare code pinned the one row whose meaning is the string
`FAULT CODE 7` as authoritative and hid DGH and EVAP entirely. There is now a
third state — `PinnedCode.ambiguous` — which pins every family's meaning,
heads the block AMBIGUOUS, and makes the answer enumerate by family and ask
which unit. `max_pinned_codes` is 6 because FC02 alone has four meanings.

**`scripts/10_novice_queries.py` is the fixture for this** (`make novice`):
bare codes, symptom-only questions, misspellings, spacing variants — the
register trade workers write in, which nothing in `queries.jsonl` represents.
It scores nothing; it flags a code with no family, an unresolved family, and a
top-k that is really one document. Run it after any change to query
understanding, retrieval or the rewrite.

**Cross-family code pins are flagged, not hidden.** Asked "the ducted heater is
throwing E:04", the family resolves to DGH correctly and the code table has no
DGH `E04` at all, because DGH prints `FC` codes. Pinning the VRF compressor
fault unflagged would be a confident wrong answer with a citation; returning
nothing would hide that the code exists elsewhere. `PinnedCode.cross_family`
carries the distinction so the generator can say "E:04 is not a gas-heating
code; on VRF it means...".

**Retrieval handles are process-cached.** Opening the LanceDB table costs ~4.8s
against 16,189 rows while the searches themselves are 30–80ms; rebuilding the
handle per query made a 150ms cascade take 7.4s. `default_store()`,
`default_embedder()` and `default_code_index()` are `lru_cache`d like
`get_settings()`; tests and the API inject their own.

**LLM access is provider-agnostic and defaults to OpenAI** (`src/seeley_rag/llm.py`,
ADR 0008). The build plan names Claude; nothing in the architecture requires it,
and the OpenAI key already used for embeddings also covers the query router,
reranking, generation and the outstanding Stage 2b vision work — one key instead
of three vendors. `generate.provider` switches back to `anthropic`, which stays
tested.

⚠ **Reasoning models are the wrong default for a router.** Measured on this key:
gpt-5-mini 8.3s, gpt-5-nano 4.4s — both spending reasoning tokens. At
`reasoning_effort="minimal"` they drop to 2.1s and 1.5s; `gpt-4.1-mini` is 1.0s
with none. `llm.is_reasoning_model()` also decides which parameters are legal —
reasoning models reject `max_tokens` for `max_completion_tokens`, and sending
`reasoning_effort` to a gpt-4.1 model is an API error.

⚠ **`pydantic-settings` reads the ambient environment.** This machine's shell
exports `ANTHROPIC_API_KEY`, so settings picked it up without it ever being in
`.env`. Convenient and a hazard: tests asserting a key's *absence* now clear it
with `monkeypatch`, and a run's provider line is worth reading before assuming
which account is being billed.

**Stage 6 is built and answering.** `scripts/07_ask.py` runs the cascade, sends
the passages to `gpt-5`, and returns a cited answer in **3–8s**. Every question
is appended to `data/reports/queries.jsonl` with its `query_id`, the chunk IDs
retrieved and the answer given — build-plan §9, written from the first answer
rather than added later.

**⚠ The prompt's rules are checked in code, not trusted.** A system prompt is a
request, and this one guards answers about gas carriage and mains electrical
work. `generate/answer.py::assemble` enforces three things afterwards:

- **Citation numbers must resolve.** A `[9]` when eight passages were supplied
  is stripped from the prose — a marker pointing at nothing looks verified.
- **Only cited passages become citations.** Listing all eight under a two-source
  answer implies corroboration that does not exist.
- **An uncited answer is forced to `confidence: low`**, whatever the model
  claimed, because the property that makes this system trustworthy is missing.

None of that makes a wrong answer right; it makes one visible.

**What the real answers look like.** Exact values survive, which is the point:
"875 Pa on High and 400 Pa on Low. Tolerances High NG +/-50 Pa and Low NG +/-
20 Pa", "flame sensor gap 4-6 mm", "flame sense 2–3 Vdc". Procedures carry the
licensed-technician note where gas or mains work is involved.

**The two hard cases both behave.** Asked "what is the warranty period on a
Tesla Powerwall" it declines, cites nothing, and says what the passages do
cover. Asked "the ducted heater is throwing E:04" it says there is no such DGH
code, attributes the RC and VRF meanings *in prose*, and states they do not
apply — then gives the DGH equivalent (FC4/DC4) with its exact thresholds.

**Generation model latency**, measured on the same question: `gpt-5` 6.7s,
`gpt-5-mini` 6.3s, `gpt-4.1` 2.5s. `gpt-5` is the default — for answers about
gas and mains electrical work, correctness outranks four seconds — and
`generate.model` changes it. The plan's §8 target was ~2.8s, which `gpt-4.1`
meets.

**Stage 7 is built and serving.** `python scripts/08_serve.py` → all six
endpoints from §9, verified over real HTTP against the live index.

**⚠ `/docs` is FastAPI's Swagger UI; the corpus inventory is at
`/docs-inventory`.** The plan asked for `/docs`, but taking that path removes
the API's own documentation from an integration seam whose whole purpose is
being integrated against.

**Filters constrain the search; they do not trim its output.** `/search`
filters are pushed into *both* retrieval channels as a LanceDB pre-filter.
Applied afterwards to an already-truncated top-k, an explicit filter returns
nothing whenever the matches sat below the cut — which is how this first
behaved: "fault code" filtered to VRF gave **0 hits post-filtered and 3
pre-filtered**. Filter values must match `^[A-Za-z0-9_]{1,64}$`; they are
lexicon identifiers concatenated into a predicate, so anything else is a 400.

**`stream` is refused, not ignored.** `stream: true` returns **501**. Returning
a whole response to a caller expecting a stream looks like it worked.

**The store opens at startup, in a lifespan hook.** Opening the table costs
~4.8s against 16,189 rows; paying it per request made the first call after each
deploy look broken. A missing index is `degraded`, not a failed boot — `/health`
still comes up to say what is wrong, reporting each dependency separately
because an empty index and a missing key need different fixes.

**`/pages/{doc_id}/{page_index}.png` is the only place a caller touches the
filesystem.** The id must match a SHA-256 or `article:{digits}`, the index must
be non-negative, the filename is rebuilt rather than taken, and the resolved
path is confirmed to sit inside the image root. Malformed id → 400; well-formed
but absent → 404.

⚠ Two stale pins bit again, as ADR 0006 predicted: `pydantic~=2.8.0` conflicted
with FastAPI's resolution and is now `>=2.8,<3` (verified on 2.13). And
`from __future__ import annotations` broke `/openapi.json` — FastAPI could not
resolve a `-> Response` string annotation, silently breaking the one artefact a
.NET client is generated from. The route is annotated `-> Any` with an explicit
`response_class`.

**Stage 8 — evaluation — is also built** (`src/seeley_rag/evaluate.py`,
`scripts/05_eval.py`), with its own tests. It joins the SME question set to
`queries.jsonl` and `feedback.jsonl` and writes an HTML report against the §10
gates. **What is missing is the question set**, not the harness: it currently
runs only the ~8 examples in `_context/04-eval/sme-question-template.yaml`, so
those numbers are a smoke test, not a verdict.

**⚠ Next is not a stage — it is `_context/01-plan/production-readiness.md`.**
The pipeline is complete end to end and `parse/vision.py` is the only stub left.
That document is the ordered path from this POC to a production system: what
blocks external exposure (rights, the AGPL PyMuPDF dependency, the SME question
set, API auth), what makes the answers trustworthy (vision for the invisible
26%, the 20% of corrupted tokens, the 38.7% guessed page labels, reranking), and
what makes the service operable. Read it before starting anything — several
items are cheaper done together, and two of build-plan §14's deferred items have
already changed status.

**⚠ Docker is not the run path for this project.** A `Dockerfile`,
`docker-compose.yml` and `.dockerignore` were written and then deleted at the
user's request — containers cost disk this machine does not have spare, and no
image was ever built. Build-plan §13 still lists a Dockerfile as a deliverable;
that item is deliberately dropped. Run everything natively: `.venv` plus the
`make` targets and `scripts/` entry points above. Do not re-create them without
asking.

**The lexicon is hand-maintained and matters.** `config/models.yaml` gained a
whole `VRF` family after the full crawl — 111 article titles referenced it and
every one resolved to `UNKNOWN` or, worse, `RC`. Family matching is
**longest-pattern-wins**, because "VRF REVERSE CYCLE…" contains RC's "Reverse
Cycle" and first-match ordering mislabelled 1,622 pages. After any lexicon edit,
run `--refresh-metadata` (seconds) rather than a full re-parse (hours).

**Note on running tests here.** The sandbox denies pytest's default temp
root, so `tmp_path` tests error with `PermissionError [WinError 5]` — a
pre-existing environment quirk, not a test failure. Run with
`--basetemp=<a writable dir>`.

**The `no_network` tripwire guards `connect`, not `socket.socket`.** It used
to replace the constructor, which broke the moment Stage 4 arrived: `ssl`
does `class SSLSocket(socket)` and cannot subclass a function, and Windows'
`ProactorEventLoop` calls `isinstance(conn, socket.socket)` on its loopback
self-pipe — so every LanceDB test *hung* instead of failing. Loopback is
allowed deliberately; everything off-machine still raises.

**Also worth reading:** `_context/03-research/portal-recon.md` §7 — the portal
publishes a sitemap with 1008 article URLs and per-article `lastmod`, which the
plan says does not exist. It could simplify discovery and enable incremental
sync. Not implemented; flagged as a decision for the user.
