# Seeley Installer RAG

Retrieval-augmented question answering over [Seeley International's public help
centre](https://seeleyinternationalhelp.freshdesk.com), so HVAC installers can
ask fault-diagnosis and installation questions and get answers **cited to the
exact manual page**.

> **Status: Stage 2 of 7.** Acquisition, triage and parsing are implemented and
> tested; the corpus is crawled and parsed to 13,156 pages. Vision transcription
> and everything downstream — chunking, indexing, retrieval, generation, the API
> — remain stubs. See [Current state](#current-state).

---

## Why this design

The single finding that shapes the whole project: **the knowledge is in the
PDFs, not the HTML.** There are ~900 articles across 143 folders, but a typical
article body reads, in its entirety, "Pdf attached TQ Service Guide 644066 M" —
plus a link to a 2 MB service manual. The help centre is a card catalogue.

A crawler that captures article text and not attachments retrieves ~900
sentences saying "Pdf attached". So attachment handling, PDF parsing and
page-accurate citation sit at the centre of the design rather than at the edge.

Three more facts follow from it:

- **A minority of articles are genuinely content** — the diagnostic and
  fault-finding articles, 80+ in Ducted Gas Heating alone. Short, written in
  installer language, disproportionately valuable per byte. They get their own
  ingestion path, so they must be distinguished from stubs rather than discarded
  with them.
- **There is no API key.** `/api/v2/*` returns 401. Acquisition is a polite
  public crawl and there is no alternative, which makes `robots.txt` a project
  gate and crawl etiquette a correctness requirement ([ADR 0002](_context/02-decisions/0002-crawl-instead-of-api.md)).
- **The same manual is attached to many articles** — 15–30% duplication. PDFs
  are stored content-addressed by SHA-256, so deduplication falls out for free.

Full specification: [`_context/01-plan/build-plan.md`](_context/01-plan/build-plan.md).

---

## Quick start

Requires Python 3.11 or newer.

```bash
make init      # create .venv, install dependencies, create data/ directories
make robots    # THE GATE: may we crawl at all? Run this first.
make test      # 328 tests, no network access
```

Then acquire a small sample:

```bash
python scripts/02_acquire.py --limit 3 --dry-run   # show the plan, fetch nothing
python scripts/02_acquire.py --limit 10            # crawl 10 articles + their PDFs
make triage                                        # classify the PDFs you just got
```

To acquire the **whole site** — all 25 categories, ~1,008 articles and every
attached PDF:

```bash
python scripts/02_acquire.py --categories
```

`--categories` with no values after it means "every category". Expect roughly
40-50 minutes at the mandatory 1 req/sec, and a few GB of PDFs.

**It resumes.** If the run dies — network drop, a kill, a reboot — re-run the
same command. It reads the manifest, skips every article already acquired
without fetching its page, skips every PDF already on disk without transferring
it, and appends only new rows. Nothing is duplicated and nothing restarts from
the beginning. Use `--overwrite` only when you genuinely want to start again.

<details>
<summary>No <code>make</code> on Windows?</summary>

`winget install ezwinports.make`, or run the recipe bodies directly — each
target is one or two plain commands:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt -r requirements-dev.txt
.venv\Scripts\python -m pip install -e .
.venv\Scripts\python -c "from seeley_rag.paths import ensure_dirs; ensure_dirs()"
```
</details>

---

## Pipeline

```
STAGE 1 · ACQUIRE          FreshdeskSource -> PortalScraper
  implemented              manifest.jsonl · html/ cache · pdf/ SHA-deduped
        |
        +-- STAGE 2a · PARSE PDF     PyMuPDF: text, tables, page PNGs   implemented
        |                            scanned/diagram pages -> vision      stub
        +-- STAGE 2b · PARSE HTML    stubs dropped, real articles -> md implemented
        |
STAGE 3 · CHUNK            page-anchored, breadcrumbed, atomic tables      stub
STAGE 4 · INDEX            LanceDB: vector + native FTS + code table       stub
STAGE 5 · RETRIEVE         code lookup -> dense+BM25 -> RRF -> rerank      stub
STAGE 6 · GENERATE         Claude Sonnet, cited answers + page images      stub
STAGE 7 · SERVE            FastAPI                                         stub
```

Stage 0 (PDF triage) is implemented and runs before Stage 2 to set the vision
budget.

---

## Commands

| Command | What it does |
|---|---|
| `make init` | venv, install, create `data/` directories |
| `make robots` | **The gate.** Checks `robots.txt` against every path the crawl needs |
| `make triage` | Classify PDFs into plain-text / diagram-heavy / scanned tiers |
| `make acquire` | Crawl the pilot categories, download and dedupe PDFs, write the manifest |
| `make lint` | `black --check`, `isort --check`, `flake8` |
| `make test` | Run the test suite |
| `make coverage` | Tests with an 80% floor on `acquire/` |
| `make clean` | Remove derived stages. **Never** touches `data/00_raw` |

`scripts/02_acquire.py` takes `--categories`, `--limit`, `--dry-run`, `--rps`,
`--no-attachments`, `--overwrite`, `--no-resume` and `--progress-every`.

---

## Crawl etiquette

Not politeness — correctness. With no API key there is no fallback channel, so
being blocked ends the project. The following are enforced in code:

- **1 request/second, single-threaded.** Never parallelise the crawl. Page
  fetches and PDF downloads share a single rate limiter, so the limit applies to
  the run as a whole rather than to each component separately.
- **Every fetch cached to disk**, keyed by URL. Re-runs cost disk reads, not
  another 1,500 requests against someone else's production server.
- **Honest `User-Agent`** carrying a contact address.
- **Immediate stop on HTTP 429 or 403.** Never retry into a block.
- **The robots gate runs before the first fetch of every run**, and refuses to
  proceed on an *undetermined* verdict as firmly as on a disallow.

As checked on 2026-08-20 the portal permits this crawl; `Allow:
/helpdesk/attachments` explicitly precedes `Disallow: /helpdesk/`, so the manual
PDFs are fetchable. That verdict can change, which is why the gate is not a
one-off.

---

## Data layout

```
data/
├── 00_raw/          WRITE-ONCE. Never modified after creation.
│   ├── html/        {sha1(url)}.html   — the fetch cache
│   ├── pdf/         {sha256}.pdf       — content-addressed, dedupe for free
│   └── manifest.jsonl
├── 01_interim/      pages.jsonl, page_images/
├── 02_processed/    chunks.jsonl, codes.jsonl
├── 03_index/        vector store
├── cache/           llm/, embeddings/
└── reports/         triage and crawl summaries
```

`data/` is gitignored entirely. Every path in the codebase comes from
`paths.py`; there are no directory string literals anywhere else, which is what
makes a later move to object storage a one-file change
([ADR 0003](_context/02-decisions/0003-local-filesystem-data-layout.md)).

A manifest row carries full provenance — `fetched_at`, source `url`,
`crawler_version` — plus the `is_stub` / `content_stream` classification computed
once at acquisition time so no later stage re-derives it.

---

## Repository layout

```
src/seeley_rag/     settings · paths · logging_conf · exceptions
                    acquire/  base portal attachments manifest robots   IMPLEMENTED
                    parse/    triage IMPLEMENTED; pdf html vision pagelabels  stub
                    chunk/ index/ retrieve/ generate/ api/              stub
scripts/            00_check_robots · 01_triage · 02_acquire  (03-05 stubs)
config/             config.yaml · models.yaml
tests/              328 tests; no test touches the network
_context/           brief · plan · ADRs · research · eval  (documentation only)
```

`_context/` holds everything **about** the project that is not **part** of it.
Nothing in `src/`, `tests/`, `scripts/` or `config/` imports from it. See
[`_context/README.md`](_context/README.md).

---

## Current state

**Implemented and verified against the live portal:**

- Robots gate, with allowed / blocked / undetermined verdicts.
- Portal scraper: category and folder discovery (143 folders, 25 categories),
  paginated article listing, article parsing, disk cache, rate limiting.
- Attachment downloader: follows the 302 to S3, streams, SHA-256 content
  addressing, deduplication.
- Manifest: append-safe JSONL writer (flushed per row), streaming reader,
  validation, summary, and crash-resume with automatic repair of a truncated
  manifest.
- PDF triage reporting all three tier fractions.
- Live scrape-volume reporting in the terminal, plus a durable crawl report.
- Crash-resume: a failed run continues from where it stopped, re-fetching
  nothing and duplicating nothing.

**Quality:** 209 tests passing, 94% coverage on `acquire/` (every module ≥90%),
lint clean.

**Provisional triage numbers** (2 manuals, 174 pages): 88.5% plain text, 6.3%
diagram-heavy, 5.2% scanned — 11.5% needs a vision call, well under the plan's
35–45% estimate. Re-run on a date-spanning sample before trusting the budget;
the 2005-era guides are the ones likely to be scanned.

**Next:** Stage 2 parsing (`parse/pdf.py`), specified in build-plan §4.2–4.5.

**Worth a decision:** the portal publishes a sitemap listing all 1008 articles
with per-article `lastmod`, which the build plan states does not exist. It could
replace folder-walk discovery and enable incremental sync. Not implemented —
see [`_context/03-research/portal-recon.md`](_context/03-research/portal-recon.md) §7.

---

## Before this reaches a real installer

Two blockers, neither technical:

1. **Content rights.** The manuals are Seeley's intellectual property. Public
   availability is not a licence to ingest, store and redistribute them through
   a derived product. Get it in writing.
2. **Liability framing.** This answers questions about gas carriage, combustion
   and mains electrical work. Any surface needs explicit framing as a reference
   aid for licensed technicians, not an instruction authority.

Also unresolved: **PyMuPDF is AGPL-3.0.** Fine for a POC; before external
serving, either buy the Artifex commercial licence or swap to `pypdfium2` +
`pdfplumber`. Decide it deliberately rather than discovering it at launch.
