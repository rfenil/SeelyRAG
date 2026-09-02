# Requirements — locked decisions, scope and constraints

The authoritative specification is `_context/01-plan/build-plan.md`. This page
is the short version: what was agreed, what is deliberately excluded, and what
cannot be changed without a new ADR.

---

## What we are building

A retrieval-augmented question-answering system over Seeley International's
public Freshdesk help centre, so that HVAC installers can ask fault-diagnosis and
installation questions and get answers **cited to the exact manual page**.

Delivered as a REST API. The integration seam to .NET later is that REST
contract.

---

## The four corpus facts

These are given, not assumptions. They drive nearly every design decision.

1. **The knowledge is in the PDFs, not the HTML.** ~900 articles across 143
   folders, but a typical article body is one sentence plus a link to a 2 MB
   service manual. The HTML is a card catalogue. An acquisition layer that
   captures article text but not attachments captures nothing of value.
2. **A minority of articles are real content.** The diagnostic and fault-finding
   articles — 80+ in Ducted Gas Heating alone — have substantial bodies, are
   written in installer language, and are disproportionately valuable per byte.
   They must be distinguished from the stubs, not discarded with them.
3. **No Freshdesk API key is available.** `/api/v2/*` returns 401. Acquisition is
   a polite public crawl, and that is the only path. See ADR 0002.
4. **The same PDF is attached to multiple articles.** Expect 15–30% duplication.
   Deduplication is a correctness requirement, not an optimisation.

---

## Locked decisions

| Decision | Choice |
|---|---|
| Acquisition | Public crawl — the only path. Adapter interface retained (ADR 0002). |
| Pilot scope | Ducted Gas Heating + Reverse Cycle, ~326 articles: the two densest of 25 categories. |
| Models | OpenAI `text-embedding-3-large` (3072-d); Claude Sonnet generation; Haiku routing. |
| Vector store | LanceDB — embedded, native vector + FTS, behind a thin protocol for later swap. |
| Runtime | Python 3.11+, FastAPI. |
| Delivery | API only. No UI in the three days. |
| Fidelity | Page-accurate citations, exact fault codes, diagram surfacing via page images. |
| Evaluation | SME-authored question set, 60 questions. Day 0, item one. |

### Explicitly cut

- **Contextual retrieval.** A genuine quality win and the first thing to add
  after the POC, but not a three-day item. See build-plan §5.2 for why the
  usual estimate of its cost and benefit does not survive scrutiny.
- **Figure detection and cropping.** Page images cover the need for now.
- **Incremental sync.** Needs an API key; full re-crawl is fine at this size.
  (Though see `03-research/portal-recon.md` §7.2 — the sitemap may make this
  cheaper than assumed.)
- **The other 23 categories.**

---

## Hard constraints

**Engineering:**

- Python 3.11+, `pip` + `venv`. Not uv, not Poetry, not conda.
- `black` + `flake8` + `isort` + `pytest`. Not ruff.
- `src/` layout; the package is `seeley_rag`.
- Local filesystem only — no S3, no DVC, no database server (ADR 0003).
- No secrets in git. `.env` is ignored; `.env.example` is committed with every
  key present and every value blank.

**Crawl etiquette — these are correctness requirements, not preferences:**

- 1 req/sec, single-threaded.
- Every fetch cached to disk.
- Honest User-Agent with a contact address.
- Stop immediately on 429 or 403. Never retry into a block.
- `robots.txt` is a hard gate, checked before the first fetch of every run.

**Data layout:**

- `data/00_raw/` is write-once. Nothing may modify or delete a file under it.
- PDFs are content-addressed by SHA-256.
- Every path comes from `paths.py`.

---

## Quality gates

From build-plan §10. Retrieval is never tuned by vibes — fixing one query while
silently breaking five is the failure mode this exists to prevent.

| Metric | Gate |
|---|---|
| Retrieval recall@8 | ≥ 0.85 |
| Page accuracy (±1 on the **printed** label) | ≥ 0.70 |
| Citation validity | ≥ 0.95 |
| Answer correctness | ≥ 0.80 |
| Refusal on unanswerable questions | ≥ 0.90 |
| p95 latency | < 6s |

Roughly 10% of the SME question set must be **unanswerable from the corpus**.
A system that confidently invents an answer for a model it has never seen is
worse than no system, because a licensed installer may act on it.

---

## Non-technical blockers

Neither is an engineering problem, and both must be resolved before this reaches
a real installer:

1. **Content rights.** The manuals are Seeley's intellectual property. Public
   availability is not a licence to ingest, store and redistribute them through
   a derived product. Crawling without an explicit relationship makes this more
   pressing, not less. Get it in writing.
2. **Liability framing.** This answers questions about gas carriage, combustion
   and mains electrical work. Any surface needs explicit framing as a reference
   aid for licensed technicians, not an instruction authority — and every answer
   cites the manual so the installer verifies against the source of truth.

---

## Current build scope

This repository currently implements **Stage 1 (acquisition) and Stage 0 (PDF
triage) only**. Every other stage is a stub that imports cleanly and raises
`NotImplementedError` naming the build-plan section that specifies it. See
`CLAUDE.md` for what is done and what comes next.
