# Seeley International Installer RAG — End-to-End Build Plan

**Target:** working POC in 2–3 days · pilot corpus · API-only · page-accurate citations, exact fault codes, diagram surfacing.
**Acquisition:** public portal crawl. **No Freshdesk API key is available** — §3 is written accordingly.

*v2 — revised after an adversarial review pass. Changes from v1 are marked ⚠ where they overturn an earlier decision.*

---

## 0. The finding that shapes everything

I crawled the portal before writing this:

| | |
|---|---|
| Solution categories | ~20 |
| Folders | 143 |
| Articles | ~900 |
| Public API (`/api/v2/...`) | **401 — and we have no key** |
| Sitemap | none |

A representative article — *TQ Service Guide Gas Ducted Heater 644066 M* — has a body consisting of exactly this:

> "Pdf attached TQ Service Guide 644066 M"

…plus a link to `/helpdesk/attachments/47234382931`, a 2.03 MB PDF named `644066-M MANUAL SERVICE TQ SERIES.pdf`.

**The knowledge is in the PDFs. The HTML is a card catalogue.** A "scrape the help centre" crawler yields ~900 sentences saying "Pdf attached" and retrieves nothing. Any plan that doesn't put PDF parsing at the centre fails.

There is a second, smaller stream that *is* real text: the diagnostic/fault-finding articles (DGH alone has 80+). Short, Q&A-shaped, written in installer language. Disproportionately valuable per byte — they get their own ingestion path (§4.4) and a retrieval boost (§7.2).

---

## 1. Locked decisions

| Decision | Choice | Consequence |
|---|---|---|
| Acquisition | **Public crawl — the only path.** Adapter interface retained as cheap insurance | See §3.0 for the risk this creates |
| Models | OpenAI + Anthropic direct | `text-embedding-3-large` (3072-d) vectors, Claude Sonnet generation, Haiku routing |
| Runtime | Standalone Python + FastAPI | REST contract is the integration seam to .NET later |
| Scope (v1) | **Ducted Gas Heating (159) + Reverse Cycle (167)** ≈ 326 articles | 2 of 20 categories, the densest two |
| Delivery | API only | No UI in the 3 days |
| Fidelity | Page-accurate citations + exact fault codes + diagram surfacing | Drives §4's parsing tiers |
| Eval | SME-authored questions | §10 template — **Day 0, item one** |
| ⚠ Contextual retrieval | **Cut from the 3 days** | Was v1's Day 2 item; see §5.2 for why, and §14 for when to add it |

### Making the fidelity requirement fit in 3 days

> **Render every PDF page to a PNG at ingest. When you cite a page, return that page's image URL with the text.**

"Show me the wiring diagram" then works for free — the diagram is on the page you're already citing. What you don't get is a cropped figure; the installer sees the whole page. For a tech on a phone that's arguably better: surrounding labels and notes come along. Figure detection is a week's work for a marginal gain. **Defer it.**

---

## 2. Architecture

```
┌───────────────────────────────────────────────────────────────┐
│ STAGE 1 · ACQUIRE                                             │
│   FreshdeskSource (ABC) → PortalScraper                       │
│   →  manifest.jsonl · raw_html/ · pdfs/ (SHA-deduped)         │
└───────────────────────────────────────────────────────────────┘
              ↓                                    ↓
┌──────────────────────────────┐  ┌──────────────────────────────┐
│ STAGE 2a · PARSE PDF         │  │ STAGE 2b · PARSE HTML        │
│  PyMuPDF: text·tables·PNG    │  │  stub detect → drop          │
│  scanned/diagram → vision    │  │  real article → md           │
│  → pages.jsonl               │  │  → articles.jsonl            │
└──────────────────────────────┘  └──────────────────────────────┘
              ↓                                    ↓
┌───────────────────────────────────────────────────────────────┐
│ STAGE 3 · CHUNK                                               │
│   page-anchored · breadcrumbs · atomic tables (capped)        │
│   multi-page table merge · fault-code sweep → codes.jsonl     │
└───────────────────────────────────────────────────────────────┘
              ↓
┌───────────────────────────────────────────────────────────────┐
│ STAGE 4 · INDEX  (LanceDB: vector + native FTS + code table)  │
│   embedding cache keyed by SHA(chunk_text)                    │
└───────────────────────────────────────────────────────────────┘
              ↓
┌───────────────────────────────────────────────────────────────┐
│ STAGE 5 · RETRIEVE   code lookup → dense+BM25 → RRF →         │
│                      stream boost → rerank → top-k            │
└───────────────────────────────────────────────────────────────┘
              ↓
┌───────────────────────────────────────────────────────────────┐
│ STAGE 6 · GENERATE + SERVE  (Sonnet · FastAPI · page images)  │
└───────────────────────────────────────────────────────────────┘
```

### Repo layout

```
seeley-rag/
├── config.yaml · models.yaml
├── src/seeley_rag/
│   ├── acquire/  base.py portal.py attachments.py
│   ├── parse/    triage.py pdf.py html.py vision.py pagelabels.py
│   ├── chunk/    chunker.py tables.py codes.py
│   ├── index/    build.py store.py embed_cache.py
│   ├── retrieve/ query.py hybrid.py rerank.py
│   ├── generate/ answer.py prompts.py
│   └── api/      main.py models.py
├── eval/  questions.yaml run_eval.py report.py
├── data/  raw/ pdfs/ pages/ page_images/ index/ embed_cache/
└── scripts/  00_triage.py … 05_eval.py
```

```bash
python -m venv .venv && source .venv/bin/activate
pip install httpx selectolax pymupdf lancedb openai anthropic \
            pydantic fastapi uvicorn tenacity rich pyyaml pandas
```

⚠ **No `tantivy`.** LanceDB shipped a native Rust FTS and is removing Tantivy support; the legacy path is local-disk only and fully reindexes on every write. Use `tbl.create_fts_index("text")` on the default native path.

⚠ **PyMuPDF is AGPL-3.0.** Fine for a POC. Before this is served to anyone outside your org, either buy the Artifex commercial licence or swap to `pypdfium2` + `pdfplumber` (BSD/MIT, pdfplumber does tables). Logged in §14 — decide it deliberately, don't discover it at launch.

---

## 3. Stage 1 — Acquisition

### 3.0 ⚠ No API key — read this before anything else

The crawl is now the only path, which makes two things load-bearing:

1. **`robots.txt` is a gate, not a checkbox.** Fetch it as the literal first action of Day 0. **If it disallows `/support/solutions/`, the acquisition stage is dead** and no amount of engineering fixes it. Pre-agree the fallback *now*, before anyone writes a scraper: (a) ask Seeley for an API key or a bulk export of the manual PDFs, (b) ask for written permission to crawl. There is no plan C — decide this at hour zero, not on Day 2.
2. **No key means no `updated_at` and no incremental sync.** Change detection is limited to re-crawling and comparing content hashes. At ~900 articles that's a 25-minute job, so it's fine — but it rules out near-real-time freshness, and you should say so to stakeholders rather than let them assume it.

Keep the `FreshdeskSource` ABC anyway. It's 30 minutes, and if a key ever appears the swap is one file.

### 3.1 The adapter

```python
# acquire/base.py
from abc import ABC, abstractmethod
from pydantic import BaseModel

class Article(BaseModel):
    id: str; title: str; url: str
    category: str; folder: str
    body_html: str; body_text: str
    updated_at: str | None
    attachments: list[dict]      # [{id, filename, url, size}]

class FreshdeskSource(ABC):
    @abstractmethod
    def list_folders(self) -> list[dict]: ...
    @abstractmethod
    def list_articles(self, folder_id: str) -> list[dict]: ...
    @abstractmethod
    def get_article(self, article_id: str) -> Article: ...
```

### 3.2 Portal scraper

```python
# acquire/portal.py
import httpx, time, re, hashlib
from pathlib import Path
from selectolax.parser import HTMLParser

BASE = "https://seeleyinternationalhelp.freshdesk.com"

class PortalScraper(FreshdeskSource):
    def __init__(self, rps=1.0, cache_dir="data/raw"):
        self.client = httpx.Client(
            headers={"User-Agent": "SeeleyInstallerBot/0.1 (+shlok@rostered.ai)"},
            timeout=30, follow_redirects=True)
        self.delay, self.cache_dir = 1.0 / rps, Path(cache_dir)

    def _get(self, path: str) -> str:
        key = self.cache_dir / (hashlib.sha1(path.encode()).hexdigest() + ".html")
        if key.exists():                    # you WILL re-run this five times today
            return key.read_text()
        time.sleep(self.delay)
        r = self.client.get(BASE + path); r.raise_for_status()
        key.parent.mkdir(parents=True, exist_ok=True); key.write_text(r.text)
        return r.text

    def list_folders(self):
        tree = HTMLParser(self._get("/support/solutions"))
        return [{"id": m.group(1), "url": a.attributes["href"], "name": a.text(strip=True)}
                for a in tree.css("a[href*='/support/solutions/folders/']")
                if (m := re.search(r"/folders/(\d+)", a.attributes["href"]))]
```

Non-negotiable rules:

1. **Cache every fetch to disk, keyed by URL.** Without it you wait 25 minutes per iteration and hammer someone else's server.
2. **1 req/sec, single-threaded.** ~900 articles + ~600 PDFs ≈ 25 min. Concurrency buys nothing and risks a block — and with no API key, getting blocked ends the project.
3. **Honest User-Agent with a contact address.**
4. Folder pagination: `?page=N`, loop until a page yields no new article links.
5. On any 429 or 403, **stop immediately** and escalate to a human. Do not retry into a block.

### 3.3 Attachments

`/helpdesk/attachments/{id}` **302s to S3** — `follow_redirects=True` is required.

- Save as `pdfs/{attachment_id}.pdf`
- **SHA-256 dedupe.** The same manual is attached across multiple folders; expect 15–30% duplication. Parsing a 200-page manual four times is the easiest hour to waste here.
- Keep `doc_id → [article_ids]` so a chunk from a shared manual cites whichever article the user arrived from.

### 3.4 Metadata is free — take all of it

Folder and category names give `product_family` and `doc_type` with near-perfect accuracy at zero LLM cost. Article titles carry model codes (`TQ`, `TH`, `THM`, `CQ3`, `CQ4`, `TE`, `TX`, `TA4`, `TE4`, `TA5`…).

Hand-write `config/models.yaml`: model code → product family → aliases. One hour, and it is the backbone of §7's filtering. **A TQ fault code answered from a Climate Wizard manual is the failure that permanently destroys installer trust.** This file is what prevents it.

**Deliverable:** `manifest.jsonl`, `pdfs/` deduped, printed summary (articles, PDFs, bytes, dupes dropped, failures).

---

## 4. Stage 2 — Parsing

Where the project is won or lost. Budget the most time here.

### 4.1 ⚠ Triage on Day 0, not Day 1

v1 buried this mid-Day-1. Wrong: **you can hand-download six representative PDFs from a browser in 20 minutes and triage them before a line of scraper exists.** Do it on Day 0. Span the date range — a 2023 manual, a 2015 one, and a 2005 one (e.g. *Service Guide TE TX TA4 … April 2005*).

```python
# parse/triage.py
import fitz
def triage(pdf_path):
    doc = fitz.open(pdf_path)
    out = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        imgs = page.get_images()
        out.append({
            "page": i,
            "chars": len(text.strip()),
            "n_images": len(imgs),
            "has_text_layer": len(text.strip()) > 100,
            # Tier C = has text but is picture-dominated
            "diagram_heavy": len(text.strip()) < 600 and len(imgs) >= 1,
            "label": page.get_label(),      # printed page number, see §4.5
        })
    return out
```

Report **three** fractions, not one: `%scanned`, `%diagram_heavy`, `%plain_text`. ⚠ v1 only modelled the scanned fraction and therefore under-budgeted vision by 2–3× (§12, §13).

This 20-minute step sets your cost and time budget for the next two days.

### 4.2 Three-tier parsing

**Tier A — digital text.** PyMuPDF per page: `page.get_text("text")`, `page.find_tables()`, `page.get_pixmap(dpi=150)`.

⚠ **`find_tables()` is edge-detection-heavy — 0.5–2s/page.** At ~10k pages that is 1.5–5 hours on its own. Gate it: only run it on pages whose text contains a table signal (≥3 lines with 2+ runs of 2+ spaces, or a fault-code regex hit). And multiprocess the parse across cores.

**Tier B — scanned** (`has_text_layer == False`). Page PNG → Claude Sonnet vision:

> Transcribe this service-manual page to markdown. Preserve all tables exactly, including every fault/error code and its description. Describe any wiring diagram or exploded parts view in one sentence prefixed `[DIAGRAM]`. Output only the transcription.

**Tier C — diagram-heavy** (has text, picture-dominated). Keep extracted text, add a one-line vision caption so the page is findable by "TQ wiring diagram". ⚠ **Tier C is a vision call too** — in illustrated service manuals it can be 20–30% of pages, plausibly more volume than Tier B. Budget it explicitly.

### 4.3 Every page gets a PNG

All tiers. 150 DPI. ~150–250 KB/page; ~10k pilot pages ≈ 2 GB. Fine locally, object storage in production.

### 4.4 ⚠ HTML articles need their own ingestion path

v1 named the diagnostic articles as high-value, gave them a `content_stream` field and a retrieval boost — then never actually ingested them. The boost would have applied to zero rows. Fix:

- Body under ~200 chars **and** has an attachment → `is_stub: true`, **do not index as content** (a "Pdf attached" chunk pollutes retrieval). Its title/metadata still decorate the PDF's chunks.
- Otherwise → strip nav/boilerplate, convert to markdown, chunk normally with a synthetic `doc_id`, `page_no: null`, `page_image: null`, `content_stream: "diagnostic_article"`, citing the article URL directly.

### 4.5 ⚠ Page numbering — reconcile it or eval will lie to you

`enumerate(doc)` is 0-based. Citations render "p.42". SMEs write `expected_page: 42` reading the **printed** number off the page. Service manuals have front matter, so printed ≠ index, routinely by 4–10. A `±1` eval tolerance hides an off-by-one but not a front-matter offset — so **page accuracy fails corpus-wide for a reason that looks like a retrieval bug** and you burn Day 3 chasing it.

Store both: `page_index` (0-based, internal) and `page_label` (`page.get_label()`, falling back to a footer regex, falling back to `page_index+1`). **Cite the label.** State in the SME template which one they're giving you.

**Deliverable:** `pages.jsonl` — `{doc_id, page_index, page_label, text, tables[], tier, image_path, source_article_ids[], product_family, doc_type, title, source_url}`.

---

## 5. Stage 3 — Chunking

### 5.1 Rules

1. **Never cross a page boundary** (except rule 4). Page provenance is a requirement; make violating it structurally impossible.
2. Target ~800 tokens, hard max ~1,200, ~120-token overlap within a page.
3. A detected table is one atomic chunk — ⚠ **capped at ~6,000 tokens.** `text-embedding-3-large` hard-caps at 8,191 and Cohere rerank truncates near 4k. "One chunk whatever its size" would 400 on exactly the most valuable content in the corpus. Split on row boundaries, repeat the header in each part.
4. ⚠ **Merge multi-page tables.** Fault-code tables run 2–4 pages with the header only on the first. Per-page detection shreds them — which is the failure v1 claimed to have mitigated and hadn't. If consecutive pages have matching column geometry and the later one has no header row, merge into one chunk anchored to the first page with a `page_range` field.
5. Breadcrumb prefix on every chunk:
   `Ducted Gas Heating > Service Guides > TQ Service Guide 644066-M > p.42`

Rule 5 alone measurably lifts retrieval: it puts the product name in the embedded text of every chunk, including chunks whose body never names the product.

### 5.2 ⚠ Contextual retrieval — cut from the 3 days

v1 scheduled this and got three things wrong, so here is the honest version:

- Anthropic's ~35% retrieval-failure reduction is measured with the **whole document** in the cached prompt, not a summary. The summary variant v1 described is a different, weaker technique that doesn't inherit the number.
- The real technique needs a doc-summary or whole-doc pass that **no stage in v1 produced**, and needs all chunks of a document processed consecutively or the prompt cache thrashes.
- The `$8` estimate used Claude 3 Haiku pricing. At current rates it's **$20–60** (§12).
- v1 scheduled it *after* embedding — but the prefix has to be in the text that gets embedded. As written it would have shipped an index whose vectors didn't contain the context.

It's a genuine quality win and it's **the first thing to add after the POC** (§14). It is not a 3-day item. Ship hybrid + rerank first; you may find you don't need it.

### 5.3 Fault-code extraction — the highest-value 90 minutes here

Installers search by code, and vector search is *bad* at codes: `E:04` and `E:05` are near-identical in embedding space and catastrophically different in meaning.

```python
CODE_PATTERNS = [
    r"\b[EFH][\s:.-]?\d{1,2}\b",     # E:04, F.12, H-3
    r"\bfault\s+code\s+(\w+)\b",
    r"\berror\s+code\s+(\w+)\b",
    r"\b\d{1,2}\s+flash(?:es)?\b",   # DGH flash codes
]
```

Sweep every page (especially table chunks) and emit:

```json
{"code": "E:04", "product_family": "DGH", "model_series": ["TQ"],
 "meaning": "Flame sensing fault — no flame detected within trial for ignition",
 "doc_id": "...", "page_label": "42", "source_url": "..."}
```

Haiku pulls `meaning` from the surrounding table row. At query time a detected code hits this table **first** and gets pinned into context. Exact lookup beats semantic search at exact-lookup problems — don't make the embedding model do arithmetic.

---

## 6. Stage 4 — Indexing

**LanceDB.** Embedded, no server, native vector + FTS + hybrid + built-in rerankers, one pip install. For 2–3 days the ops cost of Qdrant or Postgres+pgvector is a day you don't have. *Production swap:* pgvector to live inside the RosteredAI Postgres estate, or Qdrant past single-node. Keep the store behind a thin protocol so it's a one-file change.

```python
schema = {
  "chunk_id": str, "doc_id": str, "text": str, "vector": Vector(3072),
  "page_index": int | None, "page_label": str | None, "page_range": str | None,
  "page_image": str | None, "source_url": str, "article_title": str,
  "product_family": str, "model_series": list[str], "doc_type": str,
  "category": str, "folder": str, "tier": str, "is_table": bool,
  "content_stream": str,          # "pdf" | "diagnostic_article"
}
```

⚠ **Cache embeddings keyed by `sha256(final_chunk_text)`.** Day 3's fixes (table boundaries, chunk sizing) force a re-chunk → re-embed → re-index cycle two or three times. With the cache, only genuinely changed chunks re-embed and the loop drops from hours to minutes. This one hour on Day 2 is what makes Day 3 possible.

Embeddings: `text-embedding-3-large`, batch 256, $0.13/M. Pilot ≈ 12–15M tokens ≈ **$2**. Don't spend engineering time optimising a rounding error.

Build both the vector index and `create_fts_index("text")`.

---

## 7. Stage 5 — Retrieval

### 7.1 Query understanding — one Haiku call, ~200ms

```json
{"product_family": "DGH", "model_series": ["TQ"], "fault_codes": ["E:04"],
 "intent": "fault_diagnosis", "wants_diagram": false,
 "rewritten_query": "TQ series gas ducted heater E:04 flame sensing fault diagnosis"}
```

**Soft-boost inferred product family, don't hard-filter.** If an installer says "the ducted heater is throwing E:04" and the classifier guesses wrong, a hard filter returns nothing and the system looks broken. Hard-filter only when the user names a model explicitly.

### 7.2 The cascade

1. **Code lookup** — query contains a fault code → exact hit on the code table → pin into context.
2. **Dense** — top 30.
3. **BM25** — top 30. Catches model numbers, part numbers, code strings.
4. **RRF fusion** — `score = Σ 1/(60 + rank_i)`. Parameter-free, works.
5. ⚠ **Apply boosts here, before truncation** — product-family match, and `content_stream == "diagnostic_article"` × 1.2. v1 applied the stream boost *after* reranking to top-8, where it could not promote anything into the list. Boost fused scores, then truncate.
6. **Rerank to top 5–8.** Cohere `rerank-v3.5` if you can get a key — ⚠ **add this to Day 0**, because the Haiku listwise fallback costs ~$0.016/query in rerank input alone and roughly doubles the per-query figure in §12.

---

## 8. Stage 6 — Generation

**Claude Sonnet.** The system prompt must enforce:

- Answer **only** from provided context. Missing → say so, and name the manual that would have it.
- Inline citations `[1]`, `[2]` on every factual claim, resolving to `{title, page_label, url}`.
- Surface the page image when the answer depends on a diagram, table, or exploded view.
- **Exact values verbatim** — gas pressures, torque figures, part numbers, voltages. Never round, never paraphrase a number.
- **Safety.** This system answers questions about gas carriage, combustion, and mains electrical work. Where a procedure touches those, state it must be performed by an appropriately licensed technician. Never synthesise a procedure absent from the source. Where the manuals contradict the model's prior, **the manuals win**.

```json
{
  "query_id": "q_01J8...",
  "answer": "The E:04 fault on TQ series indicates ... [1]",
  "citations": [{
    "n": 1, "title": "TQ Service Guide Gas Ducted Heater 644066-M",
    "page_label": "42",
    "doc_url": "https://.../helpdesk/attachments/47234382931",
    "article_url": "https://.../articles/47001247136-tq-service-guide...",
    "page_image": "/pages/47234382931/41.png",
    "snippet": "E:04 — Flame sensing fault..."
  }],
  "confidence": "high", "product_family": "DGH", "latency_ms": 2840
}
```

Every citation resolves to a page image **and** a link back to the source Freshdesk article, so an installer verifies in two taps. That's what earns trust.

---

## 9. Stage 7 — API

```
POST /ask       {query, product_hint?, top_k?, stream?}  → answer + citations + query_id
POST /search    {query, filters}                          → raw chunks (debug)
POST /feedback  {query_id, rating, comment}               → ack
GET  /pages/{doc_id}/{page_index}.png
GET  /docs      → corpus inventory
GET  /health
```

⚠ `/ask` **must return `query_id`** — v1's `/feedback` took one that nothing produced. Log `query_id` alongside the query, retrieved chunk IDs and the answer, to JSONL. Your first week of real queries is worth more than any synthetic eval.

---

## 10. Stage 8 — Evaluation

**Start the SME question set on Day 0.** It's the only critical-path item you don't control.

Ask for **60 questions**:

```yaml
- id: dgh-001
  question: "TQ heater showing E:04, what do I check?"
  product_family: DGH
  model: TQ
  expected_source: "644066-M MANUAL SERVICE TQ SERIES.pdf"
  expected_page: 42        # the PRINTED page number on the page itself
  must_mention: ["flame sensing", "trial for ignition"]
  must_not_say: ["replace the gas valve first"]
  category: fault_diagnosis
```

Spell out in the brief that `expected_page` is the **printed** number (see §4.5).

Coverage: ~40% fault diagnosis · ~25% installation/commissioning · ~15% spec lookup (clearances, gas pressures, electrical) · ~10% "show me the diagram" · ~10% **unanswerable from the corpus**.

That last 10% matters more than people expect. A RAG that confidently invents an answer for a model it has never seen is worse than no RAG, because a licensed installer may act on it.

| Metric | How | Gate |
|---|---|---|
| Retrieval recall@8 | `expected_source` in retrieved set | ≥ 0.85 |
| Page accuracy | `expected_page` (±1) among cited labels | ≥ 0.70 |
| Citation validity | Cited page exists and contains the claim (LLM judge) | ≥ 0.95 |
| Answer correctness | LLM judge vs `must_mention` / `must_not_say` | ≥ 0.80 |
| Refusal on unanswerables | Correctly declined | ≥ 0.90 |
| p95 latency | Timed | < 6s |

`python scripts/05_eval.py` → HTML report. Re-run after every retrieval change. **Never tune retrieval by vibes** — you'll fix one query and silently break five.

---

## 11. The schedule

⚠ v1's Day 1 counted only *build* hours and ignored machine time; Day 2 summed to 9.5h before debugging. Revised, with the long parse moved to an overnight run.

### Day 0 — ~1 hour, today

- [ ] **`robots.txt`.** First action. If `/support/solutions/` is disallowed, **stop and escalate** — §3.0 has the fallbacks and there is no plan C.
- [ ] Email the SME the §10 template. Unblocks Day 3.
- [ ] Hand-download 6 PDFs (2023 / 2015 / 2005) → run triage → **record `%scanned`, `%diagram_heavy`, `%plain_text`.**
- [ ] Keys into `.env`: OpenAI, Anthropic, **and Cohere** (§7.2). Check API tier rate limits — ~1,500+ Sonnet vision calls on a fresh org will throttle.

### Day 1 — acquisition and parsing

| | |
|---|---|
| 0.5h | Scaffold, deps, config |
| 1.5h | `FreshdeskSource` ABC + `PortalScraper`, folder/article listing |
| 1h | Article parser + attachment downloader with SHA dedupe |
| 0.5h | **Pilot crawl runs** (~326 articles, 25 min unattended — build `models.yaml` during it) |
| 1h | `models.yaml` lexicon from folder names |
| 2h | PyMuPDF parser: text, gated `find_tables()`, PNG render, page labels — **multiprocessed** |
| 1h | Vision path (Tier B + Tier C) |
| 0.5h | Smoke-test on 10 PDFs, verify page labels look right |
| → | **Kick off the full parse as an overnight job.** 10k pages × (gated tables + render + ~2–4k vision calls at concurrency 10) is 2–5 machine-hours. It does not fit in the working day and doesn't need to. |

**Exit gate: the parse job is *running* and verified correct on a 10-PDF sample.** If the parser is wrong, fix it before you sleep — everything downstream inherits the damage.

### Day 2 — index, retrieve, serve

| | |
|---|---|
| 0.5h | Check overnight parse, spot-check `pages.jsonl` |
| 1.5h | Chunker: page anchoring, breadcrumbs, atomic+capped tables, multi-page merge |
| 1.5h | **Fault-code extraction + code table** |
| 0.5h | HTML diagnostic-article ingestion path (§4.4) |
| 1h | Embed + build LanceDB, **with the SHA embedding cache** |
| 3h | Retrieval cascade: query understanding, dense, BM25, RRF, boosts, rerank |
| 1h | FastAPI `/ask` + `/pages` + answer prompt + citation resolution |

⚠ `/search` and `/feedback` moved to Day 3. ⚠ Contextual enrichment cut entirely (§5.2). That takes Day 2 from 9.5h to ~9h with the retrieval cascade given a realistic 3h instead of 2h.

**Exit gate: `curl` a real question, get a cited answer with a page image.**

### Day 3 — make it true

| | |
|---|---|
| 1h | Load SME questions, build eval harness |
| 1h | **First eval run — expect to be disappointed.** That's the point. |
| 3h | Fix the top 3 failure modes. Usually: page-label offsets, product misrouting, table boundaries. *The embedding cache is what makes each re-index minutes, not hours.* |
| 1h | Re-run eval, confirm gates |
| 0.5h | `/search` + `/feedback` |
| 1h | Dockerfile, README, `/docs` inventory |
| 0.5h | Demo script: 8 questions showing fault code → cited answer → wiring diagram |

---

## 12. Cost

**Pilot build (~326 articles, est. 10k pages):**

| Item | Est. |
|---|---|
| Embeddings (~12–15M tokens) | ~$2 |
| Vision — Tier B, scanned (~15% of pages) | ~$25 |
| ⚠ Vision — Tier C, diagram-heavy (~20–30% of pages) | ~$35–50 |
| Code-meaning extraction (Haiku) | ~$2 |
| **Total one-off** | **~$65–80** |

⚠ v1 said $40–75 by modelling only Tier B. **Treat the vision line as a function of your Day 0 triage numbers, not as a fixed figure** — it's the only estimate here that can move by 3×.

**Per query:** ~$0.02–0.03 (Haiku routing + Sonnet over ~6k in / ~500 out) with Cohere rerank; **~$0.04–0.05** on the Haiku rerank fallback. 1,000 queries/day ≈ **$25–50/day**. Prompt caching on the system prompt cuts this once traffic is real.

Full corpus (~900 articles) ≈ 2.8× build cost ≈ **$180–230** — and sub-linear in practice, since the pilot deliberately took the two densest categories. Not a constraint. Don't spend engineering hours optimising a $200 line item.

---

## 13. What will actually break

Ranked by probability × damage:

1. ⚠ **`robots.txt` disallows the crawl.** With no API key this is now risk #1, not a footnote. Project-ending until a human resolves it with Seeley. *Mitigation: check it in the first five minutes of Day 0.*
2. **Vision volume 3× the estimate.** If triage shows scanned + diagram-heavy above ~50%, cost and Day 1 machine time both blow out. *Mitigation:* triage Day 0; if it's bad, restrict the pilot to post-2013 manuals and document the gap explicitly.
3. **Cross-product contamination.** DGH answer sourced from a Climate Wizard manual. Destroys trust instantly and permanently. *Mitigation:* `models.yaml` + product boosting + eval cases that specifically probe it.
4. **Fault-code tables shredded** — by chunking, by the multi-page split, or by the 6k cap. `E:04` and its meaning land in different chunks and neither retrieves. *Mitigation:* all three of atomic chunks, multi-page merge, and the separate code index. Not one of them.
5. **Page-label offset** makes page accuracy fail corpus-wide while looking like a retrieval bug. *Mitigation:* §4.5, plus print 10 label/index pairs on Day 1 and eyeball them.
6. **Duplicate PDFs parsed N times.** Wasted machine hours, duplicate results crowding out diversity. *Mitigation:* SHA-256 dedupe at download.
7. **Rate-limited or blocked.** With no key there's no fallback channel. *Mitigation:* 1 rps, disk cache, honest UA, stop-on-429.
8. **SME questions arrive late.** You end Day 3 unable to say whether it works. *Mitigation:* Day 0 item one; bootstrap a synthetic set as insurance.
9. **Confident wrong answer on gas/electrical procedure.** Lowest probability, highest consequence. *Mitigation:* strict grounding, refusal eval cases, licensed-technician framing, human review before any real installer touches it.

---

## 14. Deferred — in priority order

1. **Contextual retrieval, done properly** (whole document in a cached prompt, chunks grouped per doc, enrichment *before* embedding). The highest-value single upgrade. ~1 day + $20–60.
2. **PyMuPDF licence resolution** — commercial licence or swap to pypdfium2/pdfplumber. Must happen before external serving.
3. Figure detection and cropping (page images cover it for now).
4. Incremental sync — needs the API key; full re-crawl is fine at this size.
5. The other 18 categories.
6. UI · multi-turn memory · .NET/Ocelot integration · auth, tenancy, rate limiting · fine-tuned embeddings · model/part knowledge graph.

Every one is additive on what Day 3 produces. Nothing above requires rewriting the pipeline — which is the real test of whether a POC plan was any good.

---

## 15. Before this reaches a real installer

Two things, neither technical:

1. **Content rights.** The manuals are Seeley International's IP. Public availability is not a licence to ingest, store and redistribute through a derived product. With no API key we're also crawling without an explicit relationship — which makes this more pressing, not less. If Seeley is a client, get it in writing. If not, get it in writing *first*.
2. **Liability framing.** This answers questions about gas carriage and mains electrical work. Any surface needs explicit framing as a reference aid for licensed technicians, not an instruction authority — and every answer cites the manual so the installer verifies against the source of truth.

---

## 16. Next step

Say the word and I'll scaffold the repo — `FreshdeskSource` adapter, caching scraper with SHA dedupe, and the triage script — so you can run the Day 0 triage and see the scanned/diagram-heavy split within the hour. Those two percentages are the first real fork in this plan, and Day 1's timeline and §12's budget both hang off them.

But check `robots.txt` first. With no API key, that one file decides whether there's a project.
