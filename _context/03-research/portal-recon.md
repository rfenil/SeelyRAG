# Portal reconnaissance

Findings from the live Seeley help centre, `https://seeleyinternationalhelp.freshdesk.com`.

**Captured 2026-08-20.** Everything below was verified against the live site
during the acquisition build. Where it contradicts
`_context/01-plan/build-plan.md`, the plan predates the site's current state and
this note wins — see ADR 0004.

---

## 1. robots.txt — the project gate

```
User-agent: *
Disallow: /support/search
Disallow: /support/tickets/
Disallow: /support/login
Disallow: /support/login-verification
Disallow: /login/normal/
Allow: /helpdesk/attachments
Disallow: /helpdesk/
Disallow: /public/tickets/
Disallow: /*/hit$
Sitemap: https://seeleyinternationalhelp.freshdesk.com/support/sitemap.xml
```

**Verdict: the crawl is permitted.**

| Required path | Verdict |
|---|---|
| `/support/solutions` | allowed |
| `/support/solutions/articles` | allowed |
| `/helpdesk/attachments` | allowed |

Two details worth keeping:

- `Allow: /helpdesk/attachments` **precedes** `Disallow: /helpdesk/`. The manual
  PDFs are explicitly permitted; a naive reading of the `Disallow` alone would
  abandon a viable project.
- No `Crawl-delay` is declared, so the configured 1 req/sec stands. If Seeley
  adds one, `scripts/00_check_robots.py` prints it and tells you what to set
  `crawl.rps` to.

This verdict is not permanent. The gate runs on every acquisition run.

---

## 2. Corpus shape

| | Plan said | Actual |
|---|---|---|
| Categories | ~20 | **25** |
| Folders | 143 | **143** ✓ |
| Articles | ~900 | **1008** (per sitemap) |
| `/api/v2/*` | 401 | **401** ✓ — unchanged |
| Sitemap | none | **exists**, 1195 URLs |

Ducted Gas Heating (DGH) breaks down as:

| Folder | ID | Articles |
|---|---|---:|
| Service Guides | 47000783696 | 5 |
| Installation Manuals | 47000783697 | 12 |
| Owners Manuals | 47000783927 | 6 |
| **Diagnostics and Specific Fault Finding** | 47000225980 | **80** |
| Installation / Commissioning Information | 47000225981 | 36 |
| Product Training Programs / Product Information | 47000784683 | 21 |
| YouTube video tutorials | 47000786492 | 9 |
| | | **169** |

The 80-article diagnostics folder matches the plan's "80+ in Ducted Gas Heating
alone" and is the high-value content stream.

---

## 3. DOM structure

### Solutions index — `/support/solutions`

All 143 folders appear on one page, grouped under their category. One request
gets the whole tree.

```html
<div class="cs-s">
  <h3 class="heading accordion-heading">
    <a href="/support/solutions/47000154481">DUCTED GAS HEATING (DGH) SERVICE AND INSTALLATION</a>
  </h3>
  <div class="cs-g-c accordion-content">
    <section class="cs-g article-list"><div class="list-lead">
      <a href="/support/solutions/folders/47000783696" title="Service Guides">
        Service Guides <span class='item-count'>5</span></a>
    </div></section>
```

- Category name: `div.cs-s > h3 a`, category ID from its href.
- Folder name: the **`title` attribute**. The link text has the article count
  appended by the nested `span`, so using `.text()` yields "Service Guides 5".
- The `item-count` span is a free correctness check on pagination.

**Category names are long and shouty.** The real name is `DUCTED GAS HEATING
(DGH) SERVICE AND INSTALLATION`, not `Ducted Gas Heating (DGH)` as configured in
`pilot_categories`. `--categories` therefore matches case-insensitive
*substrings*; exact matching would silently select nothing.

### Folder page — `/support/solutions/folders/{id}`

```html
<div class="breadcrumb">
  <a href="/support/solutions"> Solution home </a>
  <a href="/support/solutions/47000154481">DUCTED GAS HEATING (DGH) ...</a>
</div>
<h2 class="heading">Service Guides</h2>
<div class="c-row c-article-row">
  <div class="ellipsis article-title">
    <a href="/support/solutions/articles/47001247136-tq-service-guide-..." class="c-link">...</a>
  </div>
</div>
```

10 articles per page. The breadcrumb gives the category, so an article page can
be parsed standalone without walking the tree first.

### Article page — `/support/solutions/articles/{id}-{slug}`

- Body: `article.article-body#article-body`.
- Modified date: a plain `<p>Modified on: Thu, 16 Jul, 2026 at 9:50 AM</p>`.
- Attachments sit **outside** `</article>`:

```html
<div class="cs-g-c attachments" id="article-47001247136-attachments">
  <div class="attachment">
    <div class="attach_content"><div class="ellipsis">
      <a href="/helpdesk/attachments/47234382931" class="filename"
         title="644066-M MANUAL SERVICE TQ SERIES.pdf">644066-M MAN... </a>
    </div><div>(2.03 MB) </div></div>
  </div>
</div>
```

**The full filename is in the `title` attribute; the link text is truncated**
("644066-M MAN..."). The filename carries the manual's part number, which is how
an installer recognises a citation, so it must come from `title`.

Body images point at `s3.amazonaws.com/.../attachments/production/...` directly,
not at `/helpdesk/attachments/`, so scoping the attachment selector to that path
excludes them.

Encoding is genuine UTF-8 (`\xe2\x80\x93` is a real en-dash). All file I/O pins
`encoding="utf-8"`; on Windows the default is cp1252 and would corrupt titles.

---

## 4. Pagination — the plan is wrong here

Tested on folder `47000225980` (80 articles):

| Request | Articles | First ID |
|---|---:|---|
| `/support/solutions/folders/47000225980` | 10 | 47001303472 |
| `?page=2` | 10 | 47001303472 — **page 1 again** |
| `/page/2` | 10 | 47001278504 — correct |
| `/page/9` (past end) | 0 | — |

`?page=N` is silently ignored. See ADR 0004. The empty page past the end makes
the plan's "loop until no new links" termination rule correct as written.

---

## 5. Article bodies now carry shared safety boilerplate

The plan quotes the TQ article body as exactly `"Pdf attached TQ Service Guide
644066 M"`. It is now **1087 characters**: a 1026-character safety notice has
been appended, byte-identical across articles.

| Article | Raw | After stripping | Residual |
|---|---:|---:|---|
| TQ Service Guide | 1087 | **60** | "TQ Service Guide Gas Ducted Heater 644066 M ⚠️ Pdf attached," |
| FC7 diagnostic | 3101 | **2074** | real troubleshooting content |

Unstripped, the 200-character stub rule misclassifies ~900 stubs as content. See
ADR 0004.

---

## 6. Triage numbers

Run against the two DGH service guides acquired during the build (174 pages):

| Tier | Pages | Share |
|---|---:|---:|
| A · plain text | 154 | **88.5%** |
| C · diagram-heavy | 11 | **6.3%** |
| B · scanned | 9 | **5.2%** |
| | | |
| **Needs a vision call** | 20 | **11.5%** |

Far cheaper than the plan's estimate of ~15% scanned plus 20–30% diagram-heavy.
If it holds, the vision line in the cost model (§12, estimated $60–75) drops
substantially.

**Treat this as provisional.** It is two 2023-era manuals. The plan is right that
the sample must span the date range — the 2005 *Service Guide TE TX TA4 …* is
exactly the document likely to be scanned, and it is in the same folder
(`47001261667`). Re-run `make triage` after a fuller crawl before committing to
a budget.

**Page labels: 0 of 174 pages expose one.** `page.get_label()` returns nothing
for these documents, so citations will fall back to `page_index + 1`. The
footer-regex fallback in `parse/pagelabels.py` is therefore load-bearing, not
optional, and build-plan §4.5's warning applies in full: verify against a printed
page before trusting any `expected_page` in the eval set.

---

## 7. Recommended follow-ups

Neither is implemented — the specified folder-walk design is what was built —
but both are cheap wins worth considering before the full-corpus crawl.

### 7.1 Use the sitemap for discovery

`https://seeleyinternationalhelp.freshdesk.com/support/sitemap.xml` returns 1195
URLs: **1008 articles**, 158 folders, 27 categories.

Discovery today costs one request for the index plus one per folder page — about
30 requests for the pilot, ~160 for the full corpus. The sitemap is **one
request** for a complete, authoritative article list, and it removes the entire
class of bug that ADR 0004 documents: no pagination to get wrong.

The folder walk still earns its place, because the sitemap carries no category or
folder metadata and that metadata is what prevents cross-product contamination.
The natural combination is sitemap for completeness, folder walk for metadata,
with a diff between the two as a coverage check.

### 7.2 `lastmod` may enable incremental sync

The sitemap carries **1009 distinct `lastmod` values** across 1195 URLs, so the
timestamps are genuinely per-article rather than a single generation stamp.
Article pages independently expose "Modified on".

The build plan states flatly that no API key means no `updated_at` and no
incremental sync. That is too pessimistic. A `lastmod` diff against the manifest
would let a re-crawl fetch only changed articles, turning a 25-minute full crawl
into seconds.

Worth validating before relying on it: confirm that `lastmod` actually moves when
an article's body changes, rather than tracking some unrelated republish event.

---

## Reproducing this

```bash
make robots                                   # section 1
python scripts/02_acquire.py --dry-run        # sections 2, 3
python scripts/01_triage.py                   # section 6
```

The HTML cache under `data/00_raw/html/` holds every page this note was written
from, so the DOM claims can be re-checked without touching the network.
