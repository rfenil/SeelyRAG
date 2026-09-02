# 0004. Correct the build plan where the live portal disagrees with it

## Status

Accepted.

## Context

The build plan was written against a reconnaissance pass of the portal made some
time before implementation. Verifying its claims against the live site on
2026-08-20 turned up two places where the portal no longer behaves as documented.
Both would have caused silent data loss rather than a visible failure, which is
the worst kind of divergence: the crawl reports success and the corpus is wrong.

Full evidence is in `_context/03-research/portal-recon.md`.

### 1. Folder pagination is `/page/{N}`, not `?page=N`

The plan (section 3.2, rule 4) specifies "Folder pagination: `?page=N`, loop
until a page yields no new article links."

Tested against the 80-article folder `47000225980` (*Diagnostics and Specific
Fault Finding*):

| Request | Result |
|---|---|
| `/support/solutions/folders/47000225980` | 10 articles |
| `...?page=2` | **the same 10 articles** — page 1 again |
| `...&#47;page/2` | 10 *different* articles |
| `...&#47;page/9` (past the end) | 0 articles |

`?page=N` is silently ignored. Combined with the plan's own termination rule —
stop when a page yields no new links — a crawler using it fetches page 1, fetches
page 1 again, sees nothing new, and concludes the folder is exhausted. It would
have captured **10 of 80 articles** and reported success.

The folder this hits hardest is the diagnostic fault-finding folder: the
highest-value content in the corpus, and the one stream the plan singles out as
disproportionately valuable per byte.

### 2. Article bodies now carry 1026 characters of shared safety boilerplate

The plan's defining example is the *TQ Service Guide* article, whose body it
quotes in full as:

> "Pdf attached TQ Service Guide 644066 M"

That article's body is now **1087 characters**. Seeley has appended a safety
notice — "⚠️ Safety Notice – Gas Heating Products / This appliance must only be
installed, commissioned, serviced and repaired by suitably qualified and
licensed personnel..." — running to 1026 characters and **byte-identical across
articles**.

The stub rule is `body_char_count < 200 AND has an attachment`. Against the
current portal that rule classifies the canonical stub as a *content article*,
because 1087 > 200. Roughly 900 card-catalogue articles would be ingested as
content, and the index would fill with near-identical copies of one safety
notice — precisely the retrieval pollution the rule exists to prevent, arriving
through a different door.

It also dilutes the genuine content stream: the FC7 diagnostic article is 3101
characters, a third of which is a notice shared with every other article, sitting
inside the text that gets embedded.

## Decision

**Implement `/page/{N}` pagination.** The plan's termination rule is kept and is
empirically correct — the portal returns an empty listing past the last page. A
regression test asserts the request path and that no `?page=` URL is ever
issued.

**Strip the boilerplate before classifying.** `strip_boilerplate()` removes each
configured marker span from the body text, and the 200-character rule then
applies to what remains. The markers live in `config/config.yaml` under
`articles.boilerplate_markers` as `start`/`end` pairs, so a change to Seeley's
notice is a config edit, not a code change. A marker that does not match is
skipped, so articles without the notice are untouched.

Measured effect:

| Article | Before | After | Classification |
|---|---:|---:|---|
| TQ Service Guide (stub) | 1087 | 60 | `is_stub`, `content_stream: pdf` |
| FC7 diagnostic (content) | 3101 | 2074 | `content_stream: diagnostic_article` |

Both land correctly, and the diagnostic article's embedded text is no longer a
third boilerplate.

Removing the notice is not a loss of safety information. The licensed-technician
framing belongs in the generation system prompt, where the plan already puts it
(section 8), and where it applies to every answer rather than to whichever chunk
happened to retrieve.

## Consequences

- Two documented deviations from the build plan. Both are corrections toward the
  portal's actual behaviour, and both are covered by tests that fail loudly if
  the code regresses to the planned-but-wrong version.
- `config/config.yaml` gains a boilerplate-marker list that must be revisited if
  Seeley changes the notice. Symptom to watch for: the stub-versus-content split
  in the crawl summary shifting sharply, which is why that split is printed at
  the end of every run.
- The stripper is deliberately conservative — an exact span between two literal
  markers, skipped when either is absent. It will not catch a reworded notice.
  That is the correct failure direction: it under-strips rather than eating real
  content.
- The plan's other claims that proved wrong are recorded in the recon note but
  did not require a decision: a sitemap does exist (the plan says none), and
  article pages do expose a modification date. Both are opportunities rather than
  corrections, and are listed there as recommended follow-ups.
