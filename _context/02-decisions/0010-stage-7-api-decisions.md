# 0010. Stage 7 API: filters that constrain, refusals that are honest

## Status

Accepted.

## Context

Stage 7 exposes the pipeline over REST. Build-plan §9 fixes the surface:

```
POST /ask       {query, product_hint?, top_k?, stream?}  -> answer + citations + query_id
POST /search    {query, filters}                          -> raw chunks (debug)
POST /feedback  {query_id, rating, comment}               -> ack
GET  /pages/{doc_id}/{page_index}.png
GET  /docs      -> corpus inventory
GET  /health
```

This is the integration seam to .NET later, so the shapes are a contract. The
decisions below are the four places implementation had to differ from, or be
stricter than, the sketch.

## Decision 1: the corpus inventory is at `/docs-inventory`

FastAPI serves its Swagger UI at `/docs`, and `/openapi.json` beside it. Taking
`/docs` for the corpus inventory would remove the API's own documentation from
an interface whose entire purpose is being integrated against — and a .NET
client is generated from exactly those two paths.

The plan's path is given up; the plan's capability is not.

## Decision 2: filters constrain the search, they do not trim its output

The obvious implementation — retrieve top-k, then drop rows that fail the
filter — is wrong, and measurably so. The cascade truncates to `top_k` *before*
anything is filtered, so an explicit filter returns nothing whenever the
matching rows sat below the cut.

Measured on the live 16,189-row index during implementation:

| Request | Post-filtered | Pre-filtered |
|---|---|---|
| "fault code" + `product_family=VRF` | **0 hits** | 3 hits |
| "gas pressure" + `product_family=EVAP` | 0 hits | 3 hits |

A filter that works only when it was not needed is worse than no filter,
because the caller reads an empty list as "nothing exists".

So `LanceDBStore.search_dense` and `search_bm25` gained an optional `where`
predicate, applied as a **pre**-filter inside both channels, and `retrieve()`
threads it through. An intermediate fix — widening the candidate pool and then
filtering — is recorded here as rejected: it reduces the failure rate without
removing the failure, which is the worse kind of fix.

⚠ **This is a hard filter, and only a caller-supplied one ever becomes one.**
An *inferred* product family is still soft-boosted, exactly as build-plan §7.1
requires: a wrong guess must cost rank, never results. The distinction is the
whole point — `product_hint` on `/ask` boosts; `product_family` on `/search`
filters, because the caller typed it.

### The predicate is built, not escaped

Filter values are constrained to `^[A-Za-z0-9_]{1,64}$` and rejected with 400
otherwise. These fields hold lexicon identifiers (`DGH`, `service_guide`), so
anything else is a caller error rather than a value to quote and hope about. The
string is concatenated into a query; validating the shape is the honest way to
do that safely.

## Decision 3: `stream` is refused, not ignored

`stream: true` returns **501 Not Implemented**.

Accepting the field and returning a whole response is the tempting alternative,
and it is dishonest: the caller sees a 200 and concludes streaming works. The
field stays in the schema so the contract is stable when streaming is built;
until then the API says plainly that it is not there.

## Decision 4: `/pages` is written as though it were hostile

It is the one endpoint where a caller's input reaches the filesystem, so
`page_image_path` is deliberately boring, with four independent checks:

1. `doc_id` must match a SHA-256 digest or `article:{digits}` — nothing else can
   name a document in this corpus.
2. `page_index` must be a non-negative integer, enforced by the route's type.
3. The filename is **rebuilt** from the index (`{index:04d}.png`), never taken
   from input.
4. The resolved path is confirmed to sit inside the image root.

Any of the four alone would stop traversal. Together they mean a change to one
does not quietly remove the protection. Malformed id → 400; well-formed but
absent → 404, because those are different problems for the caller.

## Decision 5: the store opens at startup

A `lifespan` hook opens the LanceDB table before the first request. It costs
~4.8s against 16,189 rows while searches are 30–80ms (ADR 0007), so a lazily
opened store makes the first call after every deploy look broken.

A missing index does **not** fail the boot. `/health` must come up to say what
is wrong, and it reports each dependency separately — index presence, row count,
fault-code count, provider, whether that provider has a key, and the active
rerank backend. An empty index and a missing API key are different outages with
different fixes, and one boolean would hide that. The `rerank_backend` field in
particular travels into every `/search` hit, so a consumer can tell a reranked
list from an unreranked one rather than assuming (ADR 0007).

## Consequence: two stale-pin failures, exactly as ADR 0006 predicted

**`pydantic~=2.8.0`** conflicted with FastAPI's resolution. Relaxed to
`>=2.8,<3`, verified on 2.13. This is the third time a pin written against an
older environment has broken an install (tiktoken in Stage 3, lancedb and pandas
in Stage 4).

**`from __future__ import annotations` broke `/openapi.json`.** Every annotation
becomes a string, and FastAPI could not resolve `-> Response` when building the
schema. The failure is silent in normal use — every endpoint works — and fatal
for the one artefact a .NET client is generated from. The route is now annotated
`-> Any` with an explicit `response_class=FileResponse`.

It was a test asserting `/openapi.json` returns 200 that caught it, not any
manual check, which is the argument for asserting the contract rather than the
handlers.

## Alternatives considered

**Pushing `/ask`'s `product_hint` down as a hard filter too.** Rejected. §7.1 is
explicit that an inferred family must not filter, and a hint on `/ask` is
supplied by a UI that may itself have guessed. `/search` is a debugging tool
whose caller typed the value; `/ask` serves an installer who did not.

**Serving page images through a static mount.** Simpler, and it would hand out
the whole `data/01_interim` tree by path. The endpoint exists to expose exactly
one file shape.
