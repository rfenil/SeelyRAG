# 0006. Stage 4 indexing: cache before client, upsert not append, and a tripwire that hung

## Status

Accepted.

## Context

Stage 4 embeds 16,189 chunks with `text-embedding-3-large` and builds the
LanceDB vector and full-text indexes. The build plan (§6) fixes the stack and
the schema. Four things it does not settle were decided here, three of them
forced by what the corpus and the toolchain actually did.

## Decision 1: the cache was written before the client, and is keyed on Stage 3's hash

`index/embed_cache.py` exists before `index/embedder.py` in both file order and
commit order. A cache added *after* the first full embedding run has already
failed at its job.

It is keyed by `sha256` of the final chunk text — which Stage 3 already computes
and stores as `Chunk.content_hash`, so the key is read off the chunk rather than
recomputed. A test asserts the two agree; if they ever diverge every chunk misses
on first lookup and the cache silently does nothing.

Storage is 256 shard files by two-character key prefix. One file per key is
pathological on NTFS at 16,000 entries; one file for everything is ~200 MB
rewritten after every batch, which is slower than the API call it replaces.
Writes are write-then-replace, because a shard truncated by a kill would read as
a silent miss for every key it holds.

The namespace is `{model}-{dimensions}`. Vectors of different width are not
interchangeable, so the 1024-d experiment ADR 0005 left open cannot read 3072-d
vectors, and switching back recovers the old cache instead of re-embedding.

**Measured:** first full build 906s, 63 API requests, $0.91. A no-op re-run is
**4.7s and zero API calls.**

## Decision 2: rows are upserted on `chunk_id`, and vanished rows are deleted

`merge_insert("chunk_id")`, not append. Appending would duplicate every
re-indexed chunk, and with deterministic ids from ADR 0005 the merge is exact.

Deletion matters as much: a chunk that disappears from `chunks.jsonl` — because
its page re-chunked into fewer pieces — must leave the index, or retrieval keeps
serving a row whose source text no longer exists.

Together with the cache this is the mechanism the user asked for when deferring
vision: a test simulates transcribing one page of three and asserts exactly one
text reaches the API and the row count does not grow.

## Decision 3: batches are packed by token budget, not by count

The plan says batch 256. That is a fine *count* limit and an insufficient one:
`text-embedding-3-large` also rejects a request whose combined input exceeds
roughly 300,000 tokens. A batch of 256 table chunks near their 6,000-token
ceiling (ADR 0005) would be 1.5M tokens and fail — on exactly the fault-code
content the system exists to serve.

Batching therefore respects both limits, with a 250k ceiling rather than 300k so
a disagreement between our tokeniser and theirs costs an extra request rather
than a failed one.

Two related guards: responses are sorted on the `index` field rather than
trusting arrival order, since a silently reordered batch attaches every vector to
the wrong chunk and nothing downstream could detect it; and texts are
deduplicated by key within a run as well as against the cache, because the same
boilerplate appears on hundreds of pages. The full build embedded 16,149 unique
texts for 16,189 chunks.

Retries distinguish transient from permanent: 429, timeouts and 5xx back off
exponentially; an authentication or bad-request failure raises immediately rather
than burning five attempts before failing anyway.

## Decision 4: dependency floors, not pins

The `downstream` extra pinned `lancedb~=0.13.0`, `pandas~=2.2.2` and friends —
all written before Python 3.13. Every one resolved to a release with no cp313
wheel, so pip fell back to building numpy from source and `pip install -e
".[downstream]"` failed outright with "Encountered error while generating package
metadata".

Replaced with lower bounds that carry wheels on 3.13. Installed: lancedb 0.37.1,
openai 3.5.0, pandas 3.0.5, numpy 2.5.2, pyarrow 25.0.1.

This is the same failure `tiktoken~=0.7.0` hit in Stage 3. The lesson is
recorded here because it will recur: **pins written against an older interpreter
are a build failure waiting for the next dependency to be added**, not a safety
measure.

Two consequences of the newer lancedb:

- `table_names()` is deprecated for a paginated `list_tables()` that returns a
  response object. Iterating that object yields *field* tuples, not names, so it
  is unwrapped explicitly.
- `to_lance()` requires the separate `pylance` package. Scans use LanceDB's own
  projected query builder instead, which also keeps 3,072-float vectors out of a
  scan that only needs hashes.
- `create_fts_index` is deprecated in favour of `create_index(config=FTS())` —
  but that form exists only on the *async* table API; the synchronous
  `LanceTable.create_index` has no `config` parameter. The warning is silenced at
  that one call site rather than project-wide, so a genuine deprecation elsewhere
  still fails the suite.

## Consequence 1: the no-network tripwire hung the suite instead of failing it

`tests/conftest.py` blocked network access by replacing `socket.socket` with a
function. That worked for three stages and broke the moment an async-backed
library arrived. Two independent failures:

- Python's `ssl` module does `class SSLSocket(socket)` at import time, and a
  function cannot be subclassed — so any library importing `ssl` after the
  fixture applied raised `TypeError: function() argument 'code' must be code`.
- Windows' `ProactorEventLoop` builds its self-pipe from a loopback socketpair
  and then calls `isinstance(conn, socket.socket)`. With the constructor
  replaced, that raises *inside the event loop*, and the loop hangs forever.

LanceDB is async-backed, so every store test hung rather than failing. A hanging
test is worse than a failing one: it looks like slowness, not breakage, and it
took a 10-minute timeout to notice.

The tripwire now guards `connect` and `connect_ex` and **allows loopback**. The
type stays intact, `ssl` and asyncio work, and everything off-machine still
raises with a message naming the address. Loopback is not the network the rule
exists to protect against — Seeley's production server and a paid API both are.

## Consequence 2: a full build holds every vector in memory

`build_index` embeds the whole corpus before writing any row, so 16,189 vectors
of 3,072 Python floats are live at once. The first build peaked at **~2.9 GB
RSS**. It completed, and incremental re-runs never approach it because they embed
almost nothing — but a full rebuild of a substantially larger corpus should
stream batches into the store rather than accumulate them.

Recorded rather than fixed: it is not on the path to Stage 5, and the corpus is
not currently growing. It should be fixed before the vision backfill roughly
doubles the indexable text.

## What the first real queries showed

Retrieval works, and it independently confirms that §7's cascade is necessary
rather than optional. Neither channel alone is trustworthy on product family:

| Query | Dense | BM25 |
|---|---|---|
| "Braemar evaporative cooler water pump not priming" | correct Braemar EVAP manuals | **Coolerado** — a different product line |
| "VRF outdoor unit E4 high discharge temperature" | drifts to RC | correct VRF fault table |
| "gas pressure test procedure manifold pressure" | DGH training + TQ Service Guide p.21 | **Aira** manuals, family `UNKNOWN` |
| "TQ heater fault code FC7 what do I check" | correct | correct |

BM25 wins on exact code strings and loses badly on natural-language symptom
queries, where it matches vocabulary shared across every HVAC manual. Dense is
the reverse. This is precisely the case for RRF fusion plus the product-family
soft boost — and for applying that boost *before* truncation, which build-plan
§7.2 rule 5 already warns about.

## Alternatives considered

**Hard-filtering on product family at query time.** Tempting given the drift
above, and still wrong for the reason §7.1 gives: a misclassified query returns
nothing and the system looks broken. Soft-boost, and hard-filter only when the
user names a model explicitly.

**Reducing to 1024 dimensions now.** Deferred to the eval, per ADR 0005. The
cache namespace makes the experiment cheap whenever it is wanted.
