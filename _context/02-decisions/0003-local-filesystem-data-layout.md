# 0003. Numbered local-filesystem data stages

## Status

Accepted.

## Context

This is a two-to-three day POC with no infrastructure budget and no operations
capacity. Every hour spent standing up object storage, a data-versioning tool or
a database server is an hour not spent on parsing, which is where the project is
actually won or lost.

At the same time the pipeline has real requirements that a pile of loose files
would not meet. Citations must resolve to a specific page of a specific document
fetched at a specific time, so provenance has to survive every stage. The same
manual is attached to multiple articles, so the storage layer has to make
deduplication trivial rather than something each stage remembers to do. And the
crawl will be re-run many times during development, so re-running must be cheap
and must never destroy what the previous run fetched.

## Decision

Everything lives on the local filesystem under `data/`, in numbered stage
directories:

```
data/
├── 00_raw/         html/  pdf/  manifest.jsonl     <- immutable, content-addressed
├── 01_interim/     pages.jsonl  page_images/
├── 02_processed/   chunks.jsonl  codes.jsonl
├── 03_index/       vector store
├── cache/          llm/  embeddings/
└── reports/        triage and crawl summaries
```

Three rules make this work:

**1. `data/00_raw/` is write-once.** Nothing in the codebase may modify or
delete a file underneath it once created. Every later stage reads from it and
writes elsewhere. This is enforced structurally rather than by convention: no
write helper for `00_raw` is exported outside the acquire module, and
`clean_derived()` — the only deletion helper in the project — is incapable of
reaching it.

**2. PDFs are content-addressed** as `data/00_raw/pdf/{sha256}.pdf`. The mapping
from Freshdesk attachment ID to hash lives in the manifest, not in the filename.
Deduplication then falls out for free, and re-downloads are idempotent: if the
path exists the bytes are already correct, because the path *is* the hash.

**3. Every path in the project comes from `paths.py`.** No directory string
literals anywhere else. Constants derive from a single `DATA_ROOT` read from
settings, alongside an `ensure_dirs()` that `make init` calls.

`data/` is gitignored in its entirety.

## Consequences

- **No data-versioning tooling.** There is no DVC, no lakeFS, no bucket
  versioning. The mitigation is that `00_raw` is immutable and cheap to
  re-acquire, and everything downstream is reproducible from it.
- **Re-runs are cheap.** Cached HTML and content-addressed PDFs mean a second
  crawl is mostly disk reads. This is what makes iterating on the parser
  affordable.
- **Migrating to object storage later means changing `paths.py` and nothing
  else.** That is the whole point of routing every path through one module. The
  same applies to swapping the vector store, which sits behind a thin protocol
  in `index/store.py`.
- **It does not survive multiple machines.** The data tree is local, so two
  developers each do their own crawl. At 25 minutes that is acceptable; at ten
  times the corpus it would not be, and that is the trigger to revisit this.
- **Page images will be the bulk of the data.** ~10k pilot pages at 150 DPI is
  roughly 2 GB. Fine locally, object storage in production.
- **`make clean` is deliberately asymmetric.** It removes every derived stage
  and never `00_raw`, because re-deriving costs machine time we have while
  re-acquiring costs a polite crawl against someone else's server.
