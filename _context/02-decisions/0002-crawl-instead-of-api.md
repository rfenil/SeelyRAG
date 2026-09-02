# 0002. Crawl the public portal instead of using the API

## Status

Accepted.

## Context

Seeley International's help centre runs on Freshdesk, which has a perfectly good
REST API at `/api/v2/*`. That API returns **401**, and no key is obtainable —
not "not yet requested", but unavailable to us at the time of building.

That leaves the public portal, which is served as ordinary HTML at
`/support/solutions`. There is no third option. Acquisition is either a polite
public crawl or it does not happen.

Two consequences follow immediately and neither is a detail:

1. **`robots.txt` becomes a project gate rather than a courtesy.** If it
   disallowed the solution paths, the acquisition stage would be dead and no
   amount of engineering would fix it. It had to be checked before anything was
   built on the assumption that a crawl was possible.
2. **Being blocked has no recovery path.** With an API key, a 429 is an
   inconvenience. Without one, a block on our User-Agent or IP removes the only
   channel we have.

## Decision

Acquire by crawling the public portal, behind a `FreshdeskSource` abstract base
class with `PortalScraper` as its only implementation.

Keep the ABC even though there is exactly one implementation. It costs about
thirty minutes, and if a key ever appears, an `ApiClient` implementing
`list_folders`, `list_articles` and `get_article` is the entire change —
`base.py` documents what such an implementation must guarantee so that nothing
downstream needs to move.

Crawl etiquette is treated as a correctness requirement, enforced in code rather
than left to discipline:

- 1 request per second, single-threaded. Concurrency buys nothing at this
  corpus size and risks the one thing we cannot recover from.
- Every fetch cached to disk, keyed by URL. Without it, each development
  iteration is 25 minutes and another ~1,500 requests against someone else's
  production server.
- An honest `User-Agent` carrying a contact address.
- **Stop immediately on HTTP 429 or 403.** Never retry into a block. Retries
  apply to 5xx and timeouts only.

`scripts/00_check_robots.py` runs the gate, and `scripts/02_acquire.py` calls it
before its first fetch and refuses to proceed on a blocked *or undetermined*
verdict. An ambiguous answer — a 500, a connection failure — is never read as
permission.

**Verdict, checked 2026-08-20:** the portal's `robots.txt` permits the crawl.
`/support/solutions` is unrestricted, and `Allow: /helpdesk/attachments`
precedes `Disallow: /helpdesk/`, so the manual PDFs are explicitly fetchable. No
`Crawl-delay` is declared. See `_context/03-research/portal-recon.md`.

## Consequences

- **Crawl politeness is project-critical, not a nicety.** Anyone tempted to
  raise `crawl.rps`, add concurrency, or retry a 429 should read this ADR first.
  The configuration validates `rps` at load time so an unreasonable value fails
  immediately rather than at request 400.
- **The `robots.txt` verdict is not permanent.** Seeley can change it. The gate
  runs on every acquisition run, not once.
- **Change detection is re-crawl-and-hash.** The build plan asserts that no API
  key means no `updated_at` and therefore no incremental sync. That turns out to
  be only half true: the article pages carry a "Modified on" line, and the
  sitemap carries per-article `lastmod` values (see ADR 0004). We now record
  `updated_at`, but it remains a scraped string rather than an API field, so the
  safe assumption for correctness is still full re-crawl and content hashing. At
  ~900 articles that is a 25-minute job, which is fine — but it rules out
  near-real-time freshness, and stakeholders should be told that rather than
  left to assume otherwise.
- **A single dead link must not cost the crawl.** Attachment failures are
  recorded per-article and surfaced in the run summary; only a block aborts the
  run.
- **Content rights remain unresolved and are not a technical problem.** The
  manuals are Seeley's intellectual property, and public availability is not a
  licence to ingest, store and redistribute them through a derived product.
  Crawling without an explicit relationship makes this more pressing, not less.
  This must be settled in writing before the system reaches a real installer.
