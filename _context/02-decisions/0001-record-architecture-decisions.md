# 0001. Record architecture decisions

## Status

Accepted.

## Context

This project is a three-day POC that several people will pick up after the
initial build, including agent sessions with none of the original conversation
in context. Most of its design is driven by constraints that are not visible
from the code: no API key is obtainable, the knowledge is in PDFs rather than in
the HTML that appears to hold it, and a crawl that gets blocked ends the project
outright.

Left unrecorded, those constraints get rediscovered the expensive way. Someone
"fixes" the 1 req/sec rate limit, or adds a retry on HTTP 429, or indexes the
stub articles because their bodies looked like content. Each of those is a
reasonable-looking change that quietly breaks something load-bearing.

## Decision

We will record every non-obvious decision as an architecture decision record in
`_context/02-decisions/`, using the format described in `_context/README.md`:
Title, Status, Context, Decision, Consequences.

ADRs are numbered sequentially and are immutable once accepted. A decision that
changes is superseded by a new ADR, and the old one is marked
`Status: Superseded by 000N` rather than rewritten.

An ADR is warranted whenever a future reader would otherwise ask "why is it done
this way?" — deviations from the build plan, constraints discovered late,
trade-offs taken deliberately. Routine choices do not need one.

## Consequences

- The reasoning behind a constraint travels with the repository rather than
  living in one person's memory or a chat log.
- A superseded decision remains readable, so it is possible to see what was
  believed at the time and what changed. This costs a little clutter and is
  worth it.
- Writing an ADR is a small tax on every non-obvious decision. That tax is the
  mechanism: if a decision is not worth ten minutes of writing, it probably was
  not non-obvious.
