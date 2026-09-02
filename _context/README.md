# `_context/`

Everything that is **about** this project but is not **part** of the running
system lives here.

## The rule

Nothing in `src/`, `tests/`, `scripts/` or `config/` may import from `_context/`.
It is documentation, not code. There are no `.py` files here and there never
will be. If something in here needs to be executable, it belongs in `scripts/`;
if it needs to be read at runtime, it belongs in `config/`.

The reason is simple: planning artefacts, research notes and half-formed ideas
have a different lifecycle from source. Mixing them into the source tree makes
both harder to read and makes it impossible to tell, six weeks later, which
document described a decision that was actually taken.

## Layout

The numbered subfolders reflect lifecycle order — brief, then plan, then
decisions, then research, then eval:

| Folder | Contents |
|---|---|
| `00-brief/` | Locked decisions, scope and constraints. What we agreed to build. |
| `01-plan/` | `build-plan.md` — the end-to-end build plan, the specification everything else refers to. `production-readiness.md` — what turns the finished POC into a production system, written at handoff. |
| `02-decisions/` | ADRs — one file per decision. See below. |
| `03-research/` | Findings about the world: site structure, corpus shape, recon. |
| `04-eval/` | Evaluation material, including the SME question template. |
| `scratch/` | **Gitignored.** Agent working notes and throwaway analysis. |

`scratch/` is where half-formed ideas go. Nothing in it is ever committed, so
nothing in it can be relied on. If a note in `scratch/` turns out to matter,
promote it to `03-research/` or write it up as an ADR.

## Architecture decision records

Every non-obvious decision gets an ADR in `02-decisions/`, using the standard
template:

```markdown
# NNNN. Title

## Status
Accepted | Superseded by 000N | Proposed

## Context
What forced a decision. The constraints, not the solution.

## Decision
What we chose, stated plainly.

## Consequences
What this makes easy, what it makes hard, and what it rules out.
```

Number them sequentially from `0001`.

**Never edit a decided ADR.** If a decision changes, write a new ADR that
supersedes it and mark the old one `Status: Superseded by 000N`. The value of an
ADR is that it records what was believed at the time; editing it destroys
exactly the thing that makes it worth keeping.

An ADR is warranted when a future reader would otherwise ask "why on earth is it
done this way?" — a deviation from the plan, a constraint discovered late, a
trade-off taken deliberately. Routine choices do not need one.
