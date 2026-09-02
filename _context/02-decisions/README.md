# Architecture decision records

One file per decision, numbered sequentially. See `../README.md` for the rule
about never editing a decided ADR.

| # | Decision | Status |
|---|---|---|
| [0001](0001-record-architecture-decisions.md) | Record architecture decisions | Accepted |
| [0002](0002-crawl-instead-of-api.md) | Crawl the public portal instead of using the API | Accepted |
| [0003](0003-local-filesystem-data-layout.md) | Numbered local-filesystem data stages | Accepted |
| [0004](0004-portal-reality-corrections.md) | Correct the build plan where the live portal disagrees | Accepted |
| [0005](0005-stage-3-chunking-decisions.md) | Stage 3 chunking: incremental ids, measured thresholds, precision-first codes | Accepted |
| [0006](0006-stage-4-indexing-decisions.md) | Stage 4 indexing: cache before client, upsert not append, dependency floors | Accepted |
| [0007](0007-stage-5-retrieval-decisions.md) | Stage 5 retrieval: deterministic query understanding, and an honest reranker | Accepted |
| [0008](0008-openai-as-the-llm-provider.md) | OpenAI as the default LLM provider, behind a provider-agnostic layer | Accepted |
| [0009](0009-stage-6-generation-decisions.md) | Stage 6 generation: verify the prompt's rules rather than trust them | Accepted |
| [0010](0010-stage-7-api-decisions.md) | Stage 7 API: filters that constrain, refusals that are honest | Accepted |

## Template

```markdown
# NNNN. Title

## Status
Accepted

## Context
What forced a decision.

## Decision
What we chose.

## Consequences
What this makes easy, what it makes hard, what it rules out.
```
