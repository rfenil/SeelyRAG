# 0005. Stage 3 chunking: incremental ids, measured thresholds, precision-first codes

## Status

Accepted.

## Context

Stage 3 turns 13,156 parsed pages into indexable chunks and a fault-code lookup
table. The build plan (§5) fixes the rules — page anchoring, ~800/1,200/120
token sizing, atomic capped tables, multi-page table merge, breadcrumb prefix —
but leaves four things open that only the real corpus could settle. Each was
decided by measuring, and the measurements are recorded here because the numbers
are not re-derivable from the code alone.

## Decision 1: chunk ids are deterministic and every chunk carries a content hash

`chunk_id` is `{doc_id}:p{page_index}:c{ordinal}`, never a counter or a UUID,
and `content_hash` is `sha256` of the final chunk text with its breadcrumb
prefix included.

The user asked that vision — 3,459 pages, 26.3% of the corpus, deferred for now
— be addable later "with minimal change". This is that mechanism. Re-chunking
after a vision pass produces byte-identical ids and hashes for every page vision
did not touch, so `scripts/04_index.py` reports the corpus as unchanged /
changed / new / gone, and Stage 4 embeds only `changed + new`.

Verified: a clean re-run over the full corpus reports 16,189 unchanged, 0
changed. An earlier run, made before a fix to the overlap logic, correctly
reported 189 changed.

The hash covers the breadcrumb because that prefix is part of what gets
embedded. Hashing the body alone would let a re-titled document keep a stale
vector.

## Decision 2: embeddings stay at the model's native 3,072 dimensions

`text-embedding-3-large` supports truncating to fewer dimensions at the same
price, which would shrink the index roughly threefold. The user asked for the
smaller index *and* for no loss of accuracy; those conflict, and accuracy was
stated as the binding constraint, so the native width wins.

`config/config.yaml` `index.embedding_dim` makes it a config change rather than
a code change, and the embedding cache will make the A/B nearly free once it
exists. The trade should be measured by the eval, not assumed.

Cost is not a factor either way: the full corpus is 6.99M tokens, **$0.91** at
$0.13/M. Chunk quality is worth engineering time; embedding spend is not.

## Decision 3: the multi-page table merge tolerates 24pt of column drift

The plan says to merge when "consecutive pages have matching column geometry and
the later has no header row" without saying how close "matching" is.

Measured over the corpus. Of 2,956 detected tables, only 724 lack a header and
only 81 adjacent page pairs have a header-less table following another. Of those,
36 have matching column counts, and their left/right edge deltas fall into two
clearly separated groups:

| Edge delta | Pairs | What they are |
|---|---|---|
| 0–11pt | 24 | Detector jitter on what is visibly one table |
| 12.2–12.3pt | 2 | A numbered procedure table continuing across a page |
| 70.9pt and above | 10 | Genuinely different tables — VRF capacity tables with different layouts |

An initial 12pt tolerance sat exactly on the boundary and rejected the two real
continuations. 24pt sits in the empty gap between 12.3 and 70.9: it captures
every real continuation and comes nowhere near the first false one.

Result: 19 merged table chunks, including the TQ fault-code table that chains
across pages 43–46 — the exact case §5.1 rule 4 was written for.

A merge with missing `bbox` data is **refused**, not guessed. Column count alone
is too weak a signal, and a wrong merge cites the wrong page for half its rows,
which is worse than the shredding the merge exists to prevent.

## Decision 4: a merged table shows a page range only when the labels support it

38.7% of page labels are guesses (`label_source == "index"`). Merging over them
produced pairs like `p9 → p8` and `p3 → p9`. A citation reading "pages 9-8" is
worse than one naming only the anchor page, so `page_range` is emitted only when
both labels are numeric and ascending.

Because that makes `page_range` an unreliable count of merges, `Chunk` also
carries `page_span` — always accurate. The eval needs it too: a chunk spanning
three pages cannot fairly be scored against a single expected page.

## Decision 5: the fault-code table is precision-first

The lexicon's four patterns are far too loose to use raw. A first pass over the
corpus produced 294 rows, of which a large minority were junk. Four filters were
added, each for a named failure:

1. **`fault\s+code\s+(\w+)` captures whatever word follows the phrase.** The
   first pass yielded codes named `access`, `chart`, `column`, `definition`,
   `displayed`, `does`, `history` and `Braemar`. A candidate must now match
   `^[A-Za-z]{0,2}[\s:.\-]?\d{1,2}$` — a digit is required.
2. **`[EFH][\s:.-]?\d{1,2}` matches "F 12" in a dimensions table** and "H 10" in
   a part number. A candidate is kept only when fault vocabulary appears within
   160 characters.
3. **Wide code tables are laid out `code | meaning | code | meaning`.** Joining
   every other cell welded four codes' meanings into each one, so a code now
   takes the cell *beside* it.
4. **Contents pages match the phrase patterns**, yielding dotted leaders
   ("FAULT CODE 08 EXAMPLE......") as the meaning. Empty and leader meanings are
   dropped.

Result: **136 rows, all with usable meanings.**

The governing principle: a missed code still reaches the installer through
ordinary hybrid retrieval, while a wrong one is *pinned ahead* of it. That is
also why pure-letter codes such as `PL` and `FP` are deliberately not admitted —
at this level they are indistinguishable from ordinary words.

Bare numbers captured from "Fault Code 8" normalise to `FC08`, because the DGH
manuals print "Fault Code 08" while the article titles and every installer say
"FC8". Left unnormalised, a query for one silently misses the other.

Codes are keyed by `(code_key, product_family)`, never by code alone: `E:04` on
a gas heater is not `E:04` on a VRF unit, and collapsing them would answer a DGH
question from a VRF manual — build-plan §13, risk 3.

## Consequence: a Stage 2 defect is now visible and quantified

Filtering code meanings surfaced text corruption that is **not** a chunking
problem. Two kinds:

- **Interleaved columns.** "Full wFautlel rw partoetre pcrtiootnection" is "Full
  water protection" woven into its neighbouring column, character by character.
- **Broken ToUnicode CMaps.** A few PDFs decode to a shifted alphabet:
  "(QVXUHWKHPRWRUSRZHUFDEOH" is "Ensure the motor power cable".

`chunk/codes.py::looks_corrupt` detects both — mid-word capitals at ≥1.5 per 100
characters, or a run of five or more consonants. Over the extracted code
meanings every genuinely corrupt string scores above 1.5 and the worst
legitimate one scores 0.18, an order of magnitude apart.

Corpus-wide the corruption reaches **14.2% of chunks and 20.5% of tokens.**
Rejecting it at the occurrence level rather than the row level means a code
printed corruptly in one manual and cleanly in another keeps the clean sighting:
RC/E09's meaning went from garbage to "Indoor unit full water error" this way.

The code table is protected. **The embedded chunk text is not**, and this will
cap retrieval quality regardless of how good the embeddings are. Fixing it needs
a different extraction path for the affected documents — a Stage 2 change, not
yet scoped, and worth deciding before the eval is used to judge retrieval.

## Alternatives considered

**Contextual retrieval (build-plan §5.2).** Still correctly deferred. It needs a
whole-document pass no stage produces, and it must run *before* embedding since
the prefix has to be in the embedded text. Ship hybrid + rerank first.

**LLM-extracted code meanings.** §5.3 suggests Haiku pulls `meaning` from the
surrounding table row. Deterministic adjacent-cell extraction turned out to be
sufficient and has a property an LLM does not: the meaning is verbatim source
text. A reworded fault description is a wrong answer with a citation attached.
The LLM pass remains available for the residual rows whose meanings carry
cell-boundary noise.
