# Reviewed Dataset Audit At 481 Rows

This audit checks whether the reviewed instruction dataset is ready to continue
toward 1,000 rows after batch 008.

## Snapshot

- reviewed rows: 481
- latest promoted batch: batch 008, 197 rows
- reviewed capability targets: all current targets met
- exact duplicate user prompts: rejected by validation
- SFT export: 481 rows

Current capability mix:

| capability | reviewed rows | share |
| --- | ---: | ---: |
| reasoning | 134 | 27.9% |
| instruction_following | 111 | 23.1% |
| coding | 87 | 18.1% |
| tool_use | 63 | 13.1% |
| knowledge_qa | 48 | 10.0% |
| summarization_transformation | 38 | 7.9% |

## Findings

The main issue is not exact duplication. The batch 008 duplicate preflight found
zero exact duplicates, and all 197 accepted rows were promotable.

The main issue is template repetition. The near-duplicate report shows that the
highest-similarity pairs are concentrated in `reasoning`, especially repeated
decision and deadline-calculation prompts that differ mostly by small numbers or
region identifiers.

Near-duplicate signal from the top 50 report:

- similarity >= 0.95: 8 pairs
- similarity >= 0.90: 50 pairs
- similarity >= 0.90 by capability: 46 `reasoning` pairs, 4 `tool_use` pairs
- dominant task shape among the top 50: `decision` / `decision`

This is acceptable for a small scale-up batch, but it is not a pattern to repeat
on the path to 1,000 rows.

Task-shape distribution is usable but uneven. Some combinations are naturally
small, but `reasoning` is currently concentrated in `decision`, `direct_answer`,
and `comparison`; `tool_use` is mostly `decision`; `knowledge_qa` is mostly
`direct_answer`.

Provenance is acceptable for this phase. Most rows now come from judged
candidates, with a small number of edited and historical rows. That matches the
project direction of scaling through generation and judgment rather than manual
curation.

## Decision

Do not remove batch 008 rows now. They are accepted, validated, non-exact
duplicates, and useful for reaching a larger training set.

Do not continue with batch 008-style templated seed generation. The next batch
should use more varied scenarios, domains, and task shapes before more API
spend.

## Next Batch Constraints

For the next 1,000-row push:

- keep exact-duplicate seed preflight mandatory
- avoid adding more prompts that differ only by numbers, IDs, regions, or dates
- cap any single prompt family to a small handful of rows
- prioritize underrepresented task shapes inside each capability
- especially diversify `reasoning` away from repeated release-risk and
  deadline-calculation templates
- keep generation and judging unchanged unless a measured failure appears

The next useful step is to prepare a smaller, more varied batch plan before
writing more seeds.

