# Next Reviewed Batch Plan

Date: 2026-04-29

## Goal

Grow reviewed instruction data from 29 rows to about 60 rows before the next
capability-seeking LoRA run.

This is an intermediate data-growth target, not the final 300-row target.

## Current Reviewed Distribution

| capability | reviewed rows |
| --- | ---: |
| coding | 4 |
| instruction_following | 4 |
| knowledge_qa | 2 |
| reasoning | 11 |
| summarization_transformation | 2 |
| tool_use | 6 |

## Next Increment

Target new accepted rows: about 31

| capability | new reviewed rows |
| --- | ---: |
| instruction_following | 10 |
| knowledge_qa | 8 |
| summarization_transformation | 8 |
| coding | 8 |
| reasoning | 4 |
| tool_use | 3 |

## Rules

- Prefer underrepresented capabilities first.
- Keep `reasoning` and `tool_use` growth slower until the other capabilities
  catch up.
- For `coding`, write seeds that demand short answers and explicitly forbid
  broad refactors or extra recommendations.
- Promote only `judge_accepted_candidate` or edited rows with structured
  provenance.
- Leave unclear historical rows as `historical_reviewed`; do not invent
  provenance.
