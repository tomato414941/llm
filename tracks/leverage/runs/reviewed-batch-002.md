# Reviewed Batch 002

Date: 2026-04-29

## Purpose

Move the reviewed dataset toward the 300-row `Qwen/Qwen3.5-9B` readiness gate
without increasing project-agent-specific bias.

## Inputs

- Added seed range: `lt_seed_151` through `lt_seed_190`
- Raw generated rows: 40
- Generator selection: weighted random, one answer per seed
  - `qwen3-6-plus-openrouter`: 0.50
  - `claude-sonnet-4-6-openrouter`: 0.25
  - `gpt-5-4-openrouter`: 0.25
- Generator random seed: `20260429`
- Judge selection: weighted random non-self judge, one judge per row
  - `qwen3-6-plus-openrouter`: 0.50
  - `claude-sonnet-4-6-openrouter`: 0.25
  - `gpt-5-4-openrouter`: 0.25
- Judge random seed: `20260430`

## Results

- Structural filter: 40/40 sent to judge
- Judge decisions:
  - accept: 21
  - needs_edit: 13
  - reject: 6
- Promoted reviewed rows: 16
- Reviewed dataset size after promotion: 99 rows

Four accepted rows were not promoted because acceptance alone is not sufficient:
one was an obvious near-duplicate, and three were too verbose or weakly aligned
with the requested concise training style. `needs_edit` rows were left as run
artifacts rather than manually repaired.

## Artifacts

- `tracks/leverage/runs/instruction-outputs/readiness-batch-002-raw.jsonl`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-002-filter.csv`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-002-filter-summary.csv`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-002-candidates.jsonl`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-002-judgments.jsonl`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-002-judgments.csv`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-002-judgments-summary.csv`

## Next Decision

The batch improved general-purpose instruction-following and coding coverage,
but reasoning and tool-use remain under target. The next batch should bias seed
selection toward general reasoning and tool-use tasks while keeping the same
accept-first promotion rule.
