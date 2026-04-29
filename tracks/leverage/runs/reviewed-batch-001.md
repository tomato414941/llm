# Reviewed Batch 001

Date: 2026-04-29

## Goal

Move toward the 300-row `Qwen/Qwen3.5-9B` readiness gate by adding reviewed
instruction rows for underrepresented capabilities.

## Inputs

- Added seed prompts: `lt_seed_111` through `lt_seed_150`
- Targeted capabilities:
  - reasoning: 12 seeds
  - instruction_following: 12 seeds
  - coding: 8 seeds
  - tool_use: 8 seeds

## Generation And Review

- Raw generated rows: 40
- Structural filter result: 40/40 `needs_judge`
- Non-self judge result:
  - accept: 16
  - needs_edit: 20
  - reject: 3
  - parse_error: 1

Ignored local artifacts:

- `tracks/leverage/runs/instruction-outputs/readiness-batch-001-raw.jsonl`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-001-candidates.jsonl`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-001-judgments.jsonl`
- matching CSV summaries under `tracks/leverage/runs/instruction-outputs/`

## Promotion

Promoted only non-self judge `accept` rows into the reviewed dataset:

- Added reviewed rows: 16
- New reviewed range: `instr_0060` through `instr_0075`
- Reviewed dataset size: 75 rows
- SFT export regenerated locally: 75 rows

## Next Decision

The batch improved the dataset count but also showed that many accepted or
near-accepted generated answers are too verbose. The next generation batch
should tighten seed constraints toward short, directly promotable answers before
scaling volume.
