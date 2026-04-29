# Reviewed Batch 003

Date: 2026-04-29

## Purpose

Add more general reasoning and tool-use coverage while testing the expanded
teacher-model pool.

## Inputs

- Added seed range: `lt_seed_191` through `lt_seed_230`
- Raw generated rows: 40
- Target capability mix:
  - reasoning: 20
  - tool_use: 12
  - coding: 4
  - instruction_following: 4

## Generation Pool

Generation used the default model pool documented in
`tracks/leverage/prompts/README.md`:

- `qwen3-6-plus-openrouter`: 0.30
- `gpt-5-4-openrouter`: 0.10
- `claude-sonnet-4-6-openrouter`: 0.10
- `gpt-5-5-openrouter`: 0.10
- `kimi-k2-6-openrouter`: 0.10
- `gemini-3-1-pro-preview-openrouter`: 0.10
- `deepseek-v4-pro-openrouter`: 0.10
- `glm-5-1-openrouter`: 0.10

Generation initially failed after 16 rows because one endpoint required
reasoning to be enabled. The run was resumed with provider-default reasoning
settings and completed all 40 rows.

## Judge Result

Judging exposed operational problems in the expanded pool:

- Some judge calls returned no text content or malformed JSON.
- Some judge calls were too slow for practical 40-row batch operation.
- The final usable judge set for completing the batch was reduced to
  `qwen3-6-plus-openrouter` and `gpt-5-4-openrouter`.

Only 30 of the 40 generated rows were judged in this batch. The remaining 10
rows stay as raw artifacts and were not promoted.

Final judged result:

- judged rows: 30
- accept: 16
- needs_edit: 13
- reject: 1
- promoted reviewed rows: 5
- reviewed dataset size after promotion: 104 rows

Most accepted rows were still too verbose for reviewed training data. They were
not manually repaired. This keeps the batch aligned with the accept-first,
low-human-editing policy.

## Artifacts

- `tracks/leverage/runs/instruction-outputs/readiness-batch-003-raw.jsonl`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-003-filter.csv`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-003-filter-summary.csv`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-003-candidates.jsonl`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-003-judgments.jsonl`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-003-judgments.csv`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-003-judgments-summary.csv`

## Next Decision

Do not use the expanded pool blindly for judging. Keep the broad pool for
generation, but either restrict judging to models with stable JSON responses or
add stricter per-call timeout/checkpoint behavior before the next large judged
batch.
