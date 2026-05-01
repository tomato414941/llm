# Reviewed Instruction Batch 011

Batch 011 is a live random-pool data growth run after generator failures started
being recorded as `generation_error` rows instead of stopping the batch.

## Seeds

Added 50 seeds:

- `reasoning`: 9
- `coding`: 9
- `tool_use`: 8
- `knowledge_qa`: 8
- `summarization_transformation`: 8
- `instruction_following`: 8

## Pre-Generation Duplicate Report

```text
selected_seed_count,50
duplicate_seed_count,0
duplicate_row_count,0
```

## Generation

Generation used the default random generator pool; no fixed `--model` or
`--model-label` was passed.

- generated rows: 50
- raw rows: 50
- `generation_error` rows: 0
- finish_reason `stop`: 50
- deterministic filter `needs_judge`: 50
- deterministic filter reject: 0
- recorded generation cost from provider usage metadata: `$0.02742752`

Generator distribution:

- `qwen3-6-plus-openrouter`: 14
- `deepseek-v4-pro-openrouter`: 8
- `gpt-5-4-openrouter`: 7
- `glm-5-1-openrouter`: 7
- `kimi-k2-6-openrouter`: 5
- `claude-sonnet-4-6-openrouter`: 5
- `gpt-5-5-openrouter`: 4

The new `generation_error` path was not triggered in this batch because all
provider calls returned text successfully.

## Judge Result

Judging used the default random judge pool with self-judge exclusion and the
one-retry transient failure policy.

- judged rows: 50
- self-judge rows: 0
- accept: 45
- needs_edit: 5
- reject: 0
- parse_error: 0
- retry attempts: 2
- retry recovered to accept: 2

Retry recoveries:

- `lt_seed_921`: `provider_content_error` -> `accept`
- `lt_seed_931`: `judge_json_parse_error` -> `accept`

Needs edit:

- `lt_seed_922`: code answer wrapped in Markdown fences
- `lt_seed_934`: code answer wrapped in Markdown fences
- `lt_seed_946`: SQL answer wrapped in Markdown fences
- `lt_seed_958`: Rust answer wrapped in Markdown fences
- `lt_seed_959`: command answer missed the requested explanatory sentence

Judge distribution:

- `claude-sonnet-4-6-openrouter`: 16
- `kimi-k2-6-openrouter`: 9
- `deepseek-v4-pro-openrouter`: 8
- `gpt-5-5-openrouter`: 7
- `gpt-5-4-openrouter`: 6
- `qwen3-6-plus-openrouter`: 3
- `glm-5-1-openrouter`: 1

## Promotion

Promoted exact-prompt-unique accepted rows:

- accepted rows: 45
- promoted rows: 44
- skipped exact duplicate user prompts: 1
- skipped duplicate seed: `lt_seed_942`
- reviewed dataset size after promotion: 668 rows

The duplicate skip was intentional: the accepted answer was valid, but its user
prompt exactly matched an existing reviewed row.

## Current Capability Distribution

```csv
capability,seed_count,reviewed_count,eval_count,reviewed_target,reviewed_deficit
coding,183,119,5,50,0
instruction_following,156,135,5,80,0
knowledge_qa,100,78,3,30,0
reasoning,287,182,27,80,0
summarization_transformation,73,62,2,25,0
tool_use,171,92,6,35,0
```

## Artifacts

- `tracks/leverage/runs/instruction-outputs/readiness-batch-011-seed-duplicates.csv`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-011-raw.jsonl`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-011-filter.csv`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-011-filter-summary.csv`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-011-candidates.jsonl`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-011-judgments.jsonl`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-011-judgments.csv`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-011-judgments-summary.csv`
