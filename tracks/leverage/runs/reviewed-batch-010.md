# Reviewed Instruction Batch 010

Batch 010 is a small live check of the random generator and judge pools after
adding one retry for transient judge execution failures.

## Seeds

Added 50 seeds:

- `reasoning`: 10
- `coding`: 8
- `tool_use`: 7
- `knowledge_qa`: 8
- `summarization_transformation`: 7
- `instruction_following`: 10

## Pre-Generation Duplicate Report

```text
selected_seed_count,50
duplicate_seed_count,0
duplicate_row_count,0
```

## Generation

Generation used the default random generator pool; no fixed `--model` or
`--model-label` was passed.

The first generation attempt stopped after 18 saved rows because one provider
response had non-text message content. The run was resumed with `--resume`, so
completed rows were preserved and only the remaining seeds were generated.

- generated rows: 50
- finish_reason `stop`: 50
- deterministic filter `needs_judge`: 49
- deterministic filter reject: 1
- recorded generation cost from provider usage metadata: `$0.038228695`

Generator distribution:

- `qwen3-6-plus-openrouter`: 10
- `gpt-5-4-openrouter`: 9
- `glm-5-1-openrouter`: 9
- `kimi-k2-6-openrouter`: 7
- `claude-sonnet-4-6-openrouter`: 6
- `deepseek-v4-pro-openrouter`: 5
- `gpt-5-5-openrouter`: 4

## Judge Result

Judging used the default random judge pool with self-judge exclusion and the
one-retry transient failure policy.

- judged rows: 49
- self-judge rows: 0
- accept: 48
- needs_edit: 0
- reject: 0
- parse_error: 1
- final error type: `judge_schema_error`: 1
- retry attempts: 3
- retry recovered to accept: 3

Retry recoveries:

- `lt_seed_871`: `provider_content_error` -> `accept`
- `lt_seed_887`: `judge_json_parse_error` -> `accept`
- `lt_seed_917`: `provider_content_error` -> `accept`

Not accepted:

- `lt_seed_914`: `judge_schema_error`; judge returned an invalid score type for
  `conciseness`. This is intentionally not retried by the current policy.

Judge distribution:

- `qwen3-6-plus-openrouter`: 12
- `deepseek-v4-pro-openrouter`: 9
- `claude-sonnet-4-6-openrouter`: 8
- `gpt-5-5-openrouter`: 6
- `gpt-5-4-openrouter`: 5
- `kimi-k2-6-openrouter`: 5
- `glm-5-1-openrouter`: 4

## Promotion

Promoted all exact-prompt-unique accepted rows:

- accepted rows: 48
- promoted rows: 48
- skipped exact duplicate user prompts: 0
- reviewed dataset size after promotion: 624 rows

## Artifacts

- `tracks/leverage/runs/instruction-outputs/readiness-batch-010-seed-duplicates.csv`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-010-raw.jsonl`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-010-filter.csv`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-010-filter-summary.csv`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-010-candidates.jsonl`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-010-judgments.jsonl`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-010-judgments.csv`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-010-judgments-summary.csv`
