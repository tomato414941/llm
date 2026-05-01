# Reviewed Instruction Batch 013

Batch 013 is a 300-seed random-pool growth run. The goal was to move the
reviewed dataset past 1000 rows while keeping the same generator, filter, judge,
and promotion contract.

## Seeds

Added 300 seeds:

- `reasoning`: 50
- `coding`: 50
- `tool_use`: 50
- `knowledge_qa`: 50
- `summarization_transformation`: 50
- `instruction_following`: 50

## Pre-Generation Duplicate Report

```text
selected_seed_count,300
duplicate_seed_count,0
duplicate_row_count,0
```

## Generation

Generation used the default random generator pool; no fixed `--model` or
`--model-label` was passed.

- generated rows: 300
- raw rows: 298
- `generation_error` rows: 2
- finish_reason `stop`: 297
- finish_reason `length`: 1
- deterministic filter `needs_judge`: 296
- deterministic filter reject: 4
- recorded generation cost from provider usage metadata: `$0.178123529`

Generation errors:

- `lt_seed_1367`: `kimi-k2-6-openrouter`, `provider_content_error`
- `lt_seed_1395`: `kimi-k2-6-openrouter`, `provider_content_error`

Generator distribution:

- `qwen3-6-plus-openrouter`: 113
- `deepseek-v4-pro-openrouter`: 33
- `claude-sonnet-4-6-openrouter`: 33
- `glm-5-1-openrouter`: 33
- `gpt-5-4-openrouter`: 32
- `gpt-5-5-openrouter`: 31
- `kimi-k2-6-openrouter`: 25

## Judge Result

Judging used the default random judge pool with self-judge exclusion and the
one-retry transient failure policy.

- judged rows: 296
- self-judge rows: 0
- accept: 248
- needs_edit: 38
- reject: 8
- parse_error: 2
- retry attempts: 9
- retry recovered to accept: 7

Retry outcomes:

- `lt_seed_1247`: `provider_http_error` -> `accept`
- `lt_seed_1251`: `provider_content_error` -> `parse_error`
- `lt_seed_1252`: `judge_json_parse_error` -> `accept`
- `lt_seed_1283`: `judge_json_parse_error` -> `accept`
- `lt_seed_1358`: `provider_content_error` -> `accept`
- `lt_seed_1365`: `provider_content_error` -> `accept`
- `lt_seed_1435`: `judge_json_parse_error` -> `needs_edit`
- `lt_seed_1436`: `provider_content_error` -> `accept`
- `lt_seed_1444`: `judge_json_parse_error` -> `accept`

Judge distribution:

- `qwen3-6-plus-openrouter`: 64
- `kimi-k2-6-openrouter`: 51
- `claude-sonnet-4-6-openrouter`: 47
- `gpt-5-4-openrouter`: 43
- `deepseek-v4-pro-openrouter`: 38
- `glm-5-1-openrouter`: 29
- `gpt-5-5-openrouter`: 24

## Promotion

Promoted exact-prompt-unique accepted rows:

- accepted rows: 248
- promoted rows: 248
- skipped exact duplicate user prompts: 0
- reviewed dataset size after promotion: 1083 rows
- SFT export size after promotion: 1083 rows

## Current Capability Distribution

```csv
capability,seed_count,reviewed_count,eval_count,reviewed_target,reviewed_deficit
coding,267,191,5,50,0
instruction_following,239,204,5,80,0
knowledge_qa,183,153,3,30,0
reasoning,371,257,27,80,0
summarization_transformation,156,139,2,25,0
tool_use,254,139,6,35,0
```

## Operational Notes

The 1000-row reviewed dataset milestone is reached. Generator failure recording
continues to work: two provider content errors were preserved as
`generation_error` rows and excluded before judge.

The 296-row judge run was slow. Future data growth should use smaller chunks or
make judge output resumable before increasing batch size again.

The dominant non-accept pattern remains strict-format drift: code fences,
missing exact scenario mentions, missing short reasons, and answers with extra
labels. These remain task-contract issues, not global rejection rules.

## Artifacts

- `tracks/leverage/runs/instruction-outputs/readiness-batch-013-seed-duplicates.csv`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-013-raw.jsonl`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-013-filter.csv`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-013-filter-summary.csv`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-013-candidates.jsonl`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-013-judgments.jsonl`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-013-judgments.csv`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-013-judgments-summary.csv`
