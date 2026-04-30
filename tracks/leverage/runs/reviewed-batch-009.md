# Reviewed Instruction Batch 009

Batch 009 returned to the normal policy: both generation and judging use random
model pools. It also avoids the batch 008 pattern of prompts that differ only by
small numeric substitutions.

## Seeds

Added 100 seeds:

- `reasoning`: 30
- `coding`: 20
- `tool_use`: 15
- `knowledge_qa`: 15
- `summarization_transformation`: 10
- `instruction_following`: 10

The batch intentionally varies domain, task shape, and output format.

## Pre-Generation Duplicate Report

```text
selected_seed_count,100
duplicate_seed_count,0
duplicate_row_count,0
```

This checks the full generation user prompt against the selected batch and the
existing reviewed instruction dataset.

## Generation

Generation used the default random generator pool from
`tracks/leverage/prompts/README.md`; no fixed `--model` or `--model-label` was
passed.

- generated rows: 100
- finish_reason `stop`: 100
- deterministic filter `needs_judge`: 99
- deterministic filter reject: 1
- recorded generation cost from provider usage metadata: `$0.089834492`

Generator distribution:

- `qwen3-6-plus-openrouter`: 25
- `deepseek-v4-pro-openrouter`: 16
- `claude-sonnet-4-6-openrouter`: 15
- `glm-5-1-openrouter`: 14
- `gpt-5-4-openrouter`: 12
- `gpt-5-5-openrouter`: 9
- `kimi-k2-6-openrouter`: 9

## Judge Result

Judging used the default random judge pool; no fixed `--judge-model` or
`--judge-label` was passed. Self-judge exclusions were applied.

- judged rows: 99
- self-judge rows: 0
- accept: 92
- needs_edit: 3
- reject: 1
- parse_error: 3
- average correctness: 1.929
- average instruction following: 1.919
- average conciseness: 1.879
- average safety: 1.939

Judge distribution:

- `claude-sonnet-4-6-openrouter`: 28
- `kimi-k2-6-openrouter`: 17
- `gpt-5-5-openrouter`: 16
- `deepseek-v4-pro-openrouter`: 15
- `gpt-5-4-openrouter`: 15
- `glm-5-1-openrouter`: 4
- `qwen3-6-plus-openrouter`: 4

Not accepted:

- `lt_seed_781`: parse error from truncated or malformed judge JSON.
- `lt_seed_789`: parse error because the API response content was not text.
- `lt_seed_822`: parse error from OpenRouter HTTP 503.
- `lt_seed_782`: `needs_edit`; answer was more verbose than requested.
- `lt_seed_811`: `needs_edit`; answer exceeded the word limit.
- `lt_seed_855`: `needs_edit`; answer did not satisfy the address detail constraint.
- `lt_seed_867`: `reject`; exact text constraint was violated by extra leading space.

## Promotion

Promoted all exact-prompt-unique accepted rows:

- accepted rows: 92
- promoted rows: 92
- skipped exact duplicate user prompts: 0
- reviewed dataset size after promotion: 573 rows

Reviewed capability distribution after promotion:

```text
capability,reviewed_count,reviewed_target,reviewed_deficit
coding,106,50,0
instruction_following,119,80,0
knowledge_qa,63,30,0
reasoning,161,80,0
summarization_transformation,47,25,0
tool_use,77,35,0
```

## Artifacts

- `tracks/leverage/runs/instruction-outputs/readiness-batch-009-seed-duplicates.csv`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-009-raw.jsonl`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-009-filter.csv`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-009-filter-summary.csv`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-009-candidates.jsonl`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-009-judgments.jsonl`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-009-judgments.csv`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-009-judgments-summary.csv`
