# Surface Constraint Batch 001

Date: 2026-05-06

## Goal

Add reviewed instruction data that exercises longer answers and exact surface constraints. The immediate gap was that the reviewed dataset was dominated by short answers, while benchmark failures included long-form and format-following behavior.

## Inputs

- Seed range: `lt_seed_1491` through `lt_seed_1690`
- Seed count: 200
- Duplicate seed check: 0 duplicates
- Generation pool: random OpenRouter model pool
- Judge pool: random OpenRouter model pool

## Generation

- Generated rows: 200
- Finish status: 200 `stop`
- Raw output path: `tracks/leverage/runs/instruction-outputs/surface-constraint-batch-001-raw.jsonl`

Generator distribution:

- `qwen3-6-plus-openrouter`: 90
- `glm-5-1-openrouter`: 22
- `claude-sonnet-4-6-openrouter`: 21
- `gpt-5-4-openrouter`: 20
- `kimi-k2-6-openrouter`: 18
- `gpt-5-5-openrouter`: 16
- `deepseek-v4-pro-openrouter`: 13

## Filter

- Total rows: 200
- Sent to judge: 163
- Deterministic rejects: 37

Main deterministic issues:

- `response_too_long`: 110
- `word_count_not_140`: 24
- `punctuation_forbidden`: 13

Note: `response_too_long` can coexist with `needs_judge` in the current filter, so it is not equal to final rejection.

## Judge

- Judged rows: 163
- Accepted: 118
- Needs edit: 18
- Rejected: 21
- Parse error: 6

Judge-side errors:

- `judge_json_parse_error`: 3
- `provider_http_error`: 3

## Promotion

- Promoted rows: 118
- Reviewed id range: `instr_1101` through `instr_1218`
- Reviewed dataset after promotion: 1216 rows
- SFT export after promotion: 1216 rows

Accepted row distribution:

- `instruction_following`: 66
- `summarization_transformation`: 20
- `reasoning`: 15
- `knowledge_qa`: 15
- `coding`: 1
- `tool_use`: 1

Accepted answer length:

- Minimum words: 29
- Median words: 345.5
- Maximum words: 857
- At least 140 words: 93
- At least 260 words: 78
- At least 320 words: 62

## Verification

- `uv run python -m llm.leverage.validate_reviewed_instructions tracks/leverage/datasets/reviewed-instructions/bootstrap.jsonl`
- `uv run python -m llm.leverage.export_reviewed_instructions --overwrite`
