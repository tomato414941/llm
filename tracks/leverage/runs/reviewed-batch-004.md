# Reviewed Batch 004

Date: 2026-04-30

## Purpose

Add a small general-purpose reviewed-instruction batch after tightening the
deterministic filter and judge prompt boundaries.

## Inputs

- Added seed range: `lt_seed_231` through `lt_seed_250`
- Raw generated rows: 20
- Target capability mix:
  - reasoning: 6
  - tool_use: 6
  - coding: 4
  - instruction_following: 4

## Generation

Generation used the default broad teacher pool from
`tracks/leverage/prompts/README.md` with random seed `4`.

The first generation attempt stopped after 7 rows because OpenRouter returned
HTTP 503. The run was resumed with `--resume` and completed all 20 rows.

Realized generator mix:

- `qwen3-6-plus-openrouter`: 13
- `gpt-5-4-openrouter`: 2
- `glm-5-1-openrouter`: 2
- `deepseek-v4-pro-openrouter`: 1
- `kimi-k2-6-openrouter`: 1
- `claude-sonnet-4-6-openrouter`: 1

Recorded generation cost from provider usage metadata: `$0.013414382`.

Two generated rows reported reasoning tokens despite the default
`reasoning_effort=none` and `exclude_reasoning=true` policy:

- `lt_seed_237` / `glm-5-1-openrouter`: 513 reasoning tokens
- `lt_seed_245` / `deepseek-v4-pro-openrouter`: 124 reasoning tokens

This is provider behavior to watch, but it did not block the batch because the
stored reviewed targets remain final-answer data.

## Filter Result

The deterministic filter passed all 20 rows to judging:

- generated rows: 20
- `needs_judge`: 20
- deterministic rejects: 0
- `response_too_long` warning: 2

The new deterministic checks also correctly reject the known batch-003 format
failures:

- `lt_seed_227`: `json_markdown_fence;invalid_json`
- `lt_seed_228`: `punctuation_forbidden`

## Judge Result

The first judge pass revealed a prompt-boundary issue: the judge sometimes
treated its own required JSON response shape as if it were a requirement for
candidate answers. This affected strict JSON and label-only candidates.

After clarifying that the JSON shape is only the judge response contract, the
same candidates were rejudged with the restricted non-self judge pool:

- `qwen3-6-plus-openrouter` / `qwen/qwen3.6-plus` / 0.70
- `gpt-5-4-openrouter` / `openai/gpt-5.4` / 0.30

Final judged result:

- judged rows: 20
- parse_error: 0
- accept: 10
- needs_edit: 9
- reject: 1

## Promotion

Promoted only directly usable accepted rows:

- `lt_seed_234` -> `instr_0110`
- `lt_seed_243` -> `instr_0111`
- `lt_seed_245` -> `instr_0112`
- `lt_seed_246` -> `instr_0113`
- `lt_seed_247` -> `instr_0114`
- `lt_seed_248` -> `instr_0115`
- `lt_seed_250` -> `instr_0116`

Accepted but not promoted:

- `lt_seed_233`: correct but too verbose for the reviewed dataset.
- `lt_seed_239`: correct but too verbose for the reviewed dataset.
- `lt_seed_244`: judge accepted it, but manual review found it weakens the
  required zero-or-negative timeout rejection by allowing zero if interpreted
  differently.

Reviewed dataset size after promotion: 115 rows.

## Artifacts

- `tracks/leverage/runs/instruction-outputs/readiness-batch-004-raw.jsonl`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-004-filter.csv`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-004-filter-summary.csv`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-004-candidates.jsonl`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-004-judgments.jsonl`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-004-judgments.csv`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-004-judgments-summary.csv`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-004-judgments-v2.jsonl`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-004-judgments-v2.csv`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-004-judgments-v2-summary.csv`

