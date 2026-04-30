# Judge Pool Smoke: Gemini Excluded

Date: 2026-04-30

## Purpose

Check whether excluding `google/gemini-3.1-pro-preview` from judging is enough
to stabilize the next reviewed-instruction batch.

## Setup

- Input: `tracks/leverage/runs/instruction-outputs/readiness-batch-003-candidates.jsonl`
- Limit: 5 candidates
- Reasoning policy: `reasoning_effort=none`, `exclude_reasoning=true`
- Judge candidates:
  - `qwen3-6-plus-openrouter` / `qwen/qwen3.6-plus` / 0.70
  - `gpt-5-4-openrouter` / `openai/gpt-5.4` / 0.30

## Result

- judged rows: 5
- Gemini judge rows: 0
- accept: 3
- needs_edit: 1
- parse_error: 1

The parse error came from `gpt-5-4-openrouter` judging a
`qwen3-6-plus-openrouter` generated row. The response was valid JSON, but the
score object used `safe` instead of the required `safety` key.

## Interpretation

Removing Gemini from the judge pool avoids the known reasoning-mandatory
endpoint conflict, but it does not prove the judge path is fully stable. The
remaining instability is a smaller output-contract issue: a judge can still
return a near-valid schema that fails strict parsing.

Keep the immediate fix small:

- leave Gemini out of the default judge pool
- keep recording parse errors instead of promoting those rows
- do not add broader judge infrastructure until a larger batch shows this is a
  real bottleneck

## Artifacts

These are local run artifacts under the ignored instruction-output directory:

- `tracks/leverage/runs/instruction-outputs/judge-pool-smoke-gemini-excluded-judgments.jsonl`
- `tracks/leverage/runs/instruction-outputs/judge-pool-smoke-gemini-excluded-judgments.csv`
- `tracks/leverage/runs/instruction-outputs/judge-pool-smoke-gemini-excluded-summary.csv`
