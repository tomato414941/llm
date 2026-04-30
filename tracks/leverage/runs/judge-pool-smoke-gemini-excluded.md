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

## Follow-up 20-Row Check

The same judge pool was run on the first 20 candidates to check whether the
parse error was isolated.

- judged rows: 20
- Gemini judge rows: 0
- accept: 8
- needs_edit: 9
- reject: 2
- parse_error: 1

The parse error again came from `gpt-5-4-openrouter` judging a
`qwen3-6-plus-openrouter` generated row. The JSON was otherwise valid, but the
score object used `safe` instead of `safety`.

This is small enough to handle with the existing strict parser and parse-error
recording. Do not add schema repair yet; rerun or skip parse-error rows when
promoting reviewed data.

## Follow-up With JSON Example

The judge prompt was then tightened by adding an explicit JSON example and a
direct instruction not to use synonyms such as `safe` for the `safety` score key.
The same 20-row check was rerun with the same random seed and judge pool.

- judged rows: 20
- Gemini judge rows: 0
- accept: 11
- needs_edit: 9
- parse_error: 0

This supports the small prompt-only fix. The next batch can keep the strict
parser; no schema repair or structured-output integration is needed yet.

## Artifacts

These are local run artifacts under the ignored instruction-output directory:

- `tracks/leverage/runs/instruction-outputs/judge-pool-smoke-gemini-excluded-judgments.jsonl`
- `tracks/leverage/runs/instruction-outputs/judge-pool-smoke-gemini-excluded-judgments.csv`
- `tracks/leverage/runs/instruction-outputs/judge-pool-smoke-gemini-excluded-summary.csv`
- `tracks/leverage/runs/instruction-outputs/judge-pool-smoke-gemini-excluded-20-judgments.jsonl`
- `tracks/leverage/runs/instruction-outputs/judge-pool-smoke-gemini-excluded-20-judgments.csv`
- `tracks/leverage/runs/instruction-outputs/judge-pool-smoke-gemini-excluded-20-summary.csv`
- `tracks/leverage/runs/instruction-outputs/judge-pool-smoke-json-example-20-judgments.jsonl`
- `tracks/leverage/runs/instruction-outputs/judge-pool-smoke-json-example-20-judgments.csv`
- `tracks/leverage/runs/instruction-outputs/judge-pool-smoke-json-example-20-summary.csv`
