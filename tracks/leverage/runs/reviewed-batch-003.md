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

## Follow-up: Remaining 10 Rows

Date: 2026-04-30

After adding an explicit JSON example to the judge prompt, the 10 previously
unjudged candidates (`lt_seed_221` through `lt_seed_230`) were judged with the
restricted judge pool:

- `qwen3-6-plus-openrouter` / `qwen/qwen3.6-plus` / 0.70
- `gpt-5-4-openrouter` / `openai/gpt-5.4` / 0.30

Result:

- judged rows: 10
- Gemini judge rows: 0
- parse_error: 0
- accept: 4
- needs_edit: 4
- reject: 2

All 10 rows were judged by `qwen3-6-plus-openrouter` after non-self exclusion
and weighted sampling.

The 4 accepted rows still require promotion review:

- `lt_seed_225`: usable candidate; two-sentence coding review answer.
- `lt_seed_227`: not directly promotable; answer is fenced JSON despite a
  `valid JSON only` constraint.
- `lt_seed_228`: not directly promotable; answer has punctuation despite a
  `no punctuation` constraint.
- `lt_seed_229`: usable candidate; one-sentence neutral reschedule answer.

This confirms the judge path is operational after the prompt fix, but it also
shows that judge `accept` is not enough for automatic promotion. Deterministic
format constraints still need direct inspection or checks before adding reviewed
rows.

Follow-up artifacts:

- `tracks/leverage/runs/instruction-outputs/readiness-batch-003-unjudged-json-example-judgments.jsonl`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-003-unjudged-json-example-judgments.csv`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-003-unjudged-json-example-summary.csv`

## Follow-up: JSON Fence Regeneration Check

Date: 2026-04-30

After adding `do not wrap JSON in Markdown code fences` to `json_object` seed
constraints, `lt_seed_227` was regenerated once with
`qwen3-6-plus-openrouter`.

Result:

- generated rows: 1
- finish_reason: `stop`
- Markdown code fence: no
- direct `json.loads` parse: ok
- structural filter: `needs_judge`
- non-self judge: `gpt-5-4-openrouter`
- judge decision: `accept`
- judge parse_error: 0

The regenerated answer was raw JSON:

```json
{
  "enabled": false,
  "retries": 2
}
```

This confirms the narrow JSON-only seed constraint fixed the observed fenced
JSON issue for the failing seed without broadening the rule to non-JSON tasks.

Follow-up artifacts:

- `tracks/leverage/runs/instruction-outputs/lt-seed-227-json-fence-check-raw.jsonl`
- `tracks/leverage/runs/instruction-outputs/lt-seed-227-json-fence-check-filter-v2.csv`
- `tracks/leverage/runs/instruction-outputs/lt-seed-227-json-fence-check-filter-v2-summary.csv`
- `tracks/leverage/runs/instruction-outputs/lt-seed-227-json-fence-check-candidates-v2.jsonl`
- `tracks/leverage/runs/instruction-outputs/lt-seed-227-json-fence-check-judgments.jsonl`
- `tracks/leverage/runs/instruction-outputs/lt-seed-227-json-fence-check-judgments.csv`
- `tracks/leverage/runs/instruction-outputs/lt-seed-227-json-fence-check-judgments-summary.csv`

## Follow-up: Promotion

Date: 2026-04-30

Promoted only the follow-up candidates that were directly usable as reviewed
training rows:

- `lt_seed_225` -> `instr_0106`
- regenerated `lt_seed_227` -> `instr_0107`
- `lt_seed_229` -> `instr_0108`

Skipped the originally accepted but format-invalid candidates:

- `lt_seed_227` original output: fenced JSON, replaced by the regenerated raw
  JSON candidate.
- `lt_seed_228`: included punctuation despite a `no punctuation` constraint.

Reviewed dataset size after promotion: 107 rows.

The near-duplicate summary did not show a strong duplicate for the promoted
rows. The highest prompt-level similarity involving them was 0.500
(`instr_0069` / `instr_0108`), which is a similar tone-constrained reschedule
task but not a copy.
