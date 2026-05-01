# Reviewed Instruction Batch 012

Batch 012 is a 200-seed random-pool growth run. The purpose was to test whether
the pipeline can handle a larger batch after generator failures started being
recorded as `generation_error` rows.

## Seeds

Added 200 seeds:

- `reasoning`: 34
- `coding`: 34
- `tool_use`: 33
- `knowledge_qa`: 33
- `summarization_transformation`: 33
- `instruction_following`: 33

## Pre-Generation Duplicate Report

The first generated seed set had duplicate `knowledge_qa` prompts. Those seed
prompts were revised before generation.

Final duplicate report:

```text
selected_seed_count,200
duplicate_seed_count,0
duplicate_row_count,0
```

## Generation

Generation used the default random generator pool; no fixed `--model` or
`--model-label` was passed.

- generated rows: 200
- raw rows: 198
- `generation_error` rows: 2
- finish_reason `stop`: 198
- deterministic filter `needs_judge`: 198
- deterministic filter reject: 2
- recorded generation cost from provider usage metadata: `$0.114502669`

Generation errors:

- `lt_seed_1029`: `kimi-k2-6-openrouter`, `provider_content_error`
- `lt_seed_1167`: `kimi-k2-6-openrouter`, `provider_content_error`

This confirms the new generator failure handling works in a live larger batch:
provider content errors were recorded and filtered out, and the batch continued
to completion.

Generator distribution:

- `qwen3-6-plus-openrouter`: 73
- `claude-sonnet-4-6-openrouter`: 24
- `deepseek-v4-pro-openrouter`: 23
- `glm-5-1-openrouter`: 23
- `gpt-5-5-openrouter`: 22
- `gpt-5-4-openrouter`: 20
- `kimi-k2-6-openrouter`: 15

## Judge Result

Judging used the default random judge pool with self-judge exclusion and the
one-retry transient failure policy.

- judged rows: 198
- self-judge rows: 0
- accept: 167
- needs_edit: 26
- reject: 4
- parse_error: 1
- retry attempts: 3
- retry recovered to accept: 2

Retry outcomes:

- `lt_seed_977`: `judge_json_parse_error` -> `accept`
- `lt_seed_1052`: `judge_json_parse_error` -> `parse_error`
- `lt_seed_1075`: `provider_content_error` -> `accept`

Judge distribution:

- `qwen3-6-plus-openrouter`: 39
- `kimi-k2-6-openrouter`: 38
- `claude-sonnet-4-6-openrouter`: 35
- `gpt-5-4-openrouter`: 23
- `gpt-5-5-openrouter`: 22
- `deepseek-v4-pro-openrouter`: 22
- `glm-5-1-openrouter`: 19

## Promotion

Promoted exact-prompt-unique accepted rows:

- accepted rows: 167
- promoted rows: 167
- skipped exact duplicate user prompts: 0
- reviewed dataset size after promotion: 835 rows

## Current Capability Distribution

```csv
capability,seed_count,reviewed_count,eval_count,reviewed_target,reviewed_deficit
coding,217,147,5,50,0
instruction_following,189,163,5,80,0
knowledge_qa,133,106,3,30,0
reasoning,321,216,27,80,0
summarization_transformation,106,91,2,25,0
tool_use,204,112,6,35,0
```

## Operational Notes

The 200-row generation run completed, but the 198-row judge run was slow enough
that future large growth should prefer smaller chunks or resumable judge output
before moving to 500-row batches.

The dominant `needs_edit` pattern remains strict-format drift, especially code
answers wrapped in Markdown fences and command answers adding extra structure.
This is not yet treated as a global rejection rule because those formats can be
valid in other task shapes.

## Artifacts

- `tracks/leverage/runs/instruction-outputs/readiness-batch-012-seed-duplicates.csv`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-012-raw.jsonl`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-012-filter.csv`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-012-filter-summary.csv`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-012-candidates.jsonl`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-012-judgments.jsonl`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-012-judgments.csv`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-012-judgments-summary.csv`
