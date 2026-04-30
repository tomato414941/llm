# Reviewed Instruction Batch 008

Batch 008 starts from the batch 007 failure mode: accepted rows were mostly good,
but many prompts were exact duplicates and could not be promoted.

## Seeds

Added 200 seeds:

- `reasoning`: 80
- `coding`: 45
- `tool_use`: 30
- `knowledge_qa`: 20
- `summarization_transformation`: 15
- `instruction_following`: 10

The mix prioritizes the remaining reviewed deficits, especially `reasoning`,
while still adding coverage for the smaller deficits.

## Pre-Generation Duplicate Report

Ran the exact-duplicate seed report before any teacher-model call:

```text
selected_seed_count,200
duplicate_seed_count,0
duplicate_row_count,0
```

This checks the full generation user prompt against the selected batch and the
existing reviewed instruction dataset.

## Next Step

Generated candidates for `lt_seed_571` through `lt_seed_770`, then ran the
existing filter, judge, and promotion gates.

## Generation

- generator: `qwen3-6-plus-openrouter` / `qwen/qwen3.6-plus`
- temperature: 0.1
- max tokens: 512
- reasoning effort: `none`
- exclude reasoning: true
- generated rows: 200
- finish_reason `stop`: 200
- recorded generation cost from provider usage metadata: `$0.017867850`

## Filter Result

- generated rows: 200
- `needs_judge`: 200
- deterministic rejects: 0

## Judge Result

- judge: `gpt-5-4-openrouter` / `openai/gpt-5.4`
- parse_error: 0
- accept: 197
- needs_edit: 3
- reject: 0
- average correctness: 1.980
- average instruction following: 1.990
- average conciseness: 2.000
- average safety: 2.000

Not accepted:

- `lt_seed_601`: judge returned `needs_edit`, but its free-text reason appears
  internally inconsistent.
- `lt_seed_625`: arithmetic answer was incorrect.
- `lt_seed_658`: code answer used Markdown fences where raw function code was
  requested.

## Promotion

Promoted all exact-prompt-unique accepted rows:

- accepted rows: 197
- promoted rows: 197
- skipped exact duplicate user prompts: 0

Promoted rows by capability:

- `reasoning`: 78
- `coding`: 44
- `tool_use`: 30
- `knowledge_qa`: 20
- `summarization_transformation`: 15
- `instruction_following`: 10

Reviewed dataset size after promotion: 481 rows.

All current reviewed capability targets are now met:

```text
capability,reviewed_count,reviewed_target,reviewed_deficit
coding,87,50,0
instruction_following,111,80,0
knowledge_qa,48,30,0
reasoning,134,80,0
summarization_transformation,38,25,0
tool_use,63,35,0
```

## Artifacts

- `tracks/leverage/runs/instruction-outputs/readiness-batch-008-raw.jsonl`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-008-filter.csv`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-008-filter-summary.csv`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-008-candidates.jsonl`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-008-judgments.jsonl`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-008-judgments.csv`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-008-judgments-summary.csv`
