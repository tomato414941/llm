# Reviewed Batch 006

Date: 2026-04-30

## Purpose

Scale the reviewed-data pipeline from 20-row batches to a 100-row batch. This
batch intentionally prioritizes throughput over per-failure prompt repair.

## Inputs

- Added seed range: `lt_seed_271` through `lt_seed_370`
- Raw generated rows: 100
- Target capability mix:
  - `reasoning`: 25
  - `instruction_following`: 25
  - `coding`: 20
  - `tool_use`: 15
  - `knowledge_qa`: 10
  - `summarization_transformation`: 5

## Generation

Generation used a fixed low-cost teacher model:

- generator: `qwen3-6-plus-openrouter` / `qwen/qwen3.6-plus`
- temperature: 0.1
- max tokens: 512
- reasoning effort: `none`
- exclude reasoning: true

Recorded generation cost from provider usage metadata: `$0.005295225`.

## Filter Result

- generated rows: 100
- `needs_judge`: 100
- deterministic rejects: 0

## Judge Result

Judging used a fixed stable judge:

- judge: `gpt-5-4-openrouter` / `openai/gpt-5.4`
- parse_error: 0
- accept: 64
- needs_edit: 35
- reject: 1

Accepted rows by capability:

- `instruction_following`: 24
- `reasoning`: 14
- `coding`: 9
- `knowledge_qa`: 8
- `tool_use`: 5
- `summarization_transformation`: 4

## Promotion

Promoted accepted rows directly, then removed one exact duplicate found by the
near-duplicate summary:

- promoted accepted rows: 64
- removed duplicate: `instr_0197`, same prompt as existing `instr_0045`
- net reviewed rows added: 63

Reviewed dataset size after promotion: 205 rows.

## Notes

This confirms the larger-batch path is viable:

- 100 generated candidates produced 63 net reviewed rows.
- Deterministic filtering and fixed GPT-5.4 judging handled the full batch
  without parse errors.
- The main quality issue after promotion was duplication, not API instability.

The next scale improvement should be a simple pre-promotion duplicate gate
against existing reviewed prompts. That is more valuable than manually repairing
the 35 `needs_edit` rows.

Follow-up:

- Added a reviewed-dataset validation gate for exact duplicate user prompts.
- The current 205-row reviewed dataset passes this gate.
- Future bulk promotion should run `validate_reviewed_instructions` before
  export; exact prompt reuse will fail validation instead of reaching SFT export.

## Artifacts

- `tracks/leverage/runs/instruction-outputs/readiness-batch-006-raw.jsonl`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-006-filter.csv`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-006-filter-summary.csv`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-006-candidates.jsonl`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-006-judgments.jsonl`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-006-judgments.csv`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-006-judgments-summary.csv`
