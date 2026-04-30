# Reviewed Batch 005

Date: 2026-04-30

## Purpose

Add a small batch focused on under-covered capabilities after the 125-row
reviewed dataset audit.

## Inputs

- Added seed range: `lt_seed_251` through `lt_seed_270`
- Raw generated rows: 20
- Target capability mix:
  - `tool_use`: 7
  - `coding`: 6
  - `knowledge_qa`: 4
  - `summarization_transformation`: 3

## Generation

Generation used a fixed teacher model to test seed quality before reintroducing
model diversity:

- generator: `qwen3-6-plus-openrouter` / `qwen/qwen3.6-plus`
- temperature: 0.1
- max tokens: 512
- reasoning effort: `none`
- exclude reasoning: true

Recorded generation cost from provider usage metadata: `$0.001413425`.

## Filter Result

- generated rows: 20
- `needs_judge`: 20
- deterministic rejects: 0

## Judge Result

Judging used a fixed stable judge:

- judge: `gpt-5-4-openrouter` / `openai/gpt-5.4`
- parse_error: 0
- accept: 17
- needs_edit: 3
- reject: 0

Not accepted:

- `lt_seed_260`: exceeded the at-most-two-sentences constraint.
- `lt_seed_261`: did not explicitly mention a clear parse error or avoiding
  hidden errors.
- `lt_seed_270`: did not explicitly mention that the backup completed.

## Promotion

Promoted the 17 directly usable accepted rows:

- `lt_seed_251` -> `instr_0127`
- `lt_seed_252` -> `instr_0128`
- `lt_seed_253` -> `instr_0129`
- `lt_seed_254` -> `instr_0130`
- `lt_seed_255` -> `instr_0131`
- `lt_seed_256` -> `instr_0132`
- `lt_seed_257` -> `instr_0133`
- `lt_seed_258` -> `instr_0134`
- `lt_seed_259` -> `instr_0135`
- `lt_seed_262` -> `instr_0136`
- `lt_seed_263` -> `instr_0137`
- `lt_seed_264` -> `instr_0138`
- `lt_seed_265` -> `instr_0139`
- `lt_seed_266` -> `instr_0140`
- `lt_seed_267` -> `instr_0141`
- `lt_seed_268` -> `instr_0142`
- `lt_seed_269` -> `instr_0143`

Reviewed dataset size after promotion: 142 rows.

## Artifacts

- `tracks/leverage/runs/instruction-outputs/readiness-batch-005-raw.jsonl`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-005-filter.csv`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-005-filter-summary.csv`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-005-candidates.jsonl`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-005-judgments.jsonl`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-005-judgments.csv`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-005-judgments-summary.csv`

