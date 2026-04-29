# Reviewed Instruction Review Rules

Use these rules before promoting generated answers into a committed reviewed
instruction file.

Reviewed instructions are a bootstrap mechanism and promotion boundary, not a
manual labeling strategy for building capability at scale. Prefer structural
filtering and model judging for large batches, then use these rules for sparse
promotion and spot checks.

Raw generated answers remain experiment artifacts. The expected raw location is
`tracks/leverage/runs/instruction-outputs/`; the reviewed location is
`tracks/leverage/datasets/reviewed-instructions/`.

## Required Checks

- The row is not copied from a held-out eval prompt.
- The assistant answer is correct for the prompt.
- The answer follows the requested format and constraints.
- The answer contains no API keys, tokens, private paths, or local environment
  details.
- The answer is useful for the intended student model behavior.
- The answer is not just a generic explanation that ignores the project context.
- The row has clear provenance in `source_prompt_id` and `review`.

## Review Status

Use `accepted_instruction` only when the row passed review and can be considered
for a small SFT/LoRA experiment. Do not use it for rows that only passed the
structural filter.

Do not use this status for raw generated outputs. Raw outputs should remain
under `tracks/leverage/runs/instruction-outputs/` until reviewed.

Use judge decisions consistently before promotion:

- `accept`: the row is suitable for promotion consideration after reviewer
  read-through.
- `needs_edit`: the row has useful content but is not directly training-ready.
  Use this for salvageable rows, including incomplete answers or answers that
  miss explicit constraints but can be repaired cheaply.
- `reject`: the row should leave the promotion path because it is wrong, unsafe,
  too incomplete, private, contaminated, or too expensive to repair.

For direct-promotion questions, only `accept` means yes. `needs_edit` and
`reject` both block immediate SFT use, but only `needs_edit` keeps the row as
edit material.

## Promotion Boundary

An accepted reviewed instruction is still not proof of value. It becomes useful only after:

- the training split is defined
- a held-out eval is selected
- a baseline model result exists
- a small training run shows measurable improvement

## Bootstrap Dataset Triage

Triage on 2026-04-29 for
`tracks/leverage/datasets/reviewed-instructions/bootstrap.jsonl` at 17 rows.
This is a training-readiness note only; no rows were removed.

Use in the first LoRA/SFT smoke:

- `instr_0001` through `instr_0012`
- `instr_0015`
- `instr_0017`

Hold out from the first smoke unless the experiment explicitly tests
label-only behavior:

- `instr_0013`: correct `track_distinction` label, but only `research`.
- `instr_0014`: correct `track_distinction` label, but only `operations`.
- `instr_0016`: correct `experiment_judgment` label, but only `REPEAT`.

Reason:

- The held rows are valid reviewed instructions, but they are very short
  classification targets.
- In a tiny bootstrap dataset, label-only rows can overrepresent terse answer
  behavior.
- They remain useful for future classification-focused experiments or a larger
  mixed dataset where label-only answers are a small minority.

Initial smoke recommendation:

- Train on 14 explanatory rows.
- Keep the 3 label-only rows reviewed but excluded from the first training
  subset.
- Revisit them after a baseline smoke run exists.
