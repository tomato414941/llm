# Reviewed Dataset Audit at 125 Rows

Date: 2026-04-30

## Purpose

Check the reviewed instruction dataset before starting another growth batch.
The goal is to decide whether any reviewed rows should be removed now and what
capability mix batch-005 should target.

## Current Size

- reviewed rows: 125
- exported SFT rows: 125

## Capability Distribution

| capability | reviewed | target | deficit |
| --- | ---: | ---: | ---: |
| instruction_following | 34 | 80 | 46 |
| reasoning | 31 | 80 | 49 |
| coding | 22 | 50 | 28 |
| tool_use | 15 | 35 | 20 |
| summarization_transformation | 12 | 25 | 13 |
| knowledge_qa | 11 | 30 | 19 |

Interpretation:

- `reasoning` and `instruction_following` still have the largest numeric
  deficits, but they have also received the most recent attention.
- `tool_use`, `coding`, and `knowledge_qa` are smaller and should be emphasized
  in the next batch to avoid a narrow dataset.
- `summarization_transformation` is still useful, but the current count is
  closer to the target than the other under-covered areas.

## Near-Duplicate Check

The highest near-duplicate score is 0.600:

- `instr_0109` / `instr_0115`: exact-count no-punctuation tasks with different
  required words and counts.

Other notable pairs:

- `instr_0096` / `instr_0112` at 0.583: both are coding review answers about
  mixed formatting/logic changes.
- `instr_0104` / `instr_0122` at 0.360: both are paid GPU cost-control tasks.
- JSON and exact-text instruction-following tasks recur, but with different
  required keys, labels, or counts.

Decision:

- Do not delete any row for duplication now.
- The observed pairs are repeated task shapes, not literal duplicate training
  targets.
- Avoid adding more formatting-vs-logic PR review examples in the next batch
  unless they test a clearly different behavior.

## Long-Answer Spot Check

Longest reviewed answers:

- `instr_0063` / `lt_seed_116`: 1372 chars, paid GPU SSH readiness risk.
- `instr_0061` / `lt_seed_112`: 1066 chars, caching tradeoff for monthly script.
- `instr_0064` / `lt_seed_118`: 719 chars, LoRA adapter evidence judgment.
- `instr_0125` / `lt_seed_242`: 556 chars, three bullet artifact-preservation
  checklist.
- `instr_0075` / `lt_seed_148`: 434 chars, config/docs before training.
- `instr_0105` / `lt_seed_215`: 341 chars, inspect failure output first.

Decision:

- Do not remove these rows now. They are not wrong, unsafe, or duplicates.
- Treat `instr_0063`, `instr_0061`, and `instr_0064` as legacy verbose rows.
  They should not set the style target for future reviewed data.
- If a future cleanup pass trims rows, start with those three and preserve the
  same source prompt ids only if the revised target still satisfies the original
  constraints.
- `instr_0125` is long but acceptable because the prompt requested bullet
  preservation checks.

## Batch-005 Target

Recommended next batch size: 20 seeds.

Target mix:

- `tool_use`: 7
- `coding`: 6
- `knowledge_qa`: 4
- `summarization_transformation`: 3
- `reasoning`: 0
- `instruction_following`: 0

Rationale:

- Recent work already added many reasoning and instruction-following rows.
- `tool_use` remains small relative to its target and is important for this
  project's practical assistant behavior.
- `coding` needs more focused regression, validation, and review examples.
- `knowledge_qa` and `summarization_transformation` keep the dataset from
  becoming too project-agent-specific.

## Prompt Rules for Batch-005

- For `short_answer`, put the sentence count in the generator-visible prompt,
  not only in `constraints`.
- Avoid broad "explain" prompts unless the target is truly an explanation row.
- For strict labels, JSON, or exact text, put the exact contract in the user
  prompt.
- Prefer one seed per behavior. Do not create multiple examples that only
  rename the same PR-review or GPU-cost scenario.

