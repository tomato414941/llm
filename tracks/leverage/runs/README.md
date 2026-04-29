# Leverage Runs

This directory stores concise, reviewable leverage-track run records and small
example artifacts. It is not the source of truth for datasets or generated SFT
exports.

## Read First

- `data-category-design.md`: shared top-level capability taxonomy for seed
  prompts, reviewed instruction rows, and held-out eval tasks.
- `reviewed-instruction-growth-plan.md`: current distribution plan for growing
  the reviewed dataset toward the next capability-seeking LoRA run.
- `next-lora-data-targets.md`: minimum reviewed-data and eval scale before the
  next capability-seeking LoRA run.
- `leverage-sft-smoke-runpod-59-secure-success.md`: current RunPod Secure Cloud
  LoRA/SFT smoke result, including command outcome, metrics, and cleanup
  confirmation.
- `model-spec-comparison.md`: current summary of `leverage-model-spec.jsonl`
  scoring changes and model comparison results.

These files are the current human-readable run notes.

## Examples

- `predictions.example.jsonl`: minimal prediction-file shape for
  `llm.leverage.evaluate`.
- `two-layer.example.jsonl`: example predictions across the smoke and
  project-judgment eval layers.
- `project-judgment.example.jsonl`: example output for project judgment
  tasks.
- `leverage-model-spec.example.jsonl`: example output for the policy guard eval.

## Historical Model-Spec Runs

`model-spec-comparison.md` is the committed source of truth for historical
model-spec comparisons. Do not commit raw `*-model-spec*.jsonl`,
`*-scores.csv`, or `*-summary.csv` files for that comparison family.

Keep only small example files, such as `leverage-model-spec.example.jsonl`,
when they document evaluator input or output shape.

## Instruction-Output Runs

`instruction-outputs/` contains raw teacher outputs, structural-filter results,
candidate files, judge outputs, and summaries used before manual promotion into
`tracks/leverage/datasets/reviewed-instructions/`.

Those files are run artifacts. They are not reviewed training data. The reviewed
dataset is the source of truth after manual promotion.

## What Not To Put Here

Do not add:

- LoRA adapters or model checkpoints.
- Generated SFT exports.
- Large raw model outputs.
- Secrets, API keys, provider responses containing credentials, or local
  machine details.

Use ignored paths such as `outputs/` and `tracks/leverage/sft/` for generated
artifacts. If a result matters, commit a short markdown run note with the
command, success criteria, summary metrics, cleanup status, and next decision.

## Naming Rule

For new committed notes, prefer:

```text
<topic>-<purpose>.md
```

Examples:

- `leverage-sft-smoke-runpod-59-secure-success.md`

Avoid adding another family of raw `*-scores.csv` / `*-summary.csv` files unless
the raw files are directly needed for a comparison note.
