# Leverage Runs

This directory stores concise, reviewable leverage-track run records and small
example artifacts. It is not the source of truth for datasets or generated SFT
exports.

## Read First

- `leverage-sft-smoke-runpod.md`: first weight-changing LoRA/SFT smoke run on
  RunPod, including command, metrics, and cleanup confirmation.
- `leverage-sft-smoke-failure-triage.md`: failure classification for the first
  SFT smoke and the next concrete fix.
- `model-spec-comparison.md`: current summary of `leverage-model-spec.jsonl`
  scoring changes and model comparison results.

These files are the current human-readable run notes.

## Examples

- `predictions.example.jsonl`: minimal prediction-file shape for
  `llm.leverage.evaluate`.
- `two-layer.example.jsonl`: example predictions across the smoke and
  project-judgment eval layers.
- `project-judgment-v0.example.jsonl`: example output for project judgment
  tasks.
- `leverage-model-spec.example.jsonl`: example output for the policy guard eval.

## Historical Model-Spec Runs

Files matching these patterns are historical comparison artifacts:

- `*-model-spec*.jsonl`
- `*-model-spec*-scores.csv`
- `*-model-spec*-summary.csv`

Keep them only when they explain a decision in `model-spec-comparison.md`.
Otherwise prefer a short markdown note over adding more raw run files.

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

- `leverage-sft-smoke-runpod.md`
- `leverage-sft-smoke-failure-triage.md`

Avoid adding another family of raw `*-scores.csv` / `*-summary.csv` files unless
the raw files are directly needed for a comparison note.
