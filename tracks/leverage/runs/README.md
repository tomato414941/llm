# Leverage Runs

This directory stores concise records for concrete leverage-track work:
training runs, benchmark runs, data-generation batches, dataset audits,
operational probes, and small example artifacts.

It is not the source of truth for reviewed data, generated artifacts, or
cross-run strategy.

Use:

- `tracks/leverage/datasets/` for reviewed training data.
- `tracks/leverage/sft/` for generated SFT exports.
- `outputs/` for generated predictions, adapters, checkpoints, and benchmark
  artifacts.
- `tracks/leverage/docs/` for policies, reusable procedures, and cross-run
  analysis.

## Boundary

Put notes here when they answer:

- What was run or generated?
- What inputs, command, model, hardware, and cost were used?
- What metrics or artifacts were produced?
- Was cleanup completed?
- What is the immediate next decision from this single run or batch?

Put notes in `tracks/leverage/docs/` when they answer:

- What is the current strategy across several runs?
- Which result supersedes another result?
- How should data mix, benchmark policy, or training policy change?
- What should the next experiment design be?

## Read First

For current strategy:

- `../README.md`
- `../docs/reviewed-instruction-mix-plan.md`
- `../docs/lora-sft-runpod.md`
- `../docs/qwen35-9b-adapter-regression-analysis.md`

For current run evidence:

- `qwen35-9b-baseline-1216-eval.md`
- `lm-harness-ifeval-adapter-1216-full.md`
- `reviewed-dataset-audit-1216.md`

## Conventions

- Keep run notes short and evidence-focused.
- Do not commit adapters, checkpoints, generated SFT exports, large raw outputs,
  or secrets.
- Use ignored paths such as `outputs/` and `tracks/leverage/sft/` for generated
  artifacts.
- Keep raw instruction-generation outputs under `instruction-outputs/`;
  promoted rows belong in `datasets/`.
- Prefer names like `<topic>-<purpose>.md`.
