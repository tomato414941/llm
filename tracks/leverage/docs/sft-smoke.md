# Leverage SFT Smoke Plan

This plan defines the first weight-changing leverage experiment. It is a smoke
test for the data and training path, not a claim that the model improves.

## Objective

Verify that reviewed instructions can be exported into training JSONL and used
by a small student model in a bounded LoRA or SFT run.

## Inputs

- Reviewed source data: `tracks/leverage/datasets/reviewed-instructions/bootstrap.jsonl`
- Training export: `tracks/leverage/sft/bootstrap.train.jsonl`
- Held-out evals:
  - `tracks/leverage/evals/leverage-smoke.jsonl`
  - `tracks/leverage/evals/project-judgment-v0.jsonl`
- Config: `tracks/leverage/configs/leverage-sft-smoke.toml`

## Student Model

Start with `Qwen/Qwen3-0.6B`. The first run should prefer a small student
because the goal is wiring, not capability.

## Method

Prefer LoRA for the first run. Full SFT is acceptable only if the implementation
is simpler in the selected training stack and remains bounded to the 10 reviewed
examples.

## Before Training

```bash
uv run python -m llm.leverage.validate_reviewed_instructions \
  tracks/leverage/datasets/reviewed-instructions/bootstrap.jsonl

uv run python -m llm.leverage.export_reviewed_instructions --overwrite
```

Do not launch a paid GPU job until the reviewed instructions validate and the
training export is regenerated locally.

The preferred local preflight combines those checks with the smoke config
constraints:

```bash
uv run python -m llm.leverage.sft_smoke_preflight --overwrite
```

This command validates the reviewed instruction source, regenerates the training
export, checks the eval task paths, verifies the export row count stays within
`max_train_examples`, and confirms that RunPod is not required for preflight. It
does not train a model, call external APIs, download weights, or start paid GPU
resources.

## RunPod Dry Run

After local preflight passes, inspect the RunPod execution plan without creating
a pod:

```bash
uv run python scripts/runpod/run_once.py \
  --dry-run \
  --name llm-leverage-sft-smoke \
  --gpu-type 'NVIDIA GeForce RTX 3090' \
  --max-cost 0.8 \
  --mem 24 \
  --sync tracks/leverage/configs \
  --sync tracks/leverage/datasets \
  --sync tracks/leverage/evals \
  --sync tracks/leverage/sft \
  --output outputs/leverage-sft-smoke \
  --local 'uv run python -m llm.leverage.sft_smoke_preflight --config tracks/leverage/configs/leverage-sft-smoke.toml --overwrite' \
  --remote 'uv pip install transformers peft trl accelerate' \
  --remote 'uv run python -u -c "import torch; import transformers; import peft; import trl; print(\"training packages import ok\")"' \
  --remote 'uv run python -u -m llm.leverage.train_sft_smoke --config tracks/leverage/configs/leverage-sft-smoke.toml' \
  --remote 'uv run python -u -m llm.leverage.evaluate_sft_adapter --config tracks/leverage/configs/leverage-sft-smoke.toml'
```

The dry run must show these steps in order:

- local SFT smoke preflight
- RunPod pod creation with the configured cost and runtime ceiling
- repo, reviewed data, SFT export, and eval task sync
- CUDA smoke
- training-package import smoke
- 10-row LoRA/SFT smoke command
- post-training base-vs-adapter eval command
- artifact, metrics, and notes sync from `outputs/leverage-sft-smoke`
- cleanup

Do not run the same command without `--dry-run` until the dry-run plan matches
the intended cost, GPU, image, model, output paths, and cleanup policy.

The default image is `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, chosen
for this training smoke because it provides a newer PyTorch/CUDA base than the
older from-scratch training image. Keep the first run on the default image
unless the dry-run review identifies a concrete compatibility issue.

## Success Criteria

- The reviewed instruction file validates.
- The training JSONL export is regenerated.
- The training command completes on at most 10 examples.
- An adapter or checkpoint artifact is written.
- The held-out eval command can run before and after training.

## Post-Training Eval

After a smoke adapter exists under `outputs/leverage-sft-smoke/adapter/`, compare
the base student and adapter on the same eval tasks:

```bash
uv run python -m llm.leverage.evaluate_sft_adapter --dry-run
```

Remove `--dry-run` only in an environment that can load the base model and
adapter. The command writes predictions, detailed scores, and summary scores
under `outputs/leverage-sft-smoke/`. Treat this as a wiring comparison, not as a
capability claim.

## Stop Conditions

- Dependency setup exceeds the planned runtime.
- The selected student model cannot load on the available GPU.
- Training loss becomes NaN.
- A paid run would exceed the cost cap.
- A paid resource cannot be cleaned up immediately.

## RunPod Policy

RunPod is not required for this plan by default. Use it only if local execution
cannot run the selected student model and the smoke objective still justifies
paid GPU time. If RunPod is used, set a hard cost cap, save only the expected
artifacts, and verify cleanup at the end.
