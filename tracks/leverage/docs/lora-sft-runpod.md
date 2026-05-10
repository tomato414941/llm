# LoRA/SFT RunPod Guide

This is the canonical RunPod guide for leverage-track LoRA/SFT runs. It covers
smoke runs, bounded measurement runs, and short baseline-adapter training runs.
A successful run proves that the training path works. It is not a claim that
the model improves.

## Objective

Run a bounded weight-changing experiment from reviewed instructions, with
preflight, package checks, training, optional held-out eval, artifact sync, and
pod cleanup all explicit before paid GPU time starts.

## Inputs

- Reviewed source data: `tracks/leverage/datasets/reviewed-instructions/bootstrap.jsonl`
- Training export: `tracks/leverage/sft/bootstrap.train.jsonl`
- Held-out evals:
  - `tracks/leverage/evals/leverage-smoke.jsonl`
  - `tracks/leverage/evals/project-judgment.jsonl`
- Config examples:
  - `tracks/leverage/configs/leverage-sft-smoke.toml`
  - `tracks/leverage/configs/leverage-sft-qwen35-9b.toml`
  - `tracks/leverage/configs/leverage-sft-qwen35-9b-long-form-constraint.toml`

## Student Model

Use `Qwen/Qwen3.5-0.8B` as the test student for low-cost smoke runs. The intended
project baseline is `Qwen/Qwen3.5-9B` after the smoke path and eval loop are
stable. Do not treat a 0.8B smoke result as the baseline capability target.

## Model Roles

- Qwen test student: `Qwen/Qwen3.5-0.8B`
- Qwen baseline target: `Qwen/Qwen3.5-9B`
- DeepSeek test student: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- DeepSeek baseline target: `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`
- Challenging open-weight target: `openai/gpt-oss-20b`

Use the test students for smoke, preflight, and wiring checks. Use baseline
targets only after the smoke path and eval loop are stable. Treat
`openai/gpt-oss-20b` as a separate compatibility challenge because it is a MoE
model with different loading and adaptation concerns.

The first `openai/gpt-oss-20b` Transformers probe passed on a RunPod RTX 5090
with the official PyTorch 2.8.0 template. See
`tracks/leverage/runs/gpt-oss-20b-transformers-probe-20260510.md`. Do not use it
in quality benchmarks until the Harmony final-answer extraction or serving path
is explicit.

## Method

Prefer LoRA for current leverage-track training. Full SFT is acceptable only if
the selected stack makes it simpler and the run remains bounded by the chosen
config.

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
uv run python -m llm.leverage.sft_smoke_preflight \
  --config tracks/leverage/configs/leverage-sft-smoke.toml \
  --overwrite
```

This command validates the reviewed instruction source, regenerates the training
export, checks the eval task paths, verifies the export row count stays within
`max_train_examples`, and confirms that RunPod is not required for preflight. It
does not train a model, call external APIs, download weights, or start paid GPU
resources.

## Required Training Packages

`uv sync --extra dev` does not install the optional SFT training stack. Unless
the project dependency surface or RunPod image changes, each remote training
job must install and verify these packages before launching training:

```bash
uv pip install transformers peft trl accelerate

uv run python -u -c "import torch; import transformers; import peft; import trl; print(\"training packages import ok\")"
```

Keep this check close to the training command. A missing package is setup
failure, not evidence about the model, data, trainer, or GPU.

## RunPod Dry Run

After local preflight passes, inspect the RunPod execution plan without creating
a pod:

```bash
uv run python scripts/runpod/run_once.py \
  --dry-run \
  --name llm-leverage-sft-smoke \
  --secure-cloud \
  --gpu-type 'NVIDIA GeForce RTX 4090' \
  --template-id runpod-torch-v280 \
  --allowed-cuda-version 12.8 \
  --allowed-cuda-version 12.9 \
  --allowed-cuda-version 13.0 \
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
- RunPod Secure Cloud pod creation with `runpodctl pod create` and the
  configured runtime ceiling
- repo, reviewed data, SFT export, and eval task sync
- CUDA smoke
- training-package import smoke
- configured-row LoRA/SFT smoke command
- post-training base-vs-adapter eval command
- artifact, metrics, and notes sync from `outputs/leverage-sft-smoke`
- cleanup

For a 9B short training run, keep the same shape and change only the name,
config, and output path:

```bash
uv run python scripts/runpod/run_once.py \
  --dry-run \
  --name llm-leverage-qwen35-9b-lora \
  --secure-cloud \
  --gpu-type 'NVIDIA GeForce RTX 4090' \
  --template-id runpod-torch-v280 \
  --allowed-cuda-version 12.8 \
  --allowed-cuda-version 12.9 \
  --allowed-cuda-version 13.0 \
  --mem 24 \
  --sync tracks/leverage/configs \
  --sync tracks/leverage/datasets \
  --sync tracks/leverage/evals \
  --sync tracks/leverage/sft \
  --output outputs/leverage-sft-qwen35-9b-long-form-constraint \
  --local 'uv run python -m llm.leverage.sft_smoke_preflight --config tracks/leverage/configs/leverage-sft-qwen35-9b-long-form-constraint.toml --overwrite' \
  --remote 'uv pip install transformers peft trl accelerate' \
  --remote 'uv run python -u -c "import torch; import transformers; import peft; import trl; print(\"training packages import ok\")"' \
  --remote 'uv run python -u -m llm.leverage.train_sft_smoke --config tracks/leverage/configs/leverage-sft-qwen35-9b-long-form-constraint.toml'
```

Do not run the same command without `--dry-run` until the dry-run plan matches
the intended GPU, image, model, output paths, and cleanup policy. Prefer an RTX
4090 when the selected small student fits comfortably; use an A40 when the extra
48GB VRAM headroom is worth the slightly slower/steadier profile. Check current
RunPod pricing before launch because the v2 CLI does not accept a create-time
cost ceiling flag.

The smoke config uses one epoch, batched PyTorch training, and a 60-minute
runtime ceiling. A first RunPod setup can spend most of a 30-minute window
downloading CUDA/PyTorch wheels, so a 60-minute ceiling keeps the run bounded
while staying under the current $1 smoke cost cap on a $0.69/h RTX 4090.
Batched training is the default smoke path: the old non-batched 1,083-row
`Qwen/Qwen3.5-0.8B` smoke took 731.690 seconds for training, while the batched
path took 201.648 seconds on the same row count.

Use the official RunPod PyTorch 2.8.0 template by default:
`--template-id runpod-torch-v280`. RunPod currently maps that template to
`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, which matches the project's
locked PyTorch 2.8 runtime more closely than the newer torch291 image.

The newer CUDA 12.8 image
`runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404` has successfully loaded
`Qwen/Qwen3.5-9B`, but it is no longer the default because the project installs
its own locked `.venv` with PyTorch 2.8.0. Another RTX 4090 host also failed to
start that newer image because `nvidia-container-cli` reported
`unsatisfied condition: cuda>=12.8`. If a pod stays `RUNNING` with `pod not
ready`, inspect the RunPod console log before treating it as a trainer or model
failure. Prefer a readiness-only `nvidia-smi` probe with
`--allowed-cuda-version 12.8`, or use an earlier CUDA image when host-driver
compatibility matters more than using the newest image.

## Success Criteria

- The reviewed instruction file validates.
- The training JSONL export is regenerated.
- The training command completes within the configured smoke example limit.
- An adapter or checkpoint artifact is written.
- If eval is part of the run objective, the held-out eval command can run
  before and after training.

## Capability-Seeking Run Gate

Prioritize the `Qwen/Qwen3.5-9B` path over additional side-model smoke tests.
Do not run another capability-seeking LoRA until the data and eval scale meet
the thresholds below. A smoke proves the training path works; it does not prove
that the reviewed dataset improves the student model.

| reviewed rows | name | purpose |
| ---: | --- | --- |
| 300 | readiness run | Check that a 9B LoRA run is wired correctly and does not obviously collapse. |
| 1,000 | pilot LoRA | Look for an early improvement trend against the base 9B model. |
| 3,000+ | first serious 9B LoRA | First run large enough to treat as a real capability-seeking attempt. |
| 10,000+ | dataset v1 | Candidate scale for a serious reviewed instruction dataset. |

Before the next paid 9B LoRA run:

- Reviewed instruction rows: at least 300
- Project-judgment eval tasks: at least 100
- Holdout eval tasks: at least 30
- Label-only, duplicate, and malformed reviewed rows are excluded from the
  training export
- Holdout eval prompts are not used as teacher-generation seeds for the
  training slice

The next 9B LoRA run is useful only if:

- Overall pass rate is at least the base student pass rate.
- Project-judgment pass rate improves over the base student pass rate.
- General `leverage-smoke` capabilities do not materially regress.
- RunPod cleanup is verified after the run.

Use the reviewed-instruction mix plan for dataset distribution. Use failed
project-judgment cases only as one seed source, not as the center of the
dataset.

## Post-Training Eval

After an adapter exists, compare the base student and adapter on the same eval
tasks:

```bash
uv run python -m llm.leverage.evaluate_sft_adapter --dry-run
```

Remove `--dry-run` only in an environment that can load the base model and
adapter. The command writes predictions, detailed scores, and summary scores
under the configured output directory. Treat short evals as wiring comparisons,
not as capability claims.

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
