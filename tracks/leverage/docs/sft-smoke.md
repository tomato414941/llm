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
  - `tracks/leverage/evals/project-judgment.jsonl`
- Config: `tracks/leverage/configs/leverage-sft-smoke.toml`

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

## Method

Prefer LoRA for the first run. Full SFT is acceptable only if the implementation
is simpler in the selected training stack and remains bounded to the configured
smoke example limit.

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
  --secure-cloud \
  --gpu-type 'NVIDIA GeForce RTX 4090' \
  --image runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404 \
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

Use a RunPod PyTorch image whose CUDA requirement fits the allocated host
driver. The CUDA 12.8 image
`runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404` has successfully loaded
`Qwen/Qwen3.5-9B`, but another RTX 4090 host failed to start that same image
because `nvidia-container-cli` reported `unsatisfied condition: cuda>=12.8`.
If a pod stays `RUNNING` with `pod not ready`, inspect the RunPod console log
before treating it as a trainer or model failure. Prefer a readiness-only
`nvidia-smi` probe with `--allowed-cuda-version 12.8`, or use an earlier CUDA
image when host-driver compatibility matters more than using the newest image.

## Success Criteria

- The reviewed instruction file validates.
- The training JSONL export is regenerated.
- The training command completes within the configured smoke example limit.
- An adapter or checkpoint artifact is written.
- The held-out eval command can run before and after training.

## Post-Training Eval

After a smoke adapter exists under `outputs/leverage-sft-smoke/lora-adapter/`, compare
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
