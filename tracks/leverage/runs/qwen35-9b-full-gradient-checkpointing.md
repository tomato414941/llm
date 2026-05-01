# Qwen3.5-9B Full Reviewed-Data Run With Gradient Checkpointing

Date: 2026-05-01

Goal: run the full reviewed-instruction LoRA/SFT baseline on `Qwen/Qwen3.5-9B`
after adding the RunPod CUDA filter.

## First Attempt

The first CUDA-filtered full run reached SSH, passed CUDA smoke, downloaded the
model, and loaded weights, but failed during training with CUDA OOM.

- Pod id: `koik93mdngaoi9`
- GPU: `NVIDIA GeForce RTX 4090`
- Image: `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`
- CUDA filter: `12.8`, `12.9`, `13.0`
- Failure: `torch.OutOfMemoryError`
- Cleanup: completed automatically

Interpretation: CUDA placement and model loading were no longer blockers. The
trainer needed a standard activation-memory reduction path for 9B LoRA on this
GPU class.

## Change

Enabled `method.gradient_checkpointing = true` for the Qwen3.5-9B config and
made the trainer call:

- `model.config.use_cache = False`
- `model.gradient_checkpointing_enable()`
- `model.enable_input_require_grads()` when available

This keeps the change scoped to the 9B baseline instead of lowering sequence
length or changing the dataset.

## Successful Run

- Pod: `llm-leverage-sft-qwen35-9b-full-gc-20260501-162820`
- Pod id: `doc7mh0z7fjrg3`
- RunPod reported cost rate: `$0.69/h`
- Approximate billable wall time: 1859.692 seconds, about 0.517 hours
- Approximate cost: `$0.36`
- Machine location: `CZ`
- GPU: `NVIDIA GeForce RTX 4090`
- Image: `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`
- Container disk: 120GB
- Volume: 80GB mounted at `/workspace`
- CUDA smoke: passed with `torch=2.8.0+cu128`
- Output sync: completed
- Cleanup: completed automatically

From `outputs/leverage-sft-qwen35-9b/runpod-timings.json`:

- Status: `passed`
- Total wall time: 1859.692 seconds
- Pod create: 1.174 seconds
- SSH info wait: 37.604 seconds
- Setup: 45.411 seconds
- CUDA smoke: 8.693 seconds
- Dependency command: 4.882 seconds
- Train step: 1745.938 seconds
- Output sync: 7.087 seconds

From `outputs/leverage-sft-qwen35-9b/metrics.csv`:

- Rows: 1083
- Steps: 1083
- Epochs: 1
- Batch size: 1
- Max length: 512
- Gradient checkpointing: `True`
- Tokens: 104115
- Train seconds: 1697.385
- Tokens/sec: 61.338
- Peak VRAM: 18.468GB
- Final loss: 0.099927
- Status: `completed`

Adapter artifacts were written under:

```text
outputs/leverage-sft-qwen35-9b/lora-adapter/
```

The adapter output is intentionally not committed because `outputs/` is a
generated artifact directory.

## Interpretation

The Qwen3.5-9B reviewed-data baseline is now trainable on the current RunPod
path with the CUDA filter and gradient checkpointing.

The run also exposed a real KISS issue: the trainer prints no per-step progress
and writes metrics only after completion. That made a successful 29-minute
training run look stalled for most of its runtime. The next improvement should
be minimal progress logging or periodic metrics, not another abstraction layer.
