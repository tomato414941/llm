# Qwen3.5-9B Load Smoke With Official Image

Date: 2026-05-01

Goal: check whether `Qwen/Qwen3.5-9B` can load on a RunPod RTX 4090, attach
LoRA, run one forward pass, and report VRAM before attempting any 9B training.

This was intentionally run as a one-off RunPod command. No reusable load-only
smoke framework was added.

## Setup

- GPU: `NVIDIA GeForce RTX 4090`
- Image: `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`
- Template id: not used
- Pod: `llm-leverage-qwen35-9b-load-image103-20260501-085440`
- Pod id: `p52ga2vqh00u66`
- RunPod reported cost rate: `$0.69/h`
- Model: `Qwen/Qwen3.5-9B`
- Torch dtype: `bfloat16`
- LoRA config: `r=8`, `lora_alpha=16`, `target_modules=all-linear`

Cleanup completed:

```text
runpodctl pod delete p52ga2vqh00u66
runpodctl pod list -o json
[]
```

## Result

The smoke passed.

From `outputs/leverage-qwen35-9b-load-image103/load-smoke.json`:

```json
{
  "cuda_device": "NVIDIA GeForce RTX 4090",
  "logits_shape": [
    1,
    18,
    248320
  ],
  "max_memory_allocated_gb": 16.811,
  "memory_allocated_gb": 16.8,
  "model": "Qwen/Qwen3.5-9B",
  "status": "passed"
}
```

Timing from `outputs/leverage-qwen35-9b-load-image103/runpod-timings.json`:

- Total wall time: 317.170 seconds
- SSH info wait: 35.247 seconds
- Setup: 55.349 seconds
- CUDA smoke: 11.402 seconds
- 9B load, LoRA attach, and forward command: 200.818 seconds
- Output sync: 1.619 seconds

## Interpretation

The earlier `runpod-torch-v280` attempt did not reach model loading because the
pod stayed `RUNNING` while `runpodctl ssh info` kept returning `pod not ready`.
This run used the newer official image directly and reached SSH readiness,
CUDA, model load, LoRA attach, and one forward pass.

This is not proof that 9B LoRA training will fit comfortably. It does show that
the base 9B bf16 model plus LoRA adapter and a short one-sample forward fit on
an RTX 4090 with about 16.8GB allocated. Training will require more memory for
gradients, optimizer state, and longer/batched sequences, so the next 9B test
should be a very small LoRA train step, not a full run.
