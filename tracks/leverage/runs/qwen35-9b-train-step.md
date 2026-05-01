# Qwen3.5-9B One-Step Train

Date: 2026-05-01

Goal: verify the smallest useful `Qwen/Qwen3.5-9B` LoRA training path before
attempting a longer baseline run: one reviewed row, batch size 1, one
forward/backward/optimizer step, adapter save, VRAM recording, and RunPod SSH
wait polling.

This was intentionally run as a one-off RunPod command. No reusable 9B
train-step framework was added.

## Setup

- GPU: `NVIDIA GeForce RTX 4090`
- Image: `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`
- Template id: not used
- Pod: `llm-leverage-qwen35-9b-train-step-20260501-123258`
- Pod id: `2hvxgnpcapp1a2`
- RunPod reported cost rate: `$0.69/h`
- Model: `Qwen/Qwen3.5-9B`
- Torch dtype: `bfloat16`
- LoRA config: `r=8`, `lora_alpha=16`, `target_modules=all-linear`
- Max sequence length: 512

Cleanup completed for the created pod. A separate pre-existing
`intrep-gpu-smoke` pod was left untouched.

## Result

The one-step train passed.

From `outputs/leverage-qwen35-9b-train-step/train-step.json`:

```json
{
  "status": "passed",
  "model": "Qwen/Qwen3.5-9B",
  "cuda_device": "NVIDIA GeForce RTX 4090",
  "loss": 3.721043109893799,
  "max_memory_allocated_gb": 18.013,
  "elapsed_seconds": 167.169
}
```

Local artifacts fetched:

- `outputs/leverage-qwen35-9b-train-step/train-step.json`
- `outputs/leverage-qwen35-9b-train-step/runpod-timings.json`
- `outputs/leverage-qwen35-9b-train-step/lora-adapter/adapter_config.json`
- `outputs/leverage-qwen35-9b-train-step/lora-adapter/adapter_model.safetensors`

Timing from `outputs/leverage-qwen35-9b-train-step/runpod-timings.json`:

- Total wall time: 260.165 seconds
- SSH info wait: 13.359 seconds
- Setup: 55.085 seconds
- CUDA smoke: 3.728 seconds
- Package install command: 2.221 seconds
- 9B load, LoRA attach, one train step, and adapter save: 175.063 seconds
- Output sync: 3.626 seconds

SSH wait poll history was recorded:

```json
[
  {
    "pod_status": "RUNNING",
    "pod_ports": "22/tcp",
    "ssh_info_error": "pod not ready",
    "ssh_info_has_connection": false
  },
  {
    "pod_status": "RUNNING",
    "pod_ports": "22/tcp",
    "ssh_info_error": null,
    "ssh_info_has_connection": true
  }
]
```

## Interpretation

This proves the minimum 9B LoRA training path fits on an RTX 4090 for a short
512-token single-row step. Peak allocated VRAM was about 18.0GB, leaving limited
but real headroom on a 24GB card for careful next-step experiments.

This is not yet proof that the full 1083-row baseline run is stable. The next
training test should increase work gradually while preserving cleanup, timing,
and VRAM measurement.
