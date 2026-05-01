# Qwen3.5-9B Mini32 Train

Date: 2026-05-01

Goal: move one step beyond the one-row train check by running a small but real
`Qwen/Qwen3.5-9B` LoRA training pass on 32 reviewed rows while preserving VRAM,
timing, adapter save, and RunPod SSH wait diagnostics.

This was intentionally run as a one-off RunPod command. No reusable 9B mini-run
framework was added.

## Setup

- GPU: `NVIDIA GeForce RTX 4090`
- Image: `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`
- Template id: not used
- Pod: `llm-leverage-qwen35-9b-mini32-20260501-124005`
- Pod id: `9tle3showwnjde`
- RunPod reported cost rate: `$0.69/h`
- Machine location: `RO`
- Model: `Qwen/Qwen3.5-9B`
- Torch dtype: `bfloat16`
- LoRA config: `r=8`, `lora_alpha=16`, `target_modules=all-linear`
- Examples: 32
- Max sequence length: 512
- Batch size: 1

Cleanup completed for the created pod.

## Result

The 32-row mini train passed.

From `outputs/leverage-qwen35-9b-mini32/mini32.json`:

```json
{
  "status": "passed",
  "model": "Qwen/Qwen3.5-9B",
  "examples": 32,
  "max_length": 512,
  "batch_size": 1,
  "cuda_device": "NVIDIA GeForce RTX 4090",
  "first_loss": 3.721043109893799,
  "final_loss": 3.3858039379119873,
  "mean_loss": 3.559750311076641,
  "max_memory_allocated_gb": 18.139,
  "train_seconds": 14.105,
  "elapsed_seconds": 120.762
}
```

Local artifacts fetched:

- `outputs/leverage-qwen35-9b-mini32/mini32.json`
- `outputs/leverage-qwen35-9b-mini32/runpod-timings.json`
- `outputs/leverage-qwen35-9b-mini32/lora-adapter/adapter_config.json`
- `outputs/leverage-qwen35-9b-mini32/lora-adapter/adapter_model.safetensors`

Timing from `outputs/leverage-qwen35-9b-mini32/runpod-timings.json`:

- Total wall time: 459.818 seconds
- SSH info wait: 258.081 seconds
- Setup: 39.144 seconds
- CUDA smoke: 10.550 seconds
- Package install command: 3.327 seconds
- 9B load, LoRA attach, 32 train steps, and adapter save: 131.903 seconds
- Output sync: 7.538 seconds

SSH wait poll history showed a long readiness delay:

- `pod not ready` from `2026-05-01T12:40:17Z` through
  `2026-05-01T12:44:13Z`
- SSH info became ready at `2026-05-01T12:44:24Z`

Per-step losses were noisy, as expected for single-row updates. The first loss
was `3.7210`, final loss was `3.3858`, and mean loss was `3.5598`.

## Interpretation

This confirms that 9B LoRA training is not merely loadable on an RTX 4090; a
small 32-row training pass with adapter save also completes with peak allocated
VRAM around 18.1GB.

The main operational issue was not CUDA or VRAM. It was RunPod SSH readiness:
this RO allocation took about 4.3 minutes to expose SSH despite the pod staying
`RUNNING`. The newly recorded polling diagnostics captured that delay.

The next useful step is a larger mini run, such as 128 rows, before considering
the full 1083-row baseline.
