# Qwen3.5-9B Full Reviewed-Data Attempt

Date: 2026-05-01

Goal: run the full reviewed-data `Qwen/Qwen3.5-9B` LoRA/SFT baseline through
the existing trainer instead of a long one-off command.

## Setup

- GPU: `NVIDIA GeForce RTX 4090`
- Image: `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`
- Template id: not used
- Pod: `llm-leverage-sft-qwen35-9b-20260501-125821`
- Pod id: `r3byha3plxt5vc`
- RunPod reported cost rate: `$0.69/h`
- Machine location: `US`
- Config: `tracks/leverage/configs/leverage-sft-qwen35-9b.toml`
- Intended model: `Qwen/Qwen3.5-9B`
- Intended rows: 1083

## Result

The run did not reach CUDA, dependency setup, model loading, or training.

The pod stayed `RUNNING`, but SSH never became ready within the 900-second
runner wait window.

From `outputs/leverage-sft-qwen35-9b/runpod-timings.json`:

- Status: `failed`
- Total wall time: 904.063 seconds
- SSH info wait: 900.914 seconds
- SSH poll count: 80
- First poll: `RUNNING`, `22/tcp`, `pod not ready`
- Last poll: `RUNNING`, `22/tcp`, `pod not ready`

Cleanup completed for the created pod:

```text
runpodctl pod delete r3byha3plxt5vc
```

The final RunPod pod list returned `[]`.

## Interpretation

This is another RunPod SSH readiness failure. It does not provide evidence
about whether full reviewed-data Qwen3.5-9B training succeeds or fails, because
the run never reached the remote shell.

The useful result is that the runner diagnostics captured the full readiness
failure: 80 polls over 15 minutes with stable `RUNNING / pod not ready` state.
