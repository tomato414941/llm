# Qwen3.5-9B Full Reviewed-Data Readiness Retry

Date: 2026-05-01

Goal: retry the full reviewed-data `Qwen/Qwen3.5-9B` LoRA/SFT baseline after
adding RunPod readiness observability and after a minimal `nvidia-smi` probe
had succeeded.

## Setup

- GPU: `NVIDIA GeForce RTX 4090`
- Image: `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`
- Template id: not used
- Cloud type: Secure Cloud
- Pod: `llm-leverage-sft-qwen35-9b-full-20260501-155137`
- Pod id: `2xlr2t9hvsyb47`
- RunPod reported cost rate: `$0.69/h`
- Machine location: `US`
- Container disk: 120GB
- Volume: 80GB mounted at `/workspace`
- Config: `tracks/leverage/configs/leverage-sft-qwen35-9b.toml`
- Intended model: `Qwen/Qwen3.5-9B`
- Intended rows: 1083

## Result

The run did not reach SSH, rsync, CUDA, dependency setup, model loading, or
training.

The pod stayed `RUNNING` and exposed `22/tcp`, but `runpodctl ssh info` never
returned a usable SSH connection within the 900-second runner wait window.

From `outputs/leverage-sft-qwen35-9b/runpod-timings.json`:

- Status: `failed`
- Total wall time: 906.113 seconds
- SSH info wait: 903.117 seconds
- SSH poll count: 81
- First poll: `RUNNING`, `22/tcp`, `pod not ready`
- Last poll: `RUNNING`, `22/tcp`, `pod not ready`
- Poll metadata: memory `62GB`, vCPU `16`
- Create metadata: location `US`, GPU `RTX 4090`, volume `80GB`

Cleanup completed for the created pod:

```text
runpodctl pod delete 2xlr2t9hvsyb47
runpodctl pod list -o json
[]
```

## Interpretation

This is a RunPod SSH readiness failure, not a Qwen3.5-9B model-load or trainer
failure. The run never reached the remote shell.

The useful new signal is that the created pod metadata was captured in the
timing file. This failed on a US RTX 4090 pod with the current image and a
normal `/workspace` volume, while a prior minimal US RTX 4090 probe with the
same image and volume became SSH-ready in about 35 seconds.

The current evidence supports intermittent RunPod provisioning/readiness
behavior. More full-training retries are low value until the readiness layer is
handled separately, for example by running a readiness-only probe first and
launching training only after a pod has demonstrated SSH availability.
