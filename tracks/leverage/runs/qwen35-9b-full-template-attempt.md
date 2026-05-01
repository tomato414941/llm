# Qwen3.5-9B Full Reviewed-Data Official Template Attempt

Date: 2026-05-01

Goal: test whether the RunPod official PyTorch template resolves the repeated
SSH readiness failures seen with direct image selection, before retrying the
full `Qwen/Qwen3.5-9B` LoRA/SFT baseline.

## Setup

- GPU: `NVIDIA GeForce RTX 4090`
- Template id: `runpod-torch-v280`
- Resolved image: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
- Pod: `llm-leverage-sft-qwen35-9b-template-20260501-133801`
- Pod id: `oomr9xuohokowb`
- RunPod reported cost rate: `$0.69/h`
- Machine location: `US`
- Config: `tracks/leverage/configs/leverage-sft-qwen35-9b.toml`
- Intended model: `Qwen/Qwen3.5-9B`
- Intended rows: 1083

## Result

The run did not reach CUDA, dependency setup, model loading, or training.

The pod stayed `RUNNING` and exposed `22/tcp`, but `runpodctl ssh info` never
returned a usable SSH connection within the 900-second runner wait window.

From `outputs/leverage-sft-qwen35-9b-template/runpod-timings.json`:

- Status: `failed`
- Total wall time: 905.336 seconds
- SSH info wait: 902.836 seconds
- SSH poll count: 81
- First poll: `RUNNING`, `22/tcp`, `pod not ready`
- Last poll: `RUNNING`, `22/tcp`, `pod not ready`

Cleanup completed for the created pod:

```text
runpodctl pod delete oomr9xuohokowb
runpodctl pod list -o json
[]
```

## Interpretation

Using the RunPod official PyTorch template did not resolve the SSH readiness
failure. This still does not say anything about Qwen3.5-9B training quality or
capacity, because execution never reached the remote shell.

The current evidence points to a RunPod provisioning/readiness path issue for
these fresh RTX 4090 pods, not a CUDA, model, or trainer failure. More paid
retries with the same GPU/cloud shape are low value unless a RunPod variable is
changed or the pod is inspected through the RunPod console/web terminal/logs.
