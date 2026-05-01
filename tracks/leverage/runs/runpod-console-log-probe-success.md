# RunPod Console Log Probe Follow-Up

Date: 2026-05-01

Goal: create a minimal pod that could be inspected through the RunPod console
if SSH readiness failed again.

## Setup

- GPU: `NVIDIA GeForce RTX 4090`
- Image: `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`
- Template id: not used
- Cloud type: Secure Cloud
- Pod: `llm-runpod-console-log-probe-20260501-142538`
- Pod id: `y49h9brl01u7se`
- RunPod reported cost rate: `$0.69/h`
- Machine location: `US`
- Container disk: 80GB
- Volume: 80GB mounted at `/workspace`
- Remote command: write `nvidia-smi` output under
  `outputs/runpod-console-log-probe/`

## Result

The probe passed before console log inspection was needed.

From `outputs/runpod-console-log-probe/runpod-timings.json`:

- Status: `passed`
- Total wall time: 46.463 seconds
- SSH info wait: 35.356 seconds
- SSH poll count: 4
- First poll: `RUNNING`, `22/tcp`, `pod not ready`
- Last poll: `RUNNING`, `22/tcp`, SSH connection present
- SSH ready wait: 1.084 seconds
- Output sync: 1.620 seconds

The `nvidia-smi` output confirmed:

- GPU: `NVIDIA GeForce RTX 4090`
- Driver: `575.57.08`
- CUDA: `12.9`
- VRAM: `24564MiB`

Cleanup completed for the created pod:

```text
runpodctl pod delete y49h9brl01u7se
runpodctl pod list -o json
[]
```

## Interpretation

This confirms the earlier failures were not caused by model loading. This probe
did not load a model at all; it only waited for SSH, ran `nvidia-smi`, synced
the output, and deleted the pod.

The result also shows the RunPod path is intermittent rather than permanently
broken. A US RTX 4090 pod with the same current image and normal `/workspace`
volume became SSH-ready in about 35 seconds, while earlier RO/US attempts stayed
`RUNNING` with `pod not ready` for 900 seconds.

The next training retry should keep the same current image and normal
`/workspace` volume, but avoid interpreting a 900-second SSH wait failure as a
trainer, CUDA, or model-load failure.
