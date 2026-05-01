# RunPod SSH Readiness Probe With Workspace Volume

Date: 2026-05-01

Goal: test whether removing the recent `--volume-size 0` pod shape fixes the
RunPod SSH readiness failure, without running any training.

## Setup

- GPU: `NVIDIA GeForce RTX 4090`
- Image: `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`
- Template id: not used
- Cloud type: Secure Cloud
- Pod: `llm-runpod-ssh-readiness-probe-20260501-140247`
- Pod id: `ontf05ge32ftfq`
- RunPod reported cost rate: `$0.69/h`
- Machine location: `RO`
- Container disk: 80GB
- Volume: 80GB mounted at `/workspace`
- Intended remote command: write `nvidia-smi` output under
  `outputs/runpod-ssh-readiness-probe/`

## Result

The probe did not reach SSH, rsync, `nvidia-smi`, CUDA, or any training code.

The pod stayed `RUNNING` and exposed `22/tcp`, but `runpodctl ssh info` never
returned a usable SSH connection within the 900-second runner wait window.

From `outputs/runpod-ssh-readiness-probe/runpod-timings.json`:

- Status: `failed`
- Total wall time: 913.458 seconds
- SSH info wait: 910.966 seconds
- SSH poll count: 78
- First poll: `RUNNING`, `22/tcp`, `pod not ready`
- Last poll: `RUNNING`, `22/tcp`, `pod not ready`

Cleanup completed for the created pod:

```text
runpodctl pod delete ontf05ge32ftfq
```

After cleanup, `runpodctl pod list -o json` and `runpodctl pod list --all -o
json` both returned `[]`.

## Interpretation

This lowers the probability that `--volume-size 0` was the primary cause. The
same readiness failure happened with a normal 80GB `/workspace` volume.

The strongest remaining signal is that RunPod reports the pod as `RUNNING` but
also reports `uptimeSeconds: 0` throughout the wait. That suggests the pod is
not reaching usable container startup from the client perspective.

The next useful diagnostic is not another blind training retry. It is either:

- create a pod and inspect RunPod console container/system logs while it is
  still alive, or
- change placement materially, for example pin a different data center or GPU
  type, then run the same minimal readiness probe.
