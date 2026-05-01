# RunPod CUDA Filter Probe

Date: 2026-05-01

Goal: verify that the runner can request RunPod hosts compatible with the
CUDA 12.8 image by using the REST API `allowedCudaVersions` filter.

## Setup

- GPU: `NVIDIA GeForce RTX 4090`
- Image: `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`
- CUDA filter: `12.8`, `12.9`, `13.0`
- Cloud type: Secure Cloud
- Container disk: 80GB
- Volume: 80GB mounted at `/workspace`
- Remote command: write `nvidia-smi` output under
  `outputs/runpod-cuda-filter-probe/`

## Result

The first REST-based create attempt succeeded, but the local runner crashed
while parsing the create result because `curl` output was not captured yet.
The created pod was manually cleaned up:

```text
runpodctl pod delete twv35bq4ild4bg
runpodctl pod list -o json
[]
```

After fixing command capture for `curl`, the CUDA-filtered probe passed.

- Pod: `llm-runpod-cuda-filter-probe-20260501-161637`
- Pod id: `b3xwh4z46vzpar`
- RunPod reported cost rate: `$0.69/h`
- Machine location: `CZ`
- Memory: 125GB
- vCPU: 32
- Status: `passed`
- Total wall time: 84.473 seconds
- Pod create: 1.196 seconds
- SSH info wait: 70.436 seconds
- SSH poll count: 7
- SSH ready wait: 1.391 seconds
- Output sync: 2.122 seconds

The `nvidia-smi` output confirmed:

- GPU: `NVIDIA GeForce RTX 4090`
- Driver: `580.126.20`
- CUDA: `13.0`
- VRAM: `24564MiB`

Cleanup completed automatically for the created pod:

```text
runpodctl pod list -o json
[]
```

## Interpretation

The CUDA filter did what we need for the current image path: the allocated host
reported CUDA 13.0 and satisfied the CUDA 12.8+ container requirement.

This turns the earlier `nvidia-container-cli: requirement error:
unsatisfied condition: cuda>=12.8` failure from an opaque SSH readiness timeout
into an avoidable placement issue. Future runs that use
`runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404` should pass:

```text
--allowed-cuda-version 12.8 --allowed-cuda-version 12.9 --allowed-cuda-version 13.0
```

If a future run still fails before SSH with these filters, treat it as a new
RunPod placement or container-start issue, not as evidence about the model or
trainer.
