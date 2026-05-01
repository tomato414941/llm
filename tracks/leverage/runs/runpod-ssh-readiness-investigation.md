# RunPod SSH Readiness Investigation

Date: 2026-05-01

Goal: identify the simplest official-aligned next step after repeated RunPod
pods stayed `RUNNING` while `runpodctl ssh info` returned `pod not ready`.

## Evidence

Successful recent 9B runs:

- `qwen35-9b-load-smoke-image103.md`: direct official image
  `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`, SSH ready in 35.247s.
- `qwen35-9b-train-step.md`: same direct image, SSH ready in 13.359s.

Failed recent 9B full attempts:

- `qwen35-9b-full-attempt.md`: same direct image, two fresh RTX 4090 pods,
  both stayed `RUNNING` with `22/tcp` but `pod not ready` for 900s.
- `qwen35-9b-full-template-attempt.md`: official template
  `runpod-torch-v280`, resolved image
  `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, also stayed `RUNNING`
  with `22/tcp` but `pod not ready` for 900s.

Local RunPod client checks:

- `runpodctl doctor`: API connectivity passed.
- `runpodctl doctor`: local SSH key exists and is synced to RunPod.
- `runpodctl ssh list-keys`: both account keys are visible.
- `runpodctl pod list -o json`: no active pods after cleanup.

## Official Behavior To Preserve

RunPod docs distinguish two SSH paths:

- Basic SSH is available through RunPod's proxy, but does not support SCP/SFTP.
- Full SSH for SCP/rsync requires public-IP SSH over exposed TCP 22 and a
  running ssh daemon. RunPod says official PyTorch templates often have this
  configured already.

The current runner needs full SSH because it uses `rsync` for repository and
artifact transfer.

Official docs also say pod logs are available from the RunPod console. The CLI
and API do not appear to provide an equivalent pod-log retrieval path today, so
if a pod reaches `RUNNING` but never becomes SSH-ready, console logs are the
official diagnostic path.

## Leading Hypotheses

1. RunPod provisioning/readiness issue on the allocated machine or data center.
   This is supported by repeated `RUNNING` + `22/tcp` + `pod not ready` states.

2. The latest full-run commands changed pod shape enough to affect startup.
   The suspicious difference is `--volume-size 0` while still using
   `/workspace`. Official PyTorch templates advertise a positive `/workspace`
   volume, and the generic project docs use the runner defaults rather than
   forcing zero volume.

3. Template indirection is not the only cause. The official template attempt
   failed the same way as the direct-image attempts.

## Recommended Next Probe

Run one minimal paid readiness probe, not training:

- Use the direct current image:
  `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`.
- Use RTX 4090, but pin a specific data center with available stock if possible
  instead of leaving placement entirely implicit.
- Do not use `--volume-size 0`; use the runner default or the official template
  volume shape so `/workspace` is ordinary.
- Keep the remote command tiny, for example `true` or `nvidia-smi`.
- Stop at SSH readiness and CUDA smoke. Do not start the full 1083-row train
  until readiness is stable again.

If that probe also stays `RUNNING` + `pod not ready`, the next useful action is
RunPod console inspection of container/system logs while the pod is alive, not
another blind CLI retry.

## Sources

- RunPod SSH docs:
  https://docs.runpod.io/pods/configuration/use-ssh
- RunPod pod management and logs:
  https://docs.runpod.io/pods/manage-pods
- RunPod file transfer docs:
  https://docs.runpod.io/pods/storage/transfer-files
- RunPod CLI pod reference:
  https://docs.runpod.io/runpodctl/reference/runpodctl-pod
