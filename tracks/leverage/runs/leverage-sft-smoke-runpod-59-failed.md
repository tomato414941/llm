# Leverage SFT Smoke RunPod 59-Row Attempt

Date: 2026-04-29

## Goal

Run the LoRA/SFT smoke again after growing the reviewed instruction dataset from
29 rows to 59 rows.

## Local Preflight

Status: passed

```text
validated reviewed instructions: tracks/leverage/datasets/reviewed-instructions/bootstrap.jsonl
exported training rows: 59 -> tracks/leverage/sft/bootstrap.train.jsonl
checked eval tasks: 2
checked local output root: outputs/leverage-sft-smoke
runpod.required=false
```

## RunPod Dry Run

Status: passed

The dry run showed the intended sequence:

- local SFT smoke preflight
- RunPod pod creation
- repo and leverage data sync
- CUDA smoke
- training package import smoke
- LoRA/SFT smoke training command
- post-training adapter eval command
- output sync
- pod cleanup

## Real RunPod Attempt

Status: failed before training

- Pod created: `xotxutlt4bm3av`
- Pod name: `llm-leverage-sft-smoke-20260429-122200`
- GPU: `NVIDIA GeForce RTX 3090`
- Listed price: `$0.220/hr`
- Failure: the pod stayed `RUNNING`, but `runpodctl get pod --allfields`
  never reported an SSH port.
- Timeout: `pod did not expose SSH within 900 seconds`
- Cleanup: `runpodctl remove pod xotxutlt4bm3av` succeeded.
- Final pod check: no active pods remained.

## Interpretation

This did not test model training quality. The failure happened before SSH,
repo sync, CUDA smoke, dependency import, training, or adapter evaluation.

## Follow-Up Transport Check

Tried a minimal `runpodctl exec python` CUDA check on a separate RTX 3090 pod:

- Pod created: `9bnx6ddd9d5wlv`
- Pod name: `llm-exec-cuda-check`
- Failure: `runpodctl exec python` stayed at `Waiting for Pod to come online...`
  and did not execute the CUDA check.
- Cleanup: `runpodctl remove pod 9bnx6ddd9d5wlv` succeeded.
- Final pod check: no active pods remained.

This suggests the immediate blocker is RunPod pod readiness/connectivity, not
only the custom SSH bootstrap path.

## CLI and Port-Exposure Findings

The local runner uses `/home/dev/bin/runpodctl v1.14.3` and the deprecated
command family:

```text
runpodctl create pod ...
runpodctl get pod --allfields
runpodctl remove pod ...
```

The current RunPod CLI release checked during triage was `v2.1.9`. Its help
marks the old `get`, `create`, `remove`, and `exec` command family as
deprecated and exposes the newer command family:

```text
runpodctl pod create ...
runpodctl pod list ...
runpodctl pod get ...
runpodctl ssh info ...
```

The current RunPod docs also describe TCP port access as public-IP based. The
new `pod create` command has a `--public-ip` flag for Community Cloud Pods; the
old `create pod` command used by the runner does not. This is a likely reason
the Community Cloud pod reached `RUNNING` while its `PORTS` field stayed empty.

An attempted Secure Cloud 3090 port check could not start because no matching
Secure Cloud instance was available at that moment, so it did not create a pod
or test the Secure Cloud path.

## v2 Public-IP Attempt

After moving to `runpodctl v2.1.9`, a Community Cloud pod created with
`--public-ip --ssh` exposed SSH info successfully:

- Pod created: `rwqns5j569e27s`
- Public SSH target: `root@80.15.7.37:45471`
- Failure: SSH authentication returned `Permission denied (publickey,password)`.
- Cleanup: `runpodctl pod delete rwqns5j569e27s` succeeded.
- Final pod check: no active pods remained.

The RunPod account key fingerprint and the local private key fingerprint both
matched, so the next candidate fix was to use RunPod's official PyTorch
template instead of direct image creation.

## v2 Official Template Attempts

Using the official `runpod-torch-v280` template changed the failure mode but
still did not reach training.

First template attempt:

- Pod created: `nonr7gfm6j6u7o`
- Public SSH target was exposed.
- Failure: SSH authentication returned `Permission denied (publickey,password)`.
- Cleanup: `runpodctl pod delete nonr7gfm6j6u7o` succeeded.
- Final pod check: no active pods remained.

Second template attempt after adding an ED25519 RunPod SSH key and making it the
local default key:

- Pod created: `6dijw2cx1eyex0`
- Failure: the pod stayed `RUNNING`, but `runpodctl ssh info` returned
  `pod not ready` until the 900-second timeout.
- Cleanup: `runpodctl pod delete 6dijw2cx1eyex0` succeeded.
- Final pod check: no active pods remained.

The trainer, CUDA import smoke, dependency import smoke, and adapter evaluation
commands still did not run. The remaining blocker is RunPod SSH/readiness
transport, not the LoRA/SFT code path.

## Next Step

Do not keep retrying the same Community Cloud SSH path. The next useful attempt
should change one transport variable, such as trying a Secure Cloud GPU with the
same official template, or replacing SSH orchestration with a RunPod job-style
entrypoint that does not depend on inbound SSH readiness.
