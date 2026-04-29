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

## Next Step

Fix the RunPod transport path before launching another paid attempt. The likely
area is pod creation/readiness/connection handling, not the reviewed dataset or
the SFT trainer.
