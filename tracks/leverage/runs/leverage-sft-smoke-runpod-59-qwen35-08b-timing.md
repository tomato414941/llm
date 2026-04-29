# Leverage SFT Smoke RunPod 59-Row Qwen3.5-0.8B Timing Result

Date: 2026-04-29

## Goal

Re-run the 59-row `Qwen/Qwen3.5-0.8B` LoRA/SFT smoke with RunPod step timing
enabled.

## Transport

Status: passed

- Cloud: RunPod Secure Cloud
- Template: `runpod-torch-v280`
- Image: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
- Successful GPU: `NVIDIA GeForce RTX 5090`
- Listed price: `$0.99/hr`
- Successful pod: `tm5d7hbwc94rs1`
- Timing artifact: `outputs/leverage-sft-smoke/runpod-timings.json`
- Cleanup: pod deletion succeeded.
- Final pod check: no active pods remained.

Failed setup attempts before the successful run:

- `NVIDIA A40`: RunPod reported no deployable resources before pod creation.
- `NVIDIA RTX 4090`: RunPod reported no available instance before pod
  creation.
- `NVIDIA L4`: pod was created at `$0.39/hr`, but SSH stayed `pod not ready`;
  the pod was deleted before training.

## Timing

Total successful-run wall time: `369.119s` (`6.15m`)

Step timings:

```text
local_preflight:       0.083s
pod_create:           0.611s
ssh_info_wait:       46.257s
ssh_ready_wait:       0.646s
transport_setup:      0.616s
remote_mkdir:         0.616s
repo_sync:            1.018s
setup:               63.213s
cuda_smoke:           5.330s
remote package install command: 5.430s
package_import_smoke: 11.902s
train:              176.192s
eval:                54.278s
output_sync:          1.920s
```

Approximate successful-run GPU cost: `369.119 / 3600 * $0.99 = $0.102`.
Including the short failed L4 readiness attempt, the total spend was still under
the configured `$1.00` smoke cap.

## Training

Status: passed

- Reviewed rows: 59
- Student model: `Qwen/Qwen3.5-0.8B`
- Epochs: 3
- Steps: 177
- CUDA device: `NVIDIA GeForce RTX 5090`
- Final loss: `0.481437`
- Adapter: `outputs/leverage-sft-smoke/lora-adapter`
- Metrics: `outputs/leverage-sft-smoke/metrics.csv`
- Notes: `outputs/leverage-sft-smoke/notes.md`

## Evaluation

Status: passed

- Tasks evaluated: 30
- Predictions: `outputs/leverage-sft-smoke/post-training-predictions.jsonl`
- Scores: `outputs/leverage-sft-smoke/post-training-scores.csv`
- Summary: `outputs/leverage-sft-smoke/post-training-summary.csv`

Overall result:

```text
base overall: 14/30, pass_rate=0.467
lora smoke overall: 9/30, pass_rate=0.300
```

Suite-level result:

```text
base leverage-smoke: 8/12, pass_rate=0.667
lora leverage-smoke: 5/12, pass_rate=0.417
base project-judgment: 6/18, pass_rate=0.333
lora project-judgment: 4/18, pass_rate=0.222
```

## Interpretation

This run successfully measured the RunPod smoke path. It is not evidence of
useful model improvement. The adapter underperformed the base model on this
small eval, so the next capability-seeking LoRA should remain gated on larger
reviewed data and larger held-out eval coverage.

The main timing bottlenecks were training (`176s`), setup (`63s`), eval (`54s`),
and SSH readiness (`46s`). For 0.8B smoke runs, optimizing setup and avoiding
unready GPU inventory matter almost as much as the training step itself.
