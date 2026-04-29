# Leverage SFT Smoke RunPod 59-Row Secure Cloud Result

Date: 2026-04-29

## Goal

Run the 59-row LoRA/SFT smoke on RunPod after Community Cloud SSH/readiness
attempts failed before training.

## Transport

Status: passed

- Cloud: RunPod Secure Cloud
- Template: `runpod-torch-v280`
- Image: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
- GPU: `NVIDIA A40`
- Listed price: `$0.44/hr`
- Pod: `13xt2x6enusyf3`
- SSH: passed
- CUDA smoke: passed, `cuda_device=NVIDIA A40`
- Cleanup: `runpodctl pod delete 13xt2x6enusyf3` succeeded.
- Final pod check: no active pods remained.

The previous blocker was not LoRA/SFT training. Secure Cloud with the official
template reached SSH, CUDA, training, evaluation, output sync, and cleanup.

## Training

Status: passed

- Reviewed rows: 59
- Student model: `Qwen/Qwen3-0.6B`
- Epochs: 3
- Steps: 177
- Final loss: `1.960245`
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
base overall: 11/30, pass_rate=0.367
lora smoke overall: 12/30, pass_rate=0.400
```

Suite-level result:

```text
base leverage-smoke: 8/12, pass_rate=0.667
lora leverage-smoke: 9/12, pass_rate=0.750
base project-judgment: 3/18, pass_rate=0.167
lora project-judgment: 3/18, pass_rate=0.167
```

## Interpretation

This is a wiring success, not proof of useful model improvement. The LoRA smoke
slightly improved the small overall eval, mostly by improving the
`leverage-smoke` suite. It did not improve `project-judgment`.

The next useful step is not to train longer immediately. First, inspect
per-task changes and decide whether the eval suite and teacher data distribution
are measuring the intended general capabilities.
