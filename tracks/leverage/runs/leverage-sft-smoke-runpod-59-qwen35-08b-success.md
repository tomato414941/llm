# Leverage SFT Smoke RunPod 59-Row Qwen3.5-0.8B Result

Date: 2026-04-29

## Goal

Run the 59-row LoRA/SFT smoke with the new test student
`Qwen/Qwen3.5-0.8B`.

## Transport

Status: passed

- Cloud: RunPod Secure Cloud
- Template: `runpod-torch-v280`
- Image: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
- GPU: `NVIDIA A40`
- Listed price: `$0.44/hr`
- Successful pod: `fksz6v8kaizz9z`
- CUDA smoke: passed, `cuda_device=NVIDIA A40`
- Cleanup: pod deletion succeeded.
- Final pod check: no active pods remained.

## Runner Changes Needed

Status: fixed

- Added an rsync idle timeout so stalled transfers fail before the full paid-run
  deadline.
- Moved the RunPod virtualenv to `/tmp/llm-runpod-venv` and symlinked `.venv`
  to avoid slow package installation on the network-mounted workspace.
- Set LoRA `target_modules="all-linear"` because PEFT could not infer target
  modules for `Qwen/Qwen3.5-0.8B`.

## Training

Status: passed

- Reviewed rows: 59
- Student model: `Qwen/Qwen3.5-0.8B`
- Epochs: 3
- Steps: 177
- Final loss: `0.430205`
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
lora smoke overall: 14/30, pass_rate=0.467
```

Suite-level result:

```text
base leverage-smoke: 8/12, pass_rate=0.667
lora leverage-smoke: 6/12, pass_rate=0.500
base project-judgment: 6/18, pass_rate=0.333
lora project-judgment: 8/18, pass_rate=0.444
```

## Interpretation

This is a successful smoke for the Qwen3.5-0.8B path, not proof of useful
quality improvement. The adapter tied the base model overall, improved the
project-judgment suite, and regressed the smaller leverage-smoke suite. Treat
the result as evidence that the updated test student can train and evaluate on
RunPod, not as a reason to scale training yet.
