# Qwen3.5-9B Baseline 1216-Row Adapter Eval

Date: 2026-05-09

## Goal

Evaluate the adapter produced by `qwen35-9b-baseline-1216.md` against the
configured held-out leverage eval tasks, comparing base model and adapter under
the same scoring harness.

## Config

- Config: `tracks/leverage/configs/leverage-sft-qwen35-9b.toml`
- Base model: `Qwen/Qwen3.5-9B`
- Adapter: `outputs/leverage-sft-qwen35-9b/lora-adapter`
- Eval tasks:
  - `tracks/leverage/evals/leverage-smoke.jsonl`
  - `tracks/leverage/evals/project-judgment.jsonl`
- Task count: 30
- Base label: `qwen35-9b-base-20260509`
- Adapter label: `qwen35-9b-lora-1216-20260509`

## RunPod

- Pod: `llm-leverage-eval-qwen35-9b-baseline-1216-20260509-134821`
- Pod id: `ukthiuloh0pwyr`
- Location: `NO`
- GPU: `NVIDIA GeForce RTX 4090`
- Image: `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`
- CUDA filter: `12.8`, `12.9`, `13.0`
- Cost rate: `$0.69/h`
- Total wall time: 640.257 seconds
- Approximate cost: `640.257 / 3600 * $0.69 = about $0.12`
- Cleanup: completed automatically
- Final pod list: `[]`

## Timing

From `outputs/leverage-sft-qwen35-9b/runpod-timings.json`:

- Status: `passed`
- Pod create: 1.289 seconds
- SSH info wait: 58.550 seconds
- SSH ready wait: 2.116 seconds
- Setup: 58.100 seconds
- CUDA smoke: 13.958 seconds
- Training package install: 5.835 seconds
- Inference package import smoke: 43.508 seconds
- Eval command: 438.917 seconds
- Output sync: 2.171 seconds

## Results

From `outputs/leverage-sft-qwen35-9b/post-training-summary.csv`:

- Base overall: 15/30, pass rate 0.500
- Adapter overall: 18/30, pass rate 0.600
- Base `leverage-smoke`: 9/12, pass rate 0.750
- Adapter `leverage-smoke`: 9/12, pass rate 0.750
- Base `project-judgment`: 6/18, pass rate 0.333
- Adapter `project-judgment`: 9/18, pass rate 0.500

Capability-level changes:

- `leverage-smoke` coding: base 2/2, adapter 1/2
- `leverage-smoke` summarization/transformation: base 1/2, adapter 2/2
- `project-judgment` reasoning: base 5/15, adapter 8/15
- `project-judgment` coding: base 1/3, adapter 1/3

## Interpretation

This is a small positive held-out result for the 1,216-row adapter. The adapter
improved overall score by 3 tasks, driven by `project-judgment` reasoning and
one `leverage-smoke` summarization task.

The result is not uniformly positive. The adapter lost one `leverage-smoke`
coding task while gaining one summarization task. The eval is also a strict
string/regex harness, so some failures are scoring-contract failures rather
than broad semantic failures.

The useful next conclusion is narrow: the new 1,216-row adapter did not regress
the overall held-out eval and improved the project-judgment slice. It still
needs external benchmark evaluation before making a stronger capability claim.
