# Qwen3.5-9B Load Timing 200-Row Baseline

Date: 2026-05-01

Goal: measure the cold-start cost before deciding whether a persistent RunPod
network volume is worth adding for Hugging Face model cache reuse.

## Setup

- Config: `tracks/leverage/configs/leverage-sft-qwen35-9b-batch2-200.toml`
- Train export: `tracks/leverage/sft/bootstrap-200.train.jsonl`
- Rows: 200
- Model: `Qwen/Qwen3.5-9B`
- GPU: `NVIDIA GeForce RTX 4090`
- Image: `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`
- CUDA filter: `12.8`, `12.9`, `13.0`
- Batch size: 2
- Gradient accumulation steps: 4
- Network volume: not used
- Cleanup: completed automatically

## Result

From `outputs/leverage-sft-qwen35-9b-batch2-200/metrics.csv`:

- Total seconds inside trainer: 187.327
- Pre-train seconds: 27.789
- Tokenizer load seconds: 3.177
- Model load seconds: 17.472
- Adapter setup seconds: 4.459
- CUDA transfer seconds: 2.618
- Render seconds: 0.059
- Train seconds: 159.538
- Tokens/sec: 95.049
- Peak VRAM: 19.454GB
- Final loss: 0.643455

From `outputs/leverage-sft-qwen35-9b-batch2-200/runpod-timings.json`:

- Total wall time: 321.027 seconds
- Pod create: 1.677 seconds
- SSH info wait: 24.238 seconds
- Setup: 50.880 seconds
- CUDA smoke: 8.841 seconds
- Dependency command: 5.129 seconds
- Train command: 214.482 seconds
- Output sync: 6.934 seconds

Approximate cost at `$0.69/h`: `$0.062`.

## Interpretation

The immediate Hugging Face cache reuse upside is modest for this 200-row
measurement. The measured model/tokenizer load path was about 20.6 seconds, and
the full pre-train path was 27.8 seconds. A persistent network volume may still
help longer workflows and repeated experiments, but this measurement does not
justify adding it as the next highest-priority change by itself.

The larger visible cold-start cost is environment setup, especially `uv sync`
and package installation. If startup cost becomes the priority, the simpler next
measurement is to compare a warmed Python environment or image path before
adding network-volume-specific workflow.

`gpu_utilization_percent` was still blank in progress logging, so
`torch.cuda.utilization()` is not useful enough in this RunPod image. Use
`nvidia-smi` sampling if GPU utilization becomes a decision metric.
