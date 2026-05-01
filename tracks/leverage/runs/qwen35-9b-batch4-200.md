# Qwen3.5-9B Batch Size 4 200-Row Measurement

Date: 2026-05-01

Goal: check whether `Qwen/Qwen3.5-9B` can train with `batch_size=4` on 200
reviewed rows and compare throughput against the prior `batch_size=2`
measurement.

## Setup

- Config: `tracks/leverage/configs/leverage-sft-qwen35-9b-batch4-200.toml`
- Train export: `tracks/leverage/sft/bootstrap-200.train.jsonl`
- Rows: 200
- Model: `Qwen/Qwen3.5-9B`
- GPU: `NVIDIA GeForce RTX 4090`
- Image: `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`
- CUDA filter: `12.8`, `12.9`, `13.0`
- Batch size: 4
- Max length: 512
- Gradient checkpointing: `true`
- Gradient accumulation steps: 4
- Log every steps: 10

## Result

- Pod: `llm-leverage-sft-qwen35-9b-batch4-200-20260501-181028`
- Pod id: `a9qiyrzdfd2p2v`
- Location: `US`
- RunPod reported cost rate: `$0.69/h`
- Total wall time: 255.260 seconds
- Approximate cost: `$0.049`
- Cleanup: completed automatically

From `outputs/leverage-sft-qwen35-9b-batch4-200/runpod-timings.json`:

- Status: `passed`
- Pod create: 1.083 seconds
- SSH info wait: 46.515 seconds
- Setup: 43.967 seconds
- CUDA smoke: 9.544 seconds
- Dependency command: 4.077 seconds
- Train command: 141.941 seconds
- Output sync: 2.874 seconds

From `outputs/leverage-sft-qwen35-9b-batch4-200/metrics.csv`:

- Rows: 200
- Steps: 50
- Optimizer steps: 13
- Tokens: 15164
- Total trainer seconds: 112.341
- Pre-train seconds: 42.651
- Tokenizer load seconds: 2.775
- Model load seconds: 25.045
- CUDA transfer seconds: 13.050
- Train seconds: 69.690
- Tokens/sec: 217.594
- Peak VRAM: 21.886GB
- Final loss: 0.643253
- Status: `completed`

Progress logging wrote:

```text
outputs/leverage-sft-qwen35-9b-batch4-200/logs/progress.csv
```

The last progress row was:

```text
step=50 optimizer_steps=12 tokens=15164 loss=0.643253 tokens_per_second=220.332 peak_vram_gb=21.886 gpu_utilization_percent=
```

## Comparison

Against the prior 200-row `batch_size=2` measurement:

- Train seconds improved from 159.538 to 69.690.
- Tokens/sec improved from 95.049 to 217.594.
- Peak VRAM increased from 19.454GB to 21.886GB.
- Wall time improved from 321.027 to 255.260 seconds despite slower SSH readiness.

## Interpretation

`batch_size=4` is the better current setting for Qwen3.5-9B LoRA/SFT on an RTX
4090 for this 200-row shape. It roughly doubles training throughput versus
`batch_size=2`, while staying under 24GB VRAM. The remaining headroom is not
large, so a larger batch should be treated as an experiment rather than the new
default.

`gpu_utilization_percent` remained blank, so the current
`torch.cuda.utilization()` path is still not a reliable utilization signal in
this RunPod image.
