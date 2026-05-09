# Qwen3.5-9B Baseline 1216-Row LoRA Run

Date: 2026-05-09

## Goal

Train the current reviewed-instruction Qwen3.5-9B LoRA baseline after adding
`surface-constraint-batch-001`.

This is a 1-epoch baseline run. Early stopping is intentionally not enabled;
NaN or infinity loss remains the stop condition.

## Config

- Config: `tracks/leverage/configs/leverage-sft-qwen35-9b.toml`
- Model: `Qwen/Qwen3.5-9B`
- Dataset: `tracks/leverage/datasets/reviewed-instructions/bootstrap.jsonl`
- Train export: `tracks/leverage/sft/bootstrap.train.jsonl`
- Rows: 1,216
- Epochs: 1
- Batch size: 2
- Gradient accumulation steps: 4
- Optimizer steps: 152
- Max length: 512
- Gradient checkpointing: `true`
- Output root: `outputs/leverage-sft-qwen35-9b`

## RunPod

- Pod: `llm-leverage-qwen35-9b-baseline-20260509-133051`
- Pod id: `8smgyg85certwt`
- Location: `US`
- GPU: `NVIDIA GeForce RTX 4090`
- Image: `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`
- CUDA filter: `12.8`, `12.9`, `13.0`
- Cost rate: `$0.69/h`
- Total wall time: 749.380 seconds
- Approximate cost: `749.380 / 3600 * $0.69 = about $0.14`
- Cleanup: completed automatically
- Final pod list: `[]`

## Timing

From `outputs/leverage-sft-qwen35-9b/runpod-timings.json`:

- Status: `passed`
- Pod create: 1.390 seconds
- SSH info wait: 92.130 seconds
- SSH ready wait: 1.055 seconds
- Setup: 31.069 seconds
- CUDA smoke: 10.448 seconds
- Training package install: 4.581 seconds
- Training package import smoke: 30.011 seconds
- Train command: 566.667 seconds
- Output sync: 5.181 seconds

## Metrics

From `outputs/leverage-sft-qwen35-9b/metrics.csv`:

- Rows: 1,216
- Steps: 608
- Optimizer steps: 152
- Tokens: 161,527
- Total trainer seconds: 543.985
- Pre-train seconds: 51.327
- Train seconds: 492.657
- Tokens/sec: 327.869
- Peak VRAM: 20.598GB
- GPU utilization average: 45.447%
- GPU utilization max: 97.000%
- Max driver memory used: 23,179MB / 24,564MB
- Final loss: 0.403785
- Status: `completed`

Adapter artifacts were written under:

```text
outputs/leverage-sft-qwen35-9b/lora-adapter/
```

The adapter output is intentionally not committed because `outputs/` is a
generated artifact directory.

## Interpretation

The 1,216-row reviewed dataset trained successfully with the stable Qwen3.5-9B
batch2 settings. There was no OOM and no NaN loss.

This run was faster and cheaper than the pre-run estimate. The estimate assumed
about 190k-196k tokens and 100-118 tokens/sec based on the prior 1,098-row run.
The measured run processed 161,527 tokens at 327.869 tokens/sec inside the
trainer.

The final loss is lower than the previous long-form constraint smoke run, but
that is not a capability claim. The next useful step is held-out evaluation of
the newly written adapter, especially against the same base-vs-adapter eval
suite used for earlier Qwen3.5-9B runs.

Note: old `post-training-*` files under the same output root predate this run
and should not be read as the evaluation result for this adapter.
