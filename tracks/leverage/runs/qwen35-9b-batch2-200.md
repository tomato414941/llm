# Qwen3.5-9B Batch Size 2 200-Row Measurement

Date: 2026-05-01

Goal: check whether `Qwen/Qwen3.5-9B` can train with `batch_size=2` and whether
it improves throughput versus the completed `batch_size=1` baseline.

## Setup

- Config: `tracks/leverage/configs/leverage-sft-qwen35-9b-batch2-200.toml`
- Train export: `tracks/leverage/sft/bootstrap-200.train.jsonl`
- Rows: 200
- Model: `Qwen/Qwen3.5-9B`
- GPU: `NVIDIA GeForce RTX 4090`
- Image: `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`
- CUDA filter: `12.8`, `12.9`, `13.0`
- Batch size: 2
- Max length: 512
- Gradient checkpointing: `true`
- Gradient accumulation steps: 4
- Log every steps: 10

## First Attempt

- Pod id: `t4ybbxtlomubbs`
- Location: `NL`
- Result: did not reach SSH
- Symptom: `RUNNING`, `22/tcp`, `pod not ready`, `uptimeSeconds=0`
- Cleanup: manually deleted

This did not test batch size. It was another RunPod placement/container-start
failure before the training command could run.

## Successful Retry

- Pod: `llm-leverage-sft-qwen35-9b-batch2-200-retry-20260501-172951`
- Pod id: `xza9fyehsojoer`
- Location: `US`
- RunPod reported cost rate: `$0.69/h`
- Total wall time: 280.552 seconds
- Approximate cost: `$0.054`
- Cleanup: completed automatically

From `outputs/leverage-sft-qwen35-9b-batch2-200/runpod-timings.json`:

- Status: `passed`
- Pod create: 1.517 seconds
- SSH info wait: 23.515 seconds
- Setup: 40.543 seconds
- CUDA smoke: 13.205 seconds
- Dependency command: 4.932 seconds
- Train step: 190.633 seconds
- Output sync: 1.268 seconds

From `outputs/leverage-sft-qwen35-9b-batch2-200/metrics.csv`:

- Rows: 200
- Steps: 100
- Optimizer steps: 25
- Tokens: 15164
- Train seconds: 121.882
- Tokens/sec: 124.415
- Peak VRAM: 19.454GB
- Final loss: 0.622881
- Status: `completed`

Progress logging worked and wrote:

```text
outputs/leverage-sft-qwen35-9b-batch2-200/logs/progress.csv
```

The last progress row was:

```text
step=100 optimizer_steps=25 tokens=15164 loss=0.622881 tokens_per_second=126.066 peak_vram_gb=19.454
```

## Interpretation

`batch_size=2` is viable for this 200-row Qwen3.5-9B LoRA/SFT measurement. It
used only about 1GB more peak VRAM than the prior full `batch_size=1` run
recorded in `qwen35-9b-full-gradient-checkpointing.md`.

Throughput improved materially:

- Prior full `batch_size=1` run: 61.338 tokens/sec
- This 200-row `batch_size=2` run: 124.415 tokens/sec

This is not a perfect apples-to-apples comparison because the datasets differ
in row count and token distribution, but it is strong enough to justify trying
`batch_size=4` on a 200-row measurement before changing the full baseline.

Direct GPU utilization was not captured in this run's progress CSV. The trainer
now records `gpu_utilization_percent` for future runs.
