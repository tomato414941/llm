# SFT Smoke Early Stopping

Date: 2026-05-11

## Goal

Verify that the validation-loss early stopping path runs end-to-end on the
low-cost `Qwen/Qwen3.5-0.8B` LoRA/SFT smoke before using the same plumbing for
larger Qwen3.5-9B runs.

This was a training-plumbing run, not a quality-improvement run.

## Setup

- Config: `tracks/leverage/configs/leverage-sft-smoke-early-stop.toml`
- Student model: `Qwen/Qwen3.5-0.8B`
- Method: LoRA
- Training rows: 1184
- Validation rows: 32
- Epoch cap: 2
- Batch size: 4
- Gradient accumulation steps: 1
- Early stopping:
  - validation examples: 32
  - eval every steps: 20
  - patience: 2
  - min delta: 0.001
- GPU: `NVIDIA GeForce RTX 4090`
- Cloud: RunPod Secure Cloud
- Template id: `runpod-torch-v280`
- Image: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
- CUDA filter: `12.8`, `12.9`, `13.0`
- Cost rate: `$0.69/h`
- Output path: `outputs/leverage-sft-smoke-early-stop`
- Pod: `llm-leverage-sft-smoke-early-stop-20260511-084100`
- Pod id: `moy79lfsjkyv9h`

Cleanup completed:

```text
runpodctl pod delete moy79lfsjkyv9h
runpodctl pod list -o json
[]
```

## Result

The run passed operationally. Early stopping triggered at step 100.

From `outputs/leverage-sft-smoke-early-stop/metrics.csv`:

| metric | value |
| --- | ---: |
| rows | 1216 |
| training rows | 1184 |
| validation rows | 32 |
| steps | 100 |
| optimizer steps | 100 |
| validation checks | 5 |
| final loss | 0.251565 |
| final validation loss | 2.950196 |
| best validation loss | 2.362285 |
| best validation step | 60 |
| early stopped | true |
| early stopping stop step | 100 |
| max memory allocated GB | 16.107 |

Validation loss progression from
`outputs/leverage-sft-smoke-early-stop/logs/progress.csv`:

| step | training loss | validation loss | best validation loss | checks without improvement |
| ---: | ---: | ---: | ---: | ---: |
| 20 | 2.279516 | 2.401694 | 2.401694 | 0 |
| 40 | 1.354098 | 2.380331 | 2.380331 | 0 |
| 60 | 0.671033 | 2.362285 | 2.362285 | 0 |
| 80 | 0.809742 | 2.499998 | 2.362285 | 1 |
| 100 | 0.251565 | 2.950196 | 2.362285 | 2 |

## Timing

From `outputs/leverage-sft-smoke-early-stop/runpod-timings.json`:

- Total RunPod wall time: 230.713 seconds
- SSH info wait: 22.056 seconds
- Setup: 32.834 seconds
- CUDA smoke: 7.892 seconds
- Runtime package install: 28.004 seconds
- Preflight: 9.395 seconds
- Training command: 123.815 seconds
- Output sync: 0.674 seconds

From `outputs/leverage-sft-smoke-early-stop/metrics.csv`:

- Total training script seconds: 93.750
- Pre-train seconds: 21.048
- Train seconds: 72.702
- Tokens/sec: 455.063
- GPU utilization avg percent: 33.300
- GPU utilization max percent: 89.000
- GPU memory used max MB: 21333

Approximate cost: `$0.04`.

## Interpretation

The early-stopping plumbing works on a real CUDA training run:

- validation split was created,
- validation loss was computed during training,
- progress CSV recorded validation fields,
- patience-based stopping triggered,
- metrics and notes captured the early-stopping outcome,
- adapter output was still written,
- RunPod cleanup completed.

The run also shows why this should remain opt-in: training loss continued to
fall while validation loss worsened after step 60. That is useful diagnostic
behavior for multi-epoch experiments, but not a reason to change the existing
1-epoch baseline configs.
