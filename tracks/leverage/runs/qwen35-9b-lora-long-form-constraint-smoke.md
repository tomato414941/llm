# Qwen3.5-9B Long-Form Constraint LoRA Smoke

Date: 2026-05-04

## Goal

Run `Qwen/Qwen3.5-9B` LoRA/SFT on the 1,098-row reviewed dataset after
promoting the long-form constraint rows.

This is a training stability and pipeline check, not a capability claim.

## Config

- Config: `tracks/leverage/configs/leverage-sft-qwen35-9b-long-form-constraint.toml`
- Model: `Qwen/Qwen3.5-9B`
- Dataset: `tracks/leverage/datasets/reviewed-instructions/bootstrap.jsonl`
- Train export: `tracks/leverage/sft/bootstrap.train.jsonl`
- Rows: 1,098
- Epochs: 1
- Batch size: 2
- Gradient accumulation steps: 4
- Optimizer steps: 138
- Max length: 512
- Gradient checkpointing: `true`
- Output root: `outputs/leverage-sft-qwen35-9b-long-form-constraint`

## First Attempt

- Pod: `qwen35-9b-lora-long-form-constraint-20260504-002702`
- Pod id: `9457ll25vki3yj`
- Location: `IS`
- GPU: `NVIDIA GeForce RTX 4090`
- Image: `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`
- Cost rate: `$0.69/h`
- Total wall time: 1,488.523 seconds
- Approximate cost: `$0.29`
- Result: failed before training
- Cleanup: completed automatically

Failure:

```text
RuntimeError: missing optional SFT training packages: transformers, peft, trl.
```

The attempt synced data and passed CUDA smoke, but the remote command did not
install the optional training packages before invoking the trainer.

Captured from the failed attempt before the successful rerun overwrote the local
timing artifact:

- Setup: 1,442.873 seconds
- CUDA smoke: 5.481 seconds
- Train command before failure: 3.024 seconds

## Successful Run

- Pod: `qwen35-9b-lora-long-form-constraint-20260504-005222`
- Pod id: `h93cue0ssy8gbu`
- Location: `CZ`
- GPU: `NVIDIA GeForce RTX 4090`
- Image: `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`
- Cost rate: `$0.69/h`
- Total wall time: 1,100.261 seconds
- Approximate cost: `$0.21`
- Cleanup: completed automatically
- Final pod list: `[]`

The rerun added:

```text
uv pip install transformers peft trl accelerate
```

and a package import smoke before training.

From `outputs/leverage-sft-qwen35-9b-long-form-constraint/runpod-timings.json`:

- Status: `passed`
- SSH info wait: 12.835 seconds
- SSH ready wait: 1.397 seconds
- Setup: 39.283 seconds
- CUDA smoke: 8.039 seconds
- Training package install: 4.733 seconds
- Training package import smoke: 21.893 seconds
- Train command: 995.435 seconds
- Output sync: 6.784 seconds

From `outputs/leverage-sft-qwen35-9b-long-form-constraint/metrics.csv`:

- Rows: 1,098
- Steps: 549
- Optimizer steps: 138
- Tokens: 109,860
- Total trainer seconds: 980.707
- Pre-train seconds: 36.108
- Train seconds: 944.599
- Tokens/sec: 116.303
- Peak VRAM: 20.598GB
- GPU utilization average: 19.405%
- GPU utilization max: 51.000%
- Max driver memory used: 23,555MB / 24,564MB
- Final loss: 1.720564
- Status: `completed`

Adapter artifacts were written under:

```text
outputs/leverage-sft-qwen35-9b-long-form-constraint/lora-adapter/
```

The adapter output is intentionally not committed because `outputs/` is a
generated artifact directory.

## Interpretation

The 1,098-row dataset with the new long-form constraint rows trained
successfully with the stable Qwen3.5-9B batch2 settings. There was no OOM and no
NaN loss.

Compared with the previous 1,083-row batch2 run, this run processed slightly
more rows and tokens with similar throughput:

- previous batch2 train seconds: 882.073
- this run train seconds: 944.599
- previous tokens/sec: 118.034
- this run tokens/sec: 116.303
- previous peak VRAM: 20.067GB
- this run peak VRAM: 20.598GB

The extra 15 long-form rows did not break the training path. The run does not
show whether the adapter improved IFEval; that requires a separate adapter eval.

The failed first attempt is also useful: optional SFT training packages must be
installed explicitly in RunPod one-off training commands unless they are added
to the project environment or baked into the image.
