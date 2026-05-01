# Qwen3.5-9B Full Reviewed-Data Run With Gradient Checkpointing

Date: 2026-05-01

Goal: run the full reviewed-instruction LoRA/SFT baseline on `Qwen/Qwen3.5-9B`
after adding the RunPod CUDA filter.

## First Attempt

The first CUDA-filtered full run reached SSH, passed CUDA smoke, downloaded the
model, and loaded weights, but failed during training with CUDA OOM.

- Pod id: `koik93mdngaoi9`
- GPU: `NVIDIA GeForce RTX 4090`
- Image: `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`
- CUDA filter: `12.8`, `12.9`, `13.0`
- Failure: `torch.OutOfMemoryError`
- Cleanup: completed automatically

Interpretation: CUDA placement and model loading were no longer blockers. The
trainer needed a standard activation-memory reduction path for 9B LoRA on this
GPU class.

## Change

Enabled `method.gradient_checkpointing = true` for the Qwen3.5-9B config and
made the trainer call:

- `model.config.use_cache = False`
- `model.gradient_checkpointing_enable()`
- `model.enable_input_require_grads()` when available

This keeps the change scoped to the 9B baseline instead of lowering sequence
length or changing the dataset.

## Successful Run

- Pod: `llm-leverage-sft-qwen35-9b-full-gc-20260501-162820`
- Pod id: `doc7mh0z7fjrg3`
- RunPod reported cost rate: `$0.69/h`
- Approximate billable wall time: 1859.692 seconds, about 0.517 hours
- Approximate cost: `$0.36`
- Machine location: `CZ`
- GPU: `NVIDIA GeForce RTX 4090`
- Image: `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`
- Container disk: 120GB
- Volume: 80GB mounted at `/workspace`
- CUDA smoke: passed with `torch=2.8.0+cu128`
- Output sync: completed
- Cleanup: completed automatically

From `outputs/leverage-sft-qwen35-9b/runpod-timings.json`:

- Status: `passed`
- Total wall time: 1859.692 seconds
- Pod create: 1.174 seconds
- SSH info wait: 37.604 seconds
- Setup: 45.411 seconds
- CUDA smoke: 8.693 seconds
- Dependency command: 4.882 seconds
- Train step: 1745.938 seconds
- Output sync: 7.087 seconds

From `outputs/leverage-sft-qwen35-9b/metrics.csv`:

- Rows: 1083
- Steps: 1083
- Epochs: 1
- Batch size: 1
- Max length: 512
- Gradient checkpointing: `True`
- Tokens: 104115
- Train seconds: 1697.385
- Tokens/sec: 61.338
- Peak VRAM: 18.468GB
- Final loss: 0.099927
- Status: `completed`

Adapter artifacts were written under:

```text
outputs/leverage-sft-qwen35-9b/lora-adapter/
```

The adapter output is intentionally not committed because `outputs/` is a
generated artifact directory.

## Interpretation

The Qwen3.5-9B reviewed-data baseline is now trainable on the current RunPod
path with the CUDA filter and gradient checkpointing.

The run also exposed a real KISS issue: the trainer prints no per-step progress
and writes metrics only after completion. That made a successful 29-minute
training run look stalled for most of its runtime. The next improvement should
be minimal progress logging or periodic metrics, not another abstraction layer.

## Batch Size Follow-Up

After the 200-row measurements, the full baseline was tested again with larger
batch settings.

### Full Batch4 Attempt

- Pod: `llm-leverage-sft-qwen35-9b-full-batch4-20260501-185745`
- GPU: `NVIDIA GeForce RTX 4090`
- Image: `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`
- Rows: 1083
- Batch size: 4
- Result: failed with CUDA OOM around `step=130`
- Peak observed VRAM before failure: `21.886GB`
- Cleanup: completed automatically

Interpretation: the 200-row batch4 measurement was useful for throughput, but
it was not enough to prove full-data stability. Later full-data batches required
another `1.62GiB` allocation while only about `289MiB` was free.

### Full Batch2 Run

- Pod: `llm-leverage-sft-qwen35-9b-full-batch2-20260501-190538`
- Pod id: `aae6hh59fuyat1`
- RunPod reported cost rate: `$0.69/h`
- Approximate billable wall time: 1344.578 seconds, about 0.374 hours
- Approximate cost: `$0.26`
- Machine location: `RO`
- GPU: `NVIDIA GeForce RTX 4090`
- Image: `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`
- Container disk: 120GB
- Volume: 80GB mounted at `/workspace`
- CUDA smoke: passed with `torch=2.8.0+cu128`
- Output sync: completed
- Cleanup: completed automatically

From `outputs/leverage-sft-qwen35-9b/runpod-timings.json`:

- Status: `passed`
- Total wall time: 1344.578 seconds
- SSH info wait: 188.213 seconds
- Setup: 43.849 seconds
- CUDA smoke: 31.915 seconds
- Dependency command: 10.144 seconds
- Train step: 1047.505 seconds
- Output sync: 7.640 seconds

From `outputs/leverage-sft-qwen35-9b/metrics.csv`:

- Rows: 1083
- Steps: 542
- Optimizer steps: 136
- Epochs: 1
- Batch size: 2
- Max length: 512
- Gradient checkpointing: `True`
- Gradient accumulation steps: 4
- Tokens: 104115
- Total seconds: 941.115
- Train seconds: 882.073
- Tokens/sec: 118.034
- Peak VRAM: 20.067GB
- GPU utilization average: 19.519%
- GPU utilization max: 53.000%
- Final loss: 0.113780
- Status: `completed`

Current decision: use `batch_size=2` for the full Qwen3.5-9B reviewed-data
baseline on RTX 4090. It is slower than the 200-row batch4 measurement, but it
is the simplest setting that completed the full dataset with usable VRAM
headroom.

## Post-Training Eval

After the full batch2 adapter was written, the base model and adapter were
compared on the held-out eval tasks configured for the SFT baseline.

The first eval attempt failed because the eval path loaded the 9B model without
the training config's `bfloat16` dtype. The eval loader was changed to use
`bfloat16` on CUDA and to load base and adapter sequentially instead of keeping
both models on the GPU.

- Pod: `llm-leverage-eval-qwen35-9b-full-batch2-20260501-194124`
- Pod id: `xixndzg88w6cid`
- RunPod reported cost rate: `$0.69/h`
- Approximate billable wall time: 276.281 seconds, about 0.077 hours
- Approximate cost: `$0.05`
- Machine location: `RO`
- GPU: `NVIDIA GeForce RTX 4090`
- Image: `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`
- Output sync: completed
- Cleanup: completed automatically

From `outputs/leverage-sft-qwen35-9b/runpod-timings.json`:

- Status: `passed`
- Total wall time: 276.281 seconds
- SSH info wait: 35.902 seconds
- Setup: 31.872 seconds
- CUDA smoke: 18.722 seconds
- Dependency command: 8.693 seconds
- Eval step: 159.635 seconds
- Output sync: 2.372 seconds

From `outputs/leverage-sft-qwen35-9b/post-training-summary.csv`:

- Base overall: 15/30, pass rate 0.500
- Adapter overall: 17/30, pass rate 0.567
- Base `leverage-smoke`: 9/12, pass rate 0.750
- Adapter `leverage-smoke`: 9/12, pass rate 0.750
- Base `project-judgment`: 6/18, pass rate 0.333
- Adapter `project-judgment`: 8/18, pass rate 0.444

Interpretation: this is a small positive result, not a strong capability claim.
The adapter improved the exact held-out score by 2 tasks, mainly in
`project-judgment`, while keeping `leverage-smoke` flat. The eval remains a
strict string/regex harness, so some failures are scoring-contract failures
rather than clear semantic failures.
