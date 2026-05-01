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

## GPU Utilization Rerun

After replacing the progress logger's `torch.cuda.utilization()` path with
`nvidia-smi` sampling, the same batch4 200-row run was executed again.

- Pod: `llm-leverage-sft-qwen35-9b-batch4-200-gpuutil-20260501-182832`
- Pod id: `buqij94r6lkqbt`
- Total wall time: 200.592 seconds
- Train command: 109.128 seconds
- Train seconds: 68.771
- Tokens/sec: 220.501
- Peak VRAM: 21.886GB
- Cleanup: completed automatically

The progress log now records GPU utilization and driver-reported memory:

```text
step,optimizer_steps,tokens,loss,tokens_per_second,peak_vram_gb,gpu_utilization_percent,gpu_memory_used_mb,gpu_memory_total_mb
10,2,3419,1.421450,249.703,18.979,24,21034,24564
20,5,6892,2.270327,245.775,21.886,29,23854,24564
30,7,9847,1.770413,241.267,21.886,32,23854,24564
40,10,12742,1.460791,234.733,21.886,22,23854,24564
50,12,15164,0.652468,225.530,21.886,26,23858,24564
```

Interpretation: GPU utilization samples were low, in the 22-32% range, while
memory was nearly full at about 23.9GB of 24.6GB. For this setup, the immediate
constraint is VRAM headroom rather than utilization alone.

## Continuous GPU Sampling

The trainer was then updated to record `logs/gpu-samples.csv` once per second
during the training loop and summarize utilization in `metrics.csv`.

- Pod: `llm-leverage-sft-qwen35-9b-batch4-200-gpusamples-20260501-184213`
- Pod id: `1qb2qh4vqu1uht`
- Total wall time: 257.532 seconds
- Train command: 158.513 seconds
- Train seconds: 64.049
- Tokens/sec: 236.756
- Peak VRAM: 21.886GB
- Cleanup: completed automatically

From `outputs/leverage-sft-qwen35-9b-batch4-200/metrics.csv`:

- GPU samples: 57
- Average GPU utilization: 33.754%
- Max GPU utilization: 89.000%
- Max driver-reported GPU memory: 23858MB / 24564MB

Distribution from `logs/gpu-samples.csv`:

- Samples >= 30% utilization: 40 / 57
- Samples >= 40% utilization: 10 / 57
- Samples >= 50% utilization: 3 / 57
- Samples >= 80% utilization: 1 / 57

Interpretation: step-time sampling was noisy but not fundamentally misleading.
Continuous sampling confirms that average utilization is still low for this
short-sequence, small-effective-batch workload. The model occasionally reaches
high utilization, but most samples sit below 40%. Since memory is already near
the RTX 4090 limit, the next throughput improvement should not be a simple
batch-size increase. More plausible next tests are sequence packing, longer
examples, disabling gradient checkpointing if memory permits, or a larger-VRAM
GPU.
