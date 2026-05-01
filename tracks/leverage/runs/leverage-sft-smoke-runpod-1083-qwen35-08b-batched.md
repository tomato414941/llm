# Leverage SFT Smoke: Batched Training

Date: 2026-05-01

This run measures the batched PyTorch training loop added after the initial
1083-row LoRA/SFT smoke. The goal is runtime validation before trying a larger
baseline model, not model-quality proof.

## Setup

- Student: `Qwen/Qwen3.5-0.8B`
- GPU: `NVIDIA GeForce RTX 4090`
- RunPod pod: `llm-leverage-sft-smoke-batched-20260501-072311`
- RunPod pod id: `2z6hd9zioh40er`
- Cost rate reported by RunPod: `$0.69/h`
- Reviewed rows exported: 1083
- Epochs: 1
- Batch size: 4

Cleanup completed:

- `runpodctl pod delete 2z6hd9zioh40er` returned deleted
- `runpodctl pod list -o json` returned `[]`

## Timing

From `outputs/leverage-sft-smoke/runpod-timings.json`:

- Total wall time: 385.575 seconds
- Setup: 31.863 seconds
- CUDA smoke: 8.941 seconds
- Package import smoke: 26.240 seconds
- Train command: 201.648 seconds
- Eval: 73.197 seconds
- Output sync: 0.817 seconds

Previous non-batched 1083-row smoke:

- Total wall time: 976.805 seconds
- Train command: 731.690 seconds
- Eval: 95.467 seconds

The batched train command was about 3.6x faster than the previous non-batched
run: `731.690 / 201.648 = 3.63`.

## Training Metrics

From `outputs/leverage-sft-smoke/metrics.csv`:

```csv
metric,value
rows,1083
student_model,Qwen/Qwen3.5-0.8B
cuda_device,NVIDIA GeForce RTX 4090
epochs,1
batch_size,4
steps,271
tokens,104115
train_seconds,165.709
tokens_per_second,628.302
final_loss,0.551482
status,completed
```

## Eval Summary

The held-out eval completed on 30 tasks.

```csv
model,suite,task_count,passed_count,pass_rate
qwen3.5-0.8b-base,__overall__,30,14,0.467
qwen3.5-0.8b-lora-smoke,__overall__,30,14,0.467
qwen3.5-0.8b-base,leverage-smoke,12,8,0.667
qwen3.5-0.8b-lora-smoke,leverage-smoke,12,7,0.583
qwen3.5-0.8b-base,project-judgment,18,6,0.333
qwen3.5-0.8b-lora-smoke,project-judgment,18,7,0.389
```

Interpretation: batching materially improves runtime and GPU use while preserving
the smoke contract. Eval is neutral overall and should not be treated as a
quality claim.
