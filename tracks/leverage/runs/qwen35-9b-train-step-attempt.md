# Qwen3.5-9B One-Step Train Attempt

Date: 2026-05-01

Goal: run the smallest useful 9B LoRA training check after the successful
load-only smoke: one reviewed row, batch size 1, one optimizer step, adapter
save, and peak VRAM recording.

This was intentionally kept as a one-off RunPod command. No reusable 9B train
step framework was added.

## Result

The run did not reach CUDA, model loading, or training.

- Requested GPU: `NVIDIA GeForce RTX 4090`
- Image: `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`
- Pod: `llm-leverage-qwen35-9b-train-step-20260501-093102`
- Pod id: `s5w6blv4lot3mj`
- RunPod reported cost rate: `$0.69/h`
- Failure point: pod stayed `RUNNING` but `runpodctl ssh info` kept returning
  `pod not ready`
- CUDA smoke: not reached
- Model download/load: not reached
- Backward/optimizer step: not reached
- Adapter save: not reached
- VRAM result: not measured

The pod was manually deleted to avoid idle billing:

```text
runpodctl pod delete s5w6blv4lot3mj
runpodctl pod list -o json
[]
```

Interpretation: this is another RunPod readiness/SSH failure, not evidence that
`Qwen/Qwen3.5-9B` training fails on 24GB VRAM. The earlier load-only smoke with
the same image reached SSH readiness and passed.

## Follow-up

Do not add more project code for this failure. If another 9B train-step attempt
is worth the cost, retry on a fresh pod or choose a different GPU/host class to
reduce RunPod readiness variance.
