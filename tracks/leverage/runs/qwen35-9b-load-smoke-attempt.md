# Qwen3.5-9B Load Smoke Attempt

Date: 2026-05-01

Goal: check whether `Qwen/Qwen3.5-9B` can load on a RunPod RTX 4090, attach
LoRA, run one forward pass, and report peak VRAM before attempting any 9B LoRA
training.

This was intentionally kept as a one-off RunPod remote command. No reusable
load-only smoke framework was added.

## Result

The run did not reach model loading.

- Requested GPU: `NVIDIA GeForce RTX 4090`
- Template: `runpod-torch-v280`
- Pod: `llm-leverage-qwen35-9b-load-20260501-083933`
- Pod id: `eyj47jrcft4p1s`
- RunPod reported cost rate: `$0.69/h`
- Failure point: pod stayed `RUNNING` but `runpodctl ssh info` kept returning
  `pod not ready`
- Model download/load: not reached
- LoRA attach: not reached
- Forward pass: not reached
- VRAM result: not measured

The pod was manually deleted to avoid idle billing:

```text
runpodctl pod delete eyj47jrcft4p1s
runpodctl pod list -o json
[]
```

Interpretation: this is a RunPod readiness/SSH issue, not evidence that
`Qwen/Qwen3.5-9B` fails on 24GB VRAM.

## Follow-up

Retry once with the same one-off command on a fresh pod. If SSH readiness fails
again, switch GPU/provider choice before changing project code.
