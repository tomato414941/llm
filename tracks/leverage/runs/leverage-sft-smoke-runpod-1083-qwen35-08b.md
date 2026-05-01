# Leverage SFT Smoke: 1083 Reviewed Rows

Date: 2026-05-01

This run verifies that the reviewed instruction dataset can drive a bounded
LoRA/SFT smoke on RunPod. It is a wiring and artifact test, not proof that the
small student improved in a durable way.

## Setup

- Student: `Qwen/Qwen3.5-0.8B`
- GPU: `NVIDIA GeForce RTX 4090`
- RunPod pod: `llm-leverage-sft-smoke-20260501-065231`
- RunPod pod id: `ltd8ylsikaz39v`
- Cost rate reported by RunPod: `$0.69/h`
- Reviewed rows exported: 1083
- Epochs: 1
- Steps: 1083

The first attempt used 3 epochs and timed out during training after the runner's
30-minute ceiling. The pod was deleted and `runpodctl pod list -o json` returned
an empty list. The smoke config was reduced to 1 epoch and the runtime ceiling
was raised to 60 minutes, which remains under the `$1.00` smoke cap at the
observed RTX 4090 rate.

## Timing

From `outputs/leverage-sft-smoke/runpod-timings.json`:

- Total wall time: 976.805 seconds
- Setup: 56.386 seconds
- CUDA smoke: 10.447 seconds
- Package import smoke: 33.872 seconds
- Train: 731.690 seconds
- Eval: 95.467 seconds
- Output sync: 1.820 seconds

Cleanup completed:

- `runpodctl pod delete ltd8ylsikaz39v` returned deleted
- `runpodctl pod list -o json` returned `[]`

## Training Output

From `outputs/leverage-sft-smoke/metrics.csv`:

```csv
metric,value
rows,1083
student_model,Qwen/Qwen3.5-0.8B
cuda_device,NVIDIA GeForce RTX 4090
steps,1083
final_loss,0.120919
status,completed
```

Artifacts were synced under `outputs/leverage-sft-smoke/`:

- `lora-adapter/adapter_model.safetensors`
- `lora-adapter/adapter_config.json`
- `metrics.csv`
- `notes.md`
- `post-training-predictions.jsonl`
- `post-training-scores.csv`
- `post-training-summary.csv`
- `runpod-timings.json`

## Eval Summary

The held-out eval completed on 30 tasks.

```csv
model,suite,task_count,passed_count,pass_rate
qwen3.5-0.8b-base,__overall__,30,14,0.467
qwen3.5-0.8b-lora-smoke,__overall__,30,15,0.500
qwen3.5-0.8b-base,leverage-smoke,12,8,0.667
qwen3.5-0.8b-lora-smoke,leverage-smoke,12,7,0.583
qwen3.5-0.8b-base,project-judgment,18,6,0.333
qwen3.5-0.8b-lora-smoke,project-judgment,18,8,0.444
```

Interpretation: the run successfully validates the SFT/LoRA path for the current
reviewed dataset. The eval result is mildly positive overall but too small and
too smoke-oriented to treat as model-quality evidence.
