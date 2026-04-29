# Leverage SFT Smoke RunPod Result

Date: 2026-04-28

## Goal

Verify that the reviewed leverage instruction dataset can be exported and used
for a bounded CUDA LoRA smoke run on RunPod.

## Command

```bash
uv run python scripts/runpod/run_once.py \
  --name llm-leverage-sft-smoke \
  --gpu-type 'NVIDIA GeForce RTX 3090' \
  --max-cost 0.8 \
  --mem 24 \
  --sync tracks/leverage/configs \
  --sync tracks/leverage/datasets \
  --sync tracks/leverage/evals \
  --sync tracks/leverage/sft \
  --output outputs/leverage-sft-smoke \
  --local 'uv run python -m llm.leverage.sft_smoke_preflight --config tracks/leverage/configs/leverage-sft-smoke.toml --overwrite' \
  --remote 'uv pip install transformers peft trl accelerate' \
  --remote 'uv run python -u -m llm.leverage.train_sft_smoke --config tracks/leverage/configs/leverage-sft-smoke.toml' \
  --remote 'uv run python -u -m llm.leverage.evaluate_sft_adapter --config tracks/leverage/configs/leverage-sft-smoke.toml'
```

## Result

Status: completed

- Reviewed instructions validated.
- Training export regenerated with 10 rows.
- RunPod CUDA check passed on `NVIDIA GeForce RTX 3090`.
- Training packages imported: `torch`, `transformers`, `peft`, `trl`.
- Student model loaded: `Qwen/Qwen3-0.6B`.
- LoRA smoke training completed for 30 steps.
- Adapter artifact synced to `outputs/leverage-sft-smoke/lora-adapter/`.
- Metrics and notes synced to `outputs/leverage-sft-smoke/`.
- Post-training base-vs-adapter eval completed for 30 tasks.
- RunPod pod cleanup verified with no active pods remaining.

## Metrics

```csv
metric,value
rows,10
student_model,Qwen/Qwen3-0.6B
cuda_device,NVIDIA GeForce RTX 3090
steps,30
final_loss,2.990553
status,completed
```

## Post-Training Eval

The post-training eval is a wiring comparison, not a capability claim.

```csv
model,suite,category,task_count,passed_count,avg_score,pass_rate
qwen3-0.6b-base,__overall__,__overall__,30,3,0.100,0.100
qwen3-0.6b-lora-smoke,__overall__,__overall__,30,2,0.067,0.067
qwen3-0.6b-base,leverage-smoke,__overall__,12,3,0.250,0.250
qwen3-0.6b-lora-smoke,leverage-smoke,__overall__,12,2,0.167,0.167
qwen3-0.6b-base,project-judgment,__overall__,18,0,0.000,0.000
qwen3-0.6b-lora-smoke,project-judgment,__overall__,18,0,0.000,0.000
```

## Notes

`outputs/` and `tracks/leverage/sft/` are ignored generated artifacts. This
tracked run note is the durable record for the first weight-changing leverage
smoke.

The original `Qwen/Qwen3.5-0.6B` model id did not load from Hugging Face during
the smoke. The config now uses the verified `Qwen/Qwen3-0.6B` model id directly.

## Next Step

The next useful step is to inspect the failed post-training predictions and
decide whether the first real training iteration should change data, prompt
formatting, decoding, or eval expectations.
