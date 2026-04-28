# Leverage SFT Smoke Plan

This plan defines the first weight-changing leverage experiment. It is a smoke
test for the data and training path, not a claim that the model improves.

## Objective

Verify that reviewed instructions can be exported into training JSONL and used
by a small student model in a bounded LoRA or SFT run.

## Inputs

- Reviewed source data: `datasets/reviewed-instructions/leverage-v0.jsonl`
- Training export: `data/sft/leverage_v0.train.jsonl`
- Held-out evals:
  - `evals/leverage-smoke.jsonl`
  - `evals/project-judgment-v0.jsonl`
- Config: `configs/leverage-sft-smoke.toml`

## Student Model

Start with `Qwen/Qwen3.5-0.6B` if available in the target training stack. Use
`Qwen/Qwen3-0.6B` as the fallback. The first run should prefer a small student
because the goal is wiring, not capability.

## Method

Prefer LoRA for the first run. Full SFT is acceptable only if the implementation
is simpler in the selected training stack and remains bounded to the 10 reviewed
examples.

## Before Training

```bash
uv run python -m llm.leverage.validate_reviewed-instructions \
  datasets/reviewed-instructions/leverage-v0.jsonl

uv run python -m llm.leverage.export_reviewed-instructions --overwrite
```

Do not launch a paid GPU job until the reviewed instructions validate and the
training export is regenerated locally.

## Success Criteria

- The reviewed instruction file validates.
- The training JSONL export is regenerated.
- The training command completes on at most 10 examples.
- An adapter or checkpoint artifact is written.
- The held-out eval command can run before and after training.

## Stop Conditions

- Dependency setup exceeds the planned runtime.
- The selected student model cannot load on the available GPU.
- Training loss becomes NaN.
- A paid run would exceed the cost cap.
- A paid resource cannot be cleaned up immediately.

## RunPod Policy

RunPod is not required for this plan by default. Use it only if local execution
cannot run the selected student model and the smoke objective still justifies
paid GPU time. If RunPod is used, set a hard cost cap, save only the expected
artifacts, and verify cleanup at the end.
