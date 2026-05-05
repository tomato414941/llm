# Leverage Track

This is the current mainline. The goal is to test whether reviewed instruction
data and LoRA/SFT can improve open-weight models under a simple, repeatable
loop.

## Main Loop

```text
seed prompts
  -> teacher generation
  -> filter and judge
  -> reviewed instructions
  -> SFT export
  -> LoRA smoke
  -> base-vs-adapter eval
  -> next data/model decision
```

## Current Source Of Truth

- Behavior target: `model-spec.md`
- Training config: `configs/leverage-sft-smoke.toml`
- Reviewed dataset: `datasets/reviewed-instructions/bootstrap.jsonl`
- SFT export: `sft/bootstrap.train.jsonl`
- Held-out evals: `evals/leverage-smoke.jsonl`,
  `evals/project-judgment.jsonl`, `evals/leverage-model-spec.jsonl`
- Reusable command map: `docs/commands.md`
- LoRA/SFT RunPod guide: `docs/lora-sft-runpod.md`
- Reviewed-data mix plan: `docs/reviewed-instruction-mix-plan.md`
- External benchmark plan: `docs/external-benchmarks.md`
- Latest smoke result:
  `runs/qwen35-9b-lora-long-form-constraint-smoke.md`

## Model Roles

- Smoke/test student: `Qwen/Qwen3.5-0.8B`
- Intended Qwen baseline target: `Qwen/Qwen3.5-9B`
- DeepSeek smoke/test student: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- Intended DeepSeek baseline target:
  `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`
- Challenging open-weight target: `openai/gpt-oss-20b`

Prioritize the Qwen path before side-model smoke tests. The 300-row readiness
and 1,000-row pilot thresholds have been reached; 3,000+ reviewed rows is the
next target for a serious `Qwen/Qwen3.5-9B` capability-seeking run.

## Keep It Simple

Committed files should either define the loop, provide a small reviewed dataset
or eval, document one important result, or test the tooling. Large generated
outputs, adapters, checkpoints, raw provider responses, and temporary SFT
exports stay under ignored paths such as `outputs/`.

When a new result supersedes an old run note, keep only the result that informs
the next decision.
