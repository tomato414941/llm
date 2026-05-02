# Project Agent Policy

## Purpose
- This project is a hybrid LLM learning and experimentation lab.
- The long-term goal is to understand and build toward general-purpose language models under limited compute.
- The project combines from-scratch implementation with open-model leverage.

## Scope
- Implement small language models from first principles to understand the internals.
- Study nanoGPT-style training loops and modern decoder-only Transformer design.
- Experiment with low-cost paths toward general capability, including public data, open weights, distillation, fine-tuning, and evaluation.
- Keep experiments reproducible and small enough to run on modest hardware unless explicitly noted.

## Tracks
- `from-scratch`: tokenizer, bigram/MLP baselines, causal attention, Transformer blocks, GPT-style training, sampling, and evaluation.
- `leverage`: open-model comparison, LoRA/QLoRA, SFT, DPO, distillation, quantization, and task/eval design.

## Coding Rules
- Use Python for the initial implementation.
- Use uv for Python dependency and environment management.
- Prefer PyTorch for model code.
- Keep implementation simple before adding abstractions.
- Add tests for reusable logic, data transforms, tokenizers, and training utilities.
- Do not commit datasets, checkpoints, secrets, or large generated artifacts.
- Prefer `uv add` / `uv add --dev` for managed dependencies.
- Defer CUDA-specific PyTorch indexes and heavy LLM dependencies until experiments require them.

## Benchmark Rules
- Run full benchmarks separately by benchmark task and model variant.
- A model variant is one evaluated target, such as a base model or the same base model with a LoRA adapter.
- Do not combine multiple model variants in one long benchmark job.
- Keep generated benchmark outputs under ignored `outputs/` paths and commit only concise run notes.
- After any RunPod benchmark job, verify `runpodctl pod list -o json` is empty.

## Execution Cost Rules
- Before running cost-bearing work, propose whether the run should be recorded in `tracks/leverage/docs/execution-costs.md`.
- Cost-bearing work includes RunPod jobs, external benchmarks, generated-data API calls, judge API calls, inference probes, and LoRA/SFT training.
- After cost-bearing work completes, propose recording the measured time and cost in `tracks/leverage/docs/execution-costs.md` when the result can change future execution decisions.
- Keep detailed context in a run note under `tracks/leverage/runs/`; keep the compact summary table in `tracks/leverage/docs/execution-costs.md`.

## Git Rules
- Main development happens on `main`.
- Use small commits with English messages in `type: description` format.
- Before large experiments or generated outputs, confirm `.gitignore` excludes them.
