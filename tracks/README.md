# Tracks

This directory separates project-specific LLM workstreams from shared code.

The source of truth for reusable code stays under `src/llm/`. Do not use this
directory as a second implementation tree.

## Directory Roles

`from-scratch/` stores the small-LM training track:

- configs
- local corpora and tokenized data
- tokenizer artifacts
- checkpoints
- small-LM eval prompts
- training metrics, observations, and summaries

`leverage/` stores the hosted/open-model leverage track:

- model behavior spec
- generation seed prompts
- held-out evals
- reviewed instruction datasets
- SFT exports
- local API/model run outputs and judge summaries

Leverage JSONL evaluation is local and deterministic. Store tasks under
`tracks/leverage/evals/`, store saved model predictions under
`tracks/leverage/runs/`, then run the evaluator to produce reviewable CSV
summaries. The evaluator scores saved predictions only; it does not run models,
create RunPod jobs, download weights, call APIs, or fetch datasets.

Do not commit:

- private or large datasets
- checkpoints
- large generated outputs
- secrets or local environment details
