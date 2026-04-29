# llm

Small LLM learning lab.

This repository has two tracks:

- `tracks/from-scratch/`: implement and train small decoder-only language models
  to understand the mechanics.
- `tracks/leverage/`: evaluate, adapt, and operate existing open models under
  practical constraints.

See `ROADMAP.md` for goals, non-goals, and RunPod policy. See `tracks/README.md`
for directory roles.

## Setup

```bash
uv sync --extra dev
```

## From-Scratch Quickstart

Put local text data under `tracks/from-scratch/data/raw/`. Data files are
ignored by git.

Train a BPE tokenizer:

```bash
uv run python -m llm.tokenize \
  --input tracks/from-scratch/data/raw/tinyshakespeare.txt \
  --output tracks/from-scratch/data/processed/bpe_tokenizer.json \
  --vocab-size 500
```

Prepare token tensors:

```bash
uv run python -m llm.prepare_data \
  --input tracks/from-scratch/data/raw/tinyshakespeare.txt \
  --tokenizer tracks/from-scratch/data/processed/bpe_tokenizer.json \
  --output tracks/from-scratch/data/processed/tinyshakespeare_bpe_500.pt
```

Train from a TOML config:

```bash
uv run python -m llm.train \
  --config tracks/from-scratch/configs/tinyshakespeare-bpe-500-small.toml
```

Generate from a checkpoint:

```bash
uv run python -m llm.generate \
  --checkpoint tracks/from-scratch/checkpoints/mini-gpt.pt \
  --prompt "KING:" \
  --max-new-tokens 200 \
  --seed 1337 \
  --samples 2
```

Evaluate and observe a checkpoint:

```bash
uv run python -m llm.evaluate \
  --checkpoint tracks/from-scratch/checkpoints/mini-gpt.pt \
  --tokens tracks/from-scratch/data/processed/tinyshakespeare_bpe_500.pt

uv run python -m llm.observe \
  --config tracks/from-scratch/configs/tinyshakespeare-bpe-500-small.toml
```

Append a model-size comparison row:

```bash
uv run python -m llm.scaling \
  --checkpoint tracks/from-scratch/checkpoints/tinyshakespeare-bpe-500-small.pt \
  --summary tracks/from-scratch/runs/summaries/observations.csv \
  --output tracks/from-scratch/runs/summaries/scaling.csv
```

## Leverage Quickstart

The leverage evaluator scores saved predictions only. It does not run models,
start RunPod, call APIs, download weights, or fetch datasets.

Evaluate saved predictions:

```bash
uv run python -m llm.leverage.evaluate \
  --tasks tracks/leverage/evals/leverage-smoke.jsonl \
  --tasks tracks/leverage/evals/project-judgment.jsonl \
  --predictions tracks/leverage/runs/two-layer.example.jsonl \
  --output /tmp/leverage_scores.csv \
  --summary-output /tmp/leverage_summary.csv
```

Collect predictions through an OpenAI-compatible API:

```bash
export OPENAI_BASE_URL="https://<provider>/v1"
export OPENAI_API_KEY="..."

uv run python -m llm.leverage.collect_openai \
  --tasks tracks/leverage/evals/leverage-smoke.jsonl \
  --tasks tracks/leverage/evals/project-judgment.jsonl \
  --model <provider-model-id> \
  --model-label <local_label> \
  --output tracks/leverage/runs/<local_label>.jsonl
```

The leverage track uses this loop:

```text
teacher generation -> structural filter -> model judge -> student training -> held-out eval
```

Reference docs:

- `tracks/leverage/model-spec.md`: target behavior and policy guard spec.
- `tracks/leverage/datasets/README.md`: reviewed instruction dataset lifecycle.
- `tracks/leverage/prompts/README.md`: generation seed prompts.
- `tracks/leverage/docs/sft-smoke.md`: first LoRA/SFT smoke plan.
- `tracks/leverage/runs/README.md`: which run records to read first.
- `tracks/leverage/runs/leverage-sft-smoke-runpod.md`: first RunPod SFT smoke result.
- `tracks/leverage/runs/leverage-sft-smoke-failure-triage.md`: failure classification.

## RunPod

RunPod is a paid external resource. Do not start a pod unless the objective,
cost ceiling, uploaded files, stopping condition, and cleanup plan are explicit.

The generic one-shot runner handles pod lifecycle, sync, remote commands, output
collection, and cleanup:

```bash
uv run python scripts/runpod/run_once.py --dry-run \
  --name llm-leverage-sft-smoke \
  --gpu-type 'NVIDIA GeForce RTX 3090' \
  --max-cost 0.8 \
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

Use `--dry-run` first. After any real RunPod job, verify no active pods remain:

```bash
/home/dev/bin/runpodctl get pod --allfields
```

## Project Shape

- `src/llm/`: reusable tokenizer, model, training, generation, evaluation, and
  leverage tooling.
- `scripts/`: one-shot local or RunPod orchestration.
- `tracks/`: project-specific configs, evals, datasets, prompts, and concise run
  notes.
- `tests/`: focused regression tests.

## Repository Rules

Do not commit datasets, checkpoints, secrets, or generated experiment artifacts.
The intended committed surface is source code, tests, project config, and
concise documentation.

Generated outputs belong under ignored paths such as `outputs/`,
`tracks/from-scratch/checkpoints/`, `tracks/from-scratch/data/processed/`, and
`tracks/leverage/sft/`.

Use `uv` for Python dependency and environment management.

## Verification

```bash
uv run ruff check .
uv run pytest
```

## License

MIT
