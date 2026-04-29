# llm

Small LLM learning lab.

This repository has two tracks:

- `tracks/leverage/`: evaluate, adapt, and operate existing open models under
  practical constraints.
- `tracks/from-scratch/`: implement and train small decoder-only language models
  to understand the mechanics.

See `ROADMAP.md` for goals, non-goals, and RunPod policy. See `tracks/README.md`
for directory roles.

## Setup

```bash
uv sync --extra dev
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

- `tracks/leverage/README.md`: current mainline and source-of-truth map.
- `tracks/leverage/model-spec.md`: target behavior and policy guard spec.
- `tracks/leverage/datasets/README.md`: reviewed instruction dataset lifecycle.
- `tracks/leverage/prompts/README.md`: generation seed prompts.
- `tracks/leverage/docs/sft-smoke.md`: first LoRA/SFT smoke plan.
- `tracks/leverage/runs/README.md`: which run records to read first.
- `tracks/leverage/runs/leverage-sft-smoke-runpod-59-qwen35-08b-success.md`:
  current RunPod SFT smoke result.

## From-Scratch Quickstart

The from-scratch track is secondary right now. Use it for mechanics, not for
claims about practical model quality.

```bash
uv run python -m llm.train \
  --config tracks/from-scratch/configs/tinyshakespeare-bpe-500-small.toml

uv run python -m llm.observe \
  --config tracks/from-scratch/configs/tinyshakespeare-bpe-500-small.toml
```

## RunPod

RunPod is a paid external resource. Do not start a pod unless the objective,
cost ceiling, uploaded files, stopping condition, and cleanup plan are explicit.

The generic one-shot runner handles pod lifecycle, sync, remote commands, output
collection, and cleanup:

```bash
uv run python scripts/runpod/run_once.py --dry-run \
  --name llm-leverage-sft-smoke \
  --secure-cloud \
  --gpu-type 'NVIDIA A40' \
  --template-id runpod-torch-v280 \
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

The runner expects `runpodctl` v2 and uses `runpodctl pod ...` commands.
Use `--dry-run` first. After any real RunPod job, verify no active pods remain:

```bash
/home/dev/bin/runpodctl pod list -o json
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
