# llm

Small LLM learning lab.

This project is a from-scratch path toward understanding decoder-only language
models. The current main route is:

1. train a BPE tokenizer
2. prepare token tensors
3. train a small Transformer language model
4. generate text from a checkpoint

The goal is not hyperparameter tuning yet. The goal is to keep the implementation
small enough to read, run, and change while building the core pieces of a GPT-style
model.

## Setup

```bash
uv sync --extra dev
```

## Current Workflow

Put local text data under `data/raw/`. Data files are intentionally ignored by git.

Train a BPE tokenizer:

```bash
uv run python -m llm.tokenize \
  --input data/raw/tinyshakespeare.txt \
  --output data/processed/bpe_tokenizer.json \
  --vocab-size 500
```

Prepare reusable token tensors:

```bash
uv run python -m llm.prepare_data \
  --input data/raw/tinyshakespeare.txt \
  --tokenizer data/processed/bpe_tokenizer.json \
  --output data/processed/tinyshakespeare_bpe_500.pt
```

Train the current model:

```bash
uv run python -m llm.train \
  --tokens data/processed/tinyshakespeare_bpe_500.pt \
  --checkpoint checkpoints/mini_gpt.pt \
  --metrics-output experiments/runs/tinyshakespeare_bpe_500.csv
```

`--metrics-output` is optional. When set, eval steps are appended as CSV rows with
train/validation loss and perplexity.

You can also run training from a TOML config. CLI arguments override config
values, which is useful for smoke runs:

```bash
uv run python -m llm.train \
  --config configs/tinyshakespeare_bpe_500_small.toml
```

Resume a training run from a checkpoint that includes optimizer and RNG state:

```bash
uv run python -m llm.train \
  --config configs/tinyshakespeare_bpe_500_small.toml \
  --resume checkpoints/tinyshakespeare_bpe_500_small.pt
```

Generate from a checkpoint:

```bash
uv run python -m llm.generate \
  --checkpoint checkpoints/mini_gpt.pt \
  --prompt "KING:" \
  --max-new-tokens 200 \
  --seed 1337 \
  --samples 2
```

Evaluate a checkpoint on the validation split:

```bash
uv run python -m llm.evaluate \
  --checkpoint checkpoints/mini_gpt.pt \
  --tokens data/processed/tinyshakespeare_bpe_500.pt
```

A useful observation loop is to evaluate the checkpoint first, then generate a
few seeded samples with the same checkpoint.

Write a checkpoint observation report:

```bash
uv run python -m llm.observe \
  --checkpoint checkpoints/first_observation.pt \
  --tokens data/processed/tinyshakespeare_bpe_500.pt \
  --prompt "KING:" \
  --output experiments/observations/first_observation.md \
  --summary-output experiments/summaries/observations.csv \
  --eval-iters 5 \
  --batch-size 32 \
  --max-new-tokens 200 \
  --temperature 1.0 \
  --top-k 20 \
  --seed 1337 \
  --samples 3
```

Run the same observation flow across a fixed JSONL prompt set:

```bash
uv run python -m llm.observe \
  --checkpoint checkpoints/first_observation.pt \
  --tokens data/processed/tinyshakespeare_bpe_500.pt \
  --prompt-file eval_prompts/tinyshakespeare.jsonl \
  --output experiments/observations/first_observation.md \
  --summary-output experiments/summaries/observations.csv
```

Or use the run config for the same observation settings:

```bash
uv run python -m llm.observe \
  --config configs/tinyshakespeare_bpe_500_small.toml
```

Append or update a model-size comparison row:

```bash
uv run python -m llm.scaling \
  --checkpoint checkpoints/tinyshakespeare_bpe_500_small.pt \
  --summary experiments/summaries/observations.csv \
  --output experiments/summaries/scaling.csv
```

## Project Shape

- `src/llm/tokenizer.py`: char and BPE tokenizers.
- `src/llm/tokenize.py`: trains and saves a BPE tokenizer.
- `src/llm/prepare_data.py`: encodes text into saved token tensors.
- `src/llm/models/`: Transformer components and language model.
- `src/llm/train.py`: training loop, loss/perplexity reporting, checkpoint saving.
- `src/llm/checkpoint.py`: checkpoint loading and validation.
- `src/llm/config.py`: TOML config loading for reproducible runs.
- `src/llm/generate.py`: checkpoint loading and sampling.
- `src/llm/evaluate.py`: checkpoint evaluation on prepared token data.
- `src/llm/observe.py`: checkpoint evaluation, seeded generation, and report writing.
- `src/llm/scaling.py`: scaling comparison CSV generation.
- `tests/`: focused tests for reusable model and tokenizer behavior.

## Repository Rules

Do not commit datasets, checkpoints, secrets, or generated experiment artifacts.
The intended committed surface is source code, tests, project config, and concise
documentation.

Use `uv` for Python dependency and environment management.

## Verification

```bash
uv run ruff check .
uv run pytest
```

## License

MIT
