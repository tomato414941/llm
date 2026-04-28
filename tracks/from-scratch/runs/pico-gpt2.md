# pico GPT-2

## Purpose

Make the current tiny decoder-only Transformer feel closer to GPT-2 without
turning this repository into an experiment platform.

## Changes

- Use GELU in the feed-forward block.
- Tie token embedding and language-model head weights.
- Add optional warmup plus cosine learning-rate decay.
- Keep the tinyshakespeare BPE setup and small run config.

## Run

```bash
uv run python -m llm.train \
  --config tracks/from-scratch/configs/pico-gpt2-tinyshakespeare.toml
```

## Observation

Fill this after a local run.

- validation loss:
- validation perplexity:
- sample:

## Reading

This stage is about matching the shape of GPT-2 at pico scale. It is not trying
to reproduce GPT-2 capability or WebText-scale training.
