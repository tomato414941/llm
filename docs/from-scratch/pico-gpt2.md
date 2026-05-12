# pico GPT-2

## Purpose

Make the current tiny decoder-only Transformer feel closer to GPT-2 without
turning this repository into an experiment platform.

## Changes

- Use GELU in the feed-forward block.
- Tie token embedding and language-model head weights.
- Add optional warmup plus cosine learning-rate decay.
- Keep the tinyshakespeare BPE setup.

## Run

```bash
uv run python -m llm.train \
  --tokens data/from-scratch/processed/tinyshakespeare_bpe_500.pt \
  --checkpoint runs/from-scratch/pico-gpt2-tinyshakespeare/checkpoint.pt \
  --metrics-output runs/from-scratch/pico-gpt2-tinyshakespeare/metrics.csv
```

## Observation

Fill this after a local run.

- validation loss:
- validation perplexity:
- sample:

## Reading

This stage is about matching the shape of GPT-2 at pico scale. It is not trying
to reproduce GPT-2 capability or WebText-scale training.
