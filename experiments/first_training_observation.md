# First Training Observation

## Purpose

Learn how to read a small training run before changing model size or tuning
hyperparameters.

This note is about observation, not optimization.

## Command Shape

```bash
uv run python -m llm.train \
  --tokens data/processed/tinyshakespeare_bpe_500.pt \
  --checkpoint checkpoints/mini_gpt.pt \
  --metrics-output experiments/runs/tinyshakespeare_bpe_500.csv
```

Use smaller values such as `--max-iters 100` or `--eval-interval 20` when the
goal is only to inspect behavior.

## What To Watch

- Does train loss go down over time?
- Does validation loss move in the same direction as train loss?
- Does validation loss flatten or rise while train loss keeps falling?
- Does perplexity fall enough to make the generated samples less random?
- Are samples still mostly copying local patterns instead of showing broader
  structure?

## Reading The Result

If both train and validation loss fall, the model is learning usable structure
from the data.

If train loss falls but validation loss does not, the run may be overfitting or
the validation split may be too small or unrepresentative.

If neither loss falls, first check data loading, tokenizer compatibility, block
size, and whether the model is actually receiving enough varied batches.

Generated text should be treated as a qualitative signal only. Loss and
perplexity are the primary signals for this stage.

## Do Not Do Yet

- Do not run a large hyperparameter search.
- Do not compare many model sizes at once.
- Do not treat a single generated sample as the main result.
- Do not commit checkpoints or raw datasets.

## Follow-Up Questions

- At what point does validation loss stop improving?
- Does BPE make the sample quality improve faster than char-level tokenization?
- How much context is the model actually using?
