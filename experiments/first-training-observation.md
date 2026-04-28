# First Training Observation

## Purpose

Learn how to read a small training run before changing model size or tuning
hyperparameters.

This note is about observation, not optimization.

## Command Shape

```bash
uv run python -m llm.train \
  --tokens data/processed/tinyshakespeare_bpe_500.pt \
  --checkpoint checkpoints/mini-gpt.pt \
  --metrics-output experiments/runs/tinyshakespeare-bpe-500.csv
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

## First Run

Command:

```bash
uv run python -m llm.train \
  --tokens data/processed/tinyshakespeare_bpe_500.pt \
  --checkpoint checkpoints/first-observation.pt \
  --metrics-output experiments/runs/first-observation.csv \
  --max-iters 100 \
  --eval-interval 20 \
  --eval-iters 5 \
  --block-size 32 \
  --batch-size 32 \
  --embedding-dim 32 \
  --num-heads 4 \
  --num-layers 2 \
  --generate-tokens 200 \
  --top-k 20
```

Observed metrics:

- parameters: 58,804
- train loss: 6.2159 -> 5.0818
- validation loss: 6.2180 -> 5.1068
- train perplexity: 500.63 -> 161.06
- validation perplexity: 501.68 -> 165.15

Reading:

Both train and validation loss fell steadily during this short run. This is a
basic sanity check that the data pipeline, tokenizer payload, batching, and
Transformer training loop are wired correctly.

The generated sample was still mostly local character and word-shape patterns.
That is expected for a short 100-step run on a tiny model. The main result here
is not sample quality; it is that loss and validation loss move in the expected
direction.

## Bitter Lesson Follow-Up

The first follow-up is not a tokenizer tweak, sampling tweak, or architecture
change. The next baseline keeps data, tokenizer, model shape, and sampling fixed
and increases training from 100 steps to 1,000 steps.

This follows the Bitter Lesson bias: prefer simple methods that benefit from
more compute before adding hand-designed complexity.

Result:

- 100-step observation validation perplexity: 160.17
- 1,000-step observation validation perplexity: 43.08

Reading:

The same small Transformer improves materially when given more training. The
pipeline is therefore not blocked by data loading, tokenization, or architecture
at this stage. The next scaling decision should compare more training against a
modest model-size increase, not local sampling tweaks.

## Medium 1k Capacity Follow-Up

The next baseline keeps the same data, tokenizer, learning rate, prompt, seed,
sampling, and 1,000 training steps, but increases model capacity:

- block size: 32 -> 64
- embedding dim: 32 -> 64
- layers: 2 -> 4
- parameters: 58,804 -> 267,892

Result:

- 1,000-step small validation perplexity: 43.08
- 1,000-step medium validation perplexity: 30.45

Reading:

Increasing capacity improves validation perplexity at the same step count. The
train/validation gap is larger than the small model, so capacity is helping but
overfitting risk is starting to matter. This still supports the Bitter Lesson
direction: simple scale improves the baseline before any local hand-designed
tweaks are needed.
