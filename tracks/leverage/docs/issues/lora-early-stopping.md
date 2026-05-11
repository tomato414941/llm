# LoRA Early Stopping

Status: implemented-minimal
Date: 2026-05-07
Updated: 2026-05-11

## Background

The current Qwen3.5-9B LoRA/SFT path is built around bounded 1-epoch baseline
runs. The trainer records training loss, final loss, timing, throughput, VRAM,
and GPU utilization. It stops on invalid loss values such as NaN or infinity.

This is enough for the current baseline goal: confirm that the dataset and
RunPod path can complete, then judge the adapter with held-out evaluations and
benchmarks.

## Problem

Training loss is useful, but it is not the same as task quality. A lower
training loss can mean the adapter is fitting the reviewed dataset better, but
it can also mean memorization or overfitting. The project should not treat
training loss alone as proof that the adapter improved.

Early stopping is currently not implemented. That is acceptable for short
1-epoch baseline runs, but it becomes a real gap if the project starts running:

- multiple epochs
- larger reviewed datasets
- more expensive GPU runs
- repeated model comparisons
- training runs where overfitting is likely before the planned epoch count ends

## Proposal

Keep the current 1-epoch Qwen3.5-9B baseline simple: no early stopping, but keep
recording loss and stop on NaN or infinity.

Before introducing multi-epoch LoRA/SFT runs, decide whether to add early
stopping based on one of these signals:

- validation loss on a held-out split
- held-out eval score
- benchmark score from a small fixed evaluation slice

Do not use training loss alone as the stopping signal unless the goal is only to
detect broken training.

## Non-Goals

- Do not add early stopping to the current 1-epoch baseline just to add a
  feature.
- Do not make benchmark execution part of every training step.
- Do not stop on small noisy loss changes without a patience window.
- Do not treat lower training loss as a capability claim.

## Adoption Criteria

Adopt early stopping when at least one of these becomes true:

- a planned LoRA/SFT run uses more than 1 epoch
- a run is expensive enough that stopping early would materially affect cost
- validation loss starts diverging from training loss in repeated runs
- held-out eval or benchmark scores degrade while training loss improves

The first implementation should be minimal: a validation split, an evaluation
interval, a patience value, and a clear metric written to the run note.

## Implementation

Minimal plumbing was added on 2026-05-11:

- `[early_stopping]` config table, disabled by default.
- Tail validation split when `early_stopping.enabled=true`.
- Validation-loss checks every `early_stopping.eval_every_steps`.
- Stop after `early_stopping.patience` checks without an improvement greater
  than `early_stopping.min_delta`.
- Metrics and notes fields for validation rows, validation checks, best
  validation loss, final validation loss, and whether early stopping triggered.

The existing 1-epoch smoke and Qwen3.5-9B configs keep early stopping disabled.
Enable it only for multi-epoch or materially expensive runs.
