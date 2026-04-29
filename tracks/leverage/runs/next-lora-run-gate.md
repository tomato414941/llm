# Next LoRA Run Gate

Date: 2026-04-29

## Decision

Do not run another capability-seeking LoRA until the data and eval scale meet
the thresholds below.

The completed RunPod LoRA smoke proved the training path works. It did not prove
that the current reviewed dataset improves the student model.

## Minimum Gate

Before the next LoRA run:

- Reviewed instruction rows: at least 300
- Project-judgment eval tasks: at least 100
- Holdout eval tasks: at least 30
- Label-only, duplicate, and malformed reviewed rows: excluded from the
  training export
- Holdout eval prompts: not used as teacher-generation seeds for the training
  slice

## Success Criteria

The next LoRA run is useful only if:

- Overall pass rate is at least the base student pass rate.
- Project-judgment pass rate improves over the base student pass rate.
- General `leverage-smoke` capabilities do not materially regress.
- RunPod cleanup is verified after the run.

## Rationale

The current reviewed dataset is adequate for a wiring smoke, but still too
small for a capability claim. Continuing to tune the student model on small,
hand-shaped examples would bias the project toward brittle rule injection.

The next iteration should scale reviewed examples through the existing
generation, non-self judgment, and manual-promotion loop before spending another
paid GPU run on model weights.

## Next Work

Follow `reviewed-instruction-mix-plan.md` for the dataset distribution. Use
failed project-judgment cases only as one seed source, not as the center of the
300-row dataset. Promote only reviewed rows into the training dataset, then
re-check the thresholds above before launching another LoRA run.
