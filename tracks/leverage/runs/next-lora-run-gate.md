# Next LoRA Run Gate

Date: 2026-04-29

## Decision

Prioritize the `Qwen/Qwen3.5-9B` path over additional side-model smoke tests.
Do not run another capability-seeking LoRA until the data and eval scale meet
the thresholds below.

The completed RunPod LoRA smoke proved the training path works. It did not prove
that the current reviewed dataset improves the student model.

## Staged Data Targets

| reviewed rows | name | purpose |
| ---: | --- | --- |
| 300 | readiness run | Check that a 9B LoRA run is wired correctly and does not obviously collapse. |
| 1,000 | pilot LoRA | Look for an early improvement trend against the base 9B model. |
| 3,000+ | first serious 9B LoRA | First run large enough to treat as a real capability-seeking attempt. |
| 10,000+ | dataset v1 | Candidate scale for a serious reviewed instruction dataset. |

The 300-row target is not "full training." It is the minimum paid 9B readiness
gate. Treat 1,000 rows as a pilot and 3,000+ rows as the first serious
`Qwen/Qwen3.5-9B` LoRA target.

## Minimum 9B Readiness Gate

Before the next paid 9B LoRA run:

- Reviewed instruction rows: at least 300
- Project-judgment eval tasks: at least 100
- Holdout eval tasks: at least 30
- Label-only, duplicate, and malformed reviewed rows: excluded from the
  training export
- Holdout eval prompts: not used as teacher-generation seeds for the training
  slice

## Success Criteria

The next 9B LoRA run is useful only if:

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
paid GPU run on 9B model weights.

## Next Work

Follow `reviewed-instruction-mix-plan.md` for the dataset distribution. Use
failed project-judgment cases only as one seed source, not as the center of the
300-row dataset. Promote only reviewed rows into the training dataset, then
re-check the thresholds above before launching the 9B readiness run.
