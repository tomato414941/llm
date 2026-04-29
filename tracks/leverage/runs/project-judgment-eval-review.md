# Project Judgment Eval Review

Date: 2026-04-29

Status: historical eval-design note. This file documents a one-time review of
`project-judgment` eval tasks. It is not the distribution plan for the 300-row
reviewed instruction dataset.

Source eval: `tracks/leverage/evals/project-judgment.jsonl`

Run context: first Qwen3 0.6B LoRA/SFT smoke comparison in
`tracks/leverage/runs/leverage-sft-smoke-diff.md`.

## Purpose

This is a one-time eval-design review, not a recurring manual failure-triage
process. The goal is to remove early eval design noise before using
`project-judgment.jsonl` as a stable held-out comparison layer.

Do not manually classify every failed task after each future run. After this
eval stabilizes, review only regressions, new tasks, or tasks whose scoring
contract changes.

## Classification Rules

- `good_eval`: the task measures a useful behavior and the observed failure is
  a real model/data gap.
- `too_strict`: the behavior is useful, but the current deterministic scoring
  is too brittle for acceptable natural-language variants.
- `ambiguous_prompt`: the prompt or expected answer contract is underspecified.

## Summary

| classification | count | next action |
| --- | ---: | --- |
| `good_eval` | 13 | Keep as training/eval signal. Add data only for these behaviors. |
| `too_strict` | 4 | Keep task intent, revise scoring contract before using as training target. |
| `ambiguous_prompt` | 1 | Rewrite prompt before treating failures as model gaps. |

## Task Review

| task_id | category | classification | reason | next_action |
| --- | --- | --- | --- | --- |
| `pj_exp_001` | `experiment_judgment` | `good_eval` | Both models mishandled the promotion decision when the dataset shard changed. The task is clear and measures apples-to-apples experiment judgment. | Keep. Add examples about refusing incomparable wins. |
| `pj_exp_002` | `experiment_judgment` | `good_eval` | The target concept is controlled attribution. The adapter answer noticed comparison trouble but did not identify multiple changed variables or isolation. | Keep. Add data on one-variable experiment changes. |
| `pj_exp_003` | `experiment_judgment` | `good_eval` | The prompt gives explicit labels and asks for exactly one. Choosing `DROP` or rambling is a real failure. | Keep. Add decision-label examples. |
| `pj_cost_001` | `runpod_cost_awareness` | `good_eval` | The task requires a simple cost calculation and a no decision. Base computed cost but answered yes; adapter answered no but omitted projected cost. | Keep. Add cost-cap calculation examples. |
| `pj_cost_002` | `runpod_cost_awareness` | `good_eval` | The expected behavior is operationally important: stop idle paid compute while preserving state. Both outputs miss that. | Keep. Add idle-resource cleanup examples. |
| `pj_cost_003` | `runpod_cost_awareness` | `good_eval` | The task asks for cheap safeguards before a long run. The expected `smoke test` and `cost cap` concepts are central, not wording trivia. | Keep. Add pre-launch safeguard examples. |
| `pj_track_001` | `track_distinction` | `good_eval` | The prompt asks for one of three labels. Returning operations or a long rationale is a real classification failure. | Keep. Add workstream-label examples. |
| `pj_track_002` | `track_distinction` | `ambiguous_prompt` | The desired answer is engineering-vs-research separation, but the prompt does not explicitly ask for those labels. Natural paraphrases can miss the scoring terms. | Rewrite prompt to ask for the two labels explicitly. |
| `pj_track_003` | `track_distinction` | `good_eval` | The prompt asks for exactly the operations label. The model outputs are not close. | Keep. Add operations-label examples. |
| `pj_loss_001` | `loss_curve_interpretation` | `too_strict` | Both outputs contain the overfitting concept, but exact scoring requires only `overfitting`. This tests formatting more than diagnosis. | Change scoring from `exact` to a regex or contains check for `overfitting`. |
| `pj_loss_002` | `loss_curve_interpretation` | `good_eval` | The expected checks, learning rate and data pipeline, are core debugging actions. The observed outputs do not name them. | Keep. Add non-learning-run debug examples. |
| `pj_loss_003` | `loss_curve_interpretation` | `good_eval` | The task clearly asks yes/no plus moving average. Base never reaches a final answer; adapter answers yes. | Keep. Add noisy-metric trend examples. |
| `pj_repo_001` | `coding_repo_reasoning` | `good_eval` | Adapter passes after Qwen final parsing; base fails. The task is clear and useful. | Keep as baseline-vs-adapter signal. |
| `pj_repo_002` | `coding_repo_reasoning` | `too_strict` | The expected collaboration rule is valid, but the scoring requires `leave`, `README.md`, and `unchanged`. Some acceptable answers may say not to touch or not to revert. | Relax scoring to accept equivalent no-touch language. |
| `pj_repo_003` | `coding_repo_reasoning` | `good_eval` | The target checks, worktree status and tests, are explicit repo-safety behavior. The outputs miss them. | Keep. Add unfamiliar-repo workflow examples. |
| `pj_eval_001` | `eval_design` | `too_strict` | Both outputs identify exact-match scoring as inappropriate for a long essay, but fail because they omit `open-ended` and `brittle`. | Relax scoring or make the prompt request the flaw terms. |
| `pj_eval_002` | `eval_design` | `good_eval` | The task asks for one allowed scoring type. Both models choose `exact`, so the failure is real. | Keep. Add scoring-type selection examples. |
| `pj_eval_003` | `eval_design` | `too_strict` | Base identifies missing loss curves and repo reasoning, but omits `category imbalance`. The concept is close enough that strict phrase scoring may hide partial competence. | Consider JSON/field scoring with required missing categories plus imbalance label. |

## Immediate Changes Recommended

Do not train on all `both_fail` tasks indiscriminately.

Use this split:

- Add or regenerate training data for the 13 `good_eval` behaviors.
- Revise scoring for `pj_loss_001`, `pj_repo_002`, `pj_eval_001`, and
  `pj_eval_003` before using them as improvement targets.
- Rewrite `pj_track_002` before treating its failures as model failures.

## Applied Eval Fixes

Applied on 2026-04-29:

- `pj_loss_001`: changed exact scoring to regex scoring for the `overfitting`
  concept.
- `pj_repo_002`: changed phrase scoring to regex scoring that accepts equivalent
  no-touch/no-revert language for `README.md`.
- `pj_eval_001`: changed scoring to accept exact-match criticism for open-ended
  or long-form essay tasks.
- `pj_eval_003`: changed the prompt to ask for the imbalance and missing areas,
  and relaxed `category imbalance` to `imbalance`.
- `pj_track_002`: rewrote the prompt to explicitly ask for the engineering and
  research labels that should be kept separate.

After regenerating `leverage-sft-smoke-diff.md`, the Qwen final-response score
changed from `16 / 60` to `19 / 60`. `pj_loss_001` moved to `both_pass`, and
`pj_eval_001` moved to `base_only`. The remaining project-judgment failures are
now better treated as data/model gaps unless a future run exposes a regression
or a new ambiguity.

## Stable Operation Rule

Once the above fixes are made, stop doing full manual failure classification.
Future runs should be judged by baseline-vs-adapter deltas, regressions, and
category summaries. Manual review should be limited to changed eval tasks and
unexpected regressions.
