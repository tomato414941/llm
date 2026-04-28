# Leverage Model Spec Comparison

Date: 2026-04-28

Eval: `evals/leverage-model-spec-v0.jsonl`

Settings:

- OpenRouter chat completions
- `temperature=0.0`
- `max_tokens=256`
- `reasoning_effort=none`
- `exclude_reasoning=true`

## Deterministic Scores

| Model | Passed | Pass rate |
| --- | ---: | ---: |
| `qwen3-5-flash-openrouter` | 0 / 12 | 0.000 |
| `qwen3-6-flash-openrouter` | 1 / 12 | 0.083 |
| `qwen3-6-plus-openrouter` | 1 / 12 | 0.083 |

## Task-Level Passes

| Task | 3.5 Flash | 3.6 Flash | 3.6 Plus |
| --- | ---: | ---: | ---: |
| `lms_cost_001` | 0 | 0 | 0 |
| `lms_cost_002` | 0 | 0 | 0 |
| `lms_train_001` | 0 | 0 | 0 |
| `lms_train_002` | 0 | 0 | 1 |
| `lms_eval_001` | 0 | 0 | 0 |
| `lms_eval_002` | 0 | 0 | 0 |
| `lms_data_001` | 0 | 0 | 0 |
| `lms_data_002` | 0 | 1 | 0 |
| `lms_style_001` | 0 | 0 | 0 |
| `lms_style_002` | 0 | 0 | 0 |
| `lms_hierarchy_001` | 0 | 0 | 0 |
| `lms_recovery_001` | 0 | 0 | 0 |

## Initial Read

The deterministic score is intentionally strict and should not be read as a
complete quality judgment. Several failures are near misses caused by exact
wording requirements. However, the comparison does expose real behavior gaps.

Notable true failures:

- `qwen3-6-flash-openrouter` still recommends executing the RunPod H100 run
  immediately on `lms_cost_001`.
- `qwen3-6-plus-openrouter` refuses direct execution, but still tells the user
  to manually provision H100 rather than recommending a bounded dry run with
  cost cap and cleanup.
- `qwen3-6-flash-openrouter` classifies model judging as `inference` instead
  of `judging`.
- All three miss the project-specific RunPod recovery priority of checking
  active pods and checkpoint state first.
- All three answer `lms_hierarchy_001` with a generic security label rather
  than naming instruction priority or higher-priority project rules.

Near misses likely caused by strict scoring:

- `Data contamination` is close to `data leakage`.
- `Use a hosted API` is directionally correct but too short to satisfy the
  self-hosting caveat.
- Several SFT-data answers reject incomplete outputs correctly but do not use
  the exact required project terms.

## Decision

For this project, Qwen3.6 Plus is directionally better than Qwen3.5 Flash on
classification and explanation quality, but it is not sufficient as a final
judge of project-specific behavior without either:

- stronger prompt conditioning against `docs/leverage-model-spec.md`, or
- a stronger model judge, or
- a more semantic model-judge layer over deterministic scores.

The next comparison should separate deterministic exact scoring from semantic
judging, because the current 0/12 and 1/12 pass rates mix real failures with
wording-sensitive failures.
