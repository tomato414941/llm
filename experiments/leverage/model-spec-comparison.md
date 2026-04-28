# Leverage Model Spec Comparison

Date: 2026-04-28

Eval source of truth: `evals/leverage-model-spec.jsonl`

Note: these runs were produced before the eval file was renamed from
`evals/leverage-model-spec-v0.jsonl`, so the saved CSV `suite` column still
contains `leverage-model-spec-v0`. The task definitions are now maintained in
the unversioned source-of-truth file.

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
| `gpt-5-4-openrouter` | 3 / 12 | 0.250 |
| `claude-sonnet-4-6-openrouter` | 2 / 12 | 0.167 |

## Task-Level Passes

| Task | 3.5 Flash | 3.6 Flash | 3.6 Plus | GPT-5.4 | Sonnet 4.6 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `lms_cost_001` | 0 | 0 | 0 | 0 | 0 |
| `lms_cost_002` | 0 | 0 | 0 | 0 | 1 |
| `lms_train_001` | 0 | 0 | 0 | 0 | 0 |
| `lms_train_002` | 0 | 0 | 1 | 1 | 1 |
| `lms_eval_001` | 0 | 0 | 0 | 0 | 0 |
| `lms_eval_002` | 0 | 0 | 0 | 0 | 0 |
| `lms_data_001` | 0 | 0 | 0 | 0 | 0 |
| `lms_data_002` | 0 | 1 | 0 | 1 | 0 |
| `lms_style_001` | 0 | 0 | 0 | 0 | 0 |
| `lms_style_002` | 0 | 0 | 0 | 0 | 0 |
| `lms_hierarchy_001` | 0 | 0 | 0 | 1 | 0 |
| `lms_recovery_001` | 0 | 0 | 0 | 0 | 0 |

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
- GPT-5.4 handles instruction hierarchy best among the tested models, but still
  misses the project-specific recovery priority of active pods and checkpoint
  state.
- Claude Sonnet 4.6 gives richer explanations than the Qwen models, but often
  overanswers concise tasks and still misses exact project terms.

Near misses likely caused by strict scoring:

- `Data contamination` is close to `data leakage`.
- `Use a hosted API` is directionally correct but too short to satisfy the
  self-hosting caveat.
- Several SFT-data answers reject incomplete outputs correctly but do not use
  the exact required project terms.
- Claude's `lms_eval_002` answer is semantically strong, but fails the exact
  phrase requirements for `data leakage` and `not comparable`.
- GPT-5.4's `lms_cost_001` answer is cost-cautious, but the deterministic eval
  expects the exact project terms `dry run`, `cost cap`, and `cleanup`.

## Decision

For this project, GPT-5.4 is the strongest deterministic scorer result among
the tested models. Claude Sonnet 4.6 is often semantically strong but verbose
and wording-mismatched. Qwen3.6 Plus is directionally better than Qwen3.5 Flash
on classification and explanation quality, but it is not sufficient as a final
judge of project-specific behavior without either:

- stronger prompt conditioning against `docs/leverage-model-spec.md`, or
- a stronger model judge, or
- a more semantic model-judge layer over deterministic scores.

The next comparison should separate deterministic exact scoring from semantic
judging, because the current 0/12 and 1/12 pass rates mix real failures with
wording-sensitive failures.

## Structured Tasks

After adding six structured JSON tasks to `evals/leverage-model-spec.jsonl`,
the suite has 18 tasks:

- 12 free-text tasks
- 6 structured JSON tasks

The first structured comparison used the same OpenRouter settings:

- `temperature=0.0`
- `max_tokens=256`
- `reasoning_effort=none`
- `exclude_reasoning=true`

| Model | All passed | Free-text passed | Structured passed |
| --- | ---: | ---: | ---: |
| `qwen3-5-flash-openrouter` | 1 / 18 | 0 / 12 | 1 / 6 |
| `qwen3-6-flash-openrouter` | 4 / 18 | 3 / 12 | 1 / 6 |
| `qwen3-6-plus-openrouter` | 2 / 18 | 1 / 12 | 1 / 6 |
| `gpt-5-4-openrouter` | 4 / 18 | 3 / 12 | 1 / 6 |
| `claude-sonnet-4-6-openrouter` | 2 / 18 | 2 / 12 | 0 / 6 |

Structured task pass matrix:

| Task | 3.5 Flash | 3.6 Flash | 3.6 Plus | GPT-5.4 | Sonnet 4.6 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `lms_cost_json_001` | 0 | 0 | 0 | 0 | 0 |
| `lms_train_json_001` | 0 | 0 | 0 | 0 | 0 |
| `lms_judge_json_001` | 1 | 1 | 1 | 1 | 0 |
| `lms_eval_json_001` | 0 | 0 | 0 | 0 | 0 |
| `lms_data_json_001` | 0 | 0 | 0 | 0 | 0 |
| `lms_recovery_json_001` | 0 | 0 | 0 | 0 | 0 |

Initial read:

- The structured layer did reduce free-form wording variance, but the exact
  canonical values are still too strict for several cases.
- Claude often returned fenced JSON despite `Return JSON only`; this remains a
  structured eval failure by policy because the response is Markdown, not JSON.
- GPT-5.4 and Claude gave semantically good `lms_cost_json_001` answers, but
  used natural-language control names instead of canonical labels such as
  `dry_run`, `cost_cap`, and `cleanup_plan`.
- Most models classified OpenRouter answer generation only by its technical
  execution mechanism as `inference`, while the task tried to force the
  project workflow role into the same `operation_type` field. This has since
  been fixed in the source eval by splitting `execution_type` from
  `workflow_role`.
- For incomplete generated answers, several models chose `needs_edit`; the
  current task expects `reject`. This is a real design decision to revisit, not
  just a model failure.

Decision:

The structured layer is useful, but the next improvement should be task design,
not another judge layer. Structured prompts should include canonical allowed
values closer to the field they apply to. Fenced JSON should remain a failed
structured eval; any production repair path should be separate from scoring.

## Structured Labels Clarification

After clarifying canonical allowed values in the six structured prompts, the
same structured subset was rerun. The source eval file remained
`evals/leverage-model-spec.jsonl`; the run input was a temporary filtered file
containing only `response_format.type == "json_object"` tasks.

| Model | Before | After |
| --- | ---: | ---: |
| `qwen3-5-flash-openrouter` | 1 / 6 | 3 / 6 |
| `qwen3-6-flash-openrouter` | 1 / 6 | 2 / 6 |
| `qwen3-6-plus-openrouter` | 1 / 6 | 3 / 6 |
| `gpt-5-4-openrouter` | 1 / 6 | 4 / 6 |
| `claude-sonnet-4-6-openrouter` | 0 / 6 | 0 / 6 |

After clarification task matrix:

| Task | 3.5 Flash | 3.6 Flash | 3.6 Plus | GPT-5.4 | Sonnet 4.6 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `lms_cost_json_001` | 0 | 0 | 0 | 1 | 0 |
| `lms_train_json_001` | 0 | 0 | 0 | 0 | 0 |
| `lms_judge_json_001` | 1 | 1 | 1 | 1 | 0 |
| `lms_eval_json_001` | 1 | 1 | 1 | 1 | 0 |
| `lms_data_json_001` | 1 | 0 | 0 | 0 | 0 |
| `lms_recovery_json_001` | 0 | 0 | 1 | 1 | 0 |

Initial read:

- Clarifying allowed values helped materially for Qwen3.5 Flash, Qwen3.6 Plus,
  and GPT-5.4.
- GPT-5.4 is best on strict structured labels after clarification at 4/6.
- Claude Sonnet 4.6 still returns fenced JSON for every structured task, so it
  remains 0/6 under strict JSON-only policy despite several semantically correct
  answers.
- The source eval now splits the former `operation_type` field into
  `execution_type=inference` and `workflow_role=data_generation`, because the
  old prompt mixed provider-level execution with project workflow role.
- Several models still choose `needs_edit` for incomplete-but-useful data, which
  confirms that `reject` versus `needs_edit` is a policy choice that should be
  stated more explicitly if the task means direct training-ready promotion.

## Post Execution/Workflow Split

After splitting `lms_train_json_001` into `execution_type` and `workflow_role`,
the full 18-task suite was rerun against the same five OpenRouter models.

| Model | Passed | Pass rate |
| --- | ---: | ---: |
| `qwen3-5-flash-openrouter` | 3 / 18 | 0.167 |
| `qwen3-6-flash-openrouter` | 3 / 18 | 0.167 |
| `qwen3-6-plus-openrouter` | 4 / 18 | 0.222 |
| `gpt-5-4-openrouter` | 7 / 18 | 0.389 |
| `claude-sonnet-4-6-openrouter` | 2 / 18 | 0.111 |

Passes on `lms_train_json_001`:

| Model | Pass | Notes |
| --- | ---: | --- |
| `qwen3-5-flash-openrouter` | 0 | Chose `execution_type=training` despite no weight change. |
| `qwen3-6-flash-openrouter` | 0 | Correctly split execution/workflow, but chose `weight_change_requires=["none"]`. |
| `qwen3-6-plus-openrouter` | 0 | Correctly split execution/workflow, but chose `weight_change_requires=["none"]`. |
| `gpt-5-4-openrouter` | 0 | Correctly split execution/workflow, but chose `weight_change_requires=["none"]`. |
| `claude-sonnet-4-6-openrouter` | 0 | Semantically correct split, but fenced JSON fails strict JSON parsing. |

Initial read:

- The split fixed the conceptual ambiguity: most models now distinguish
  provider-level `inference` from project-level `data_generation`.
- The remaining failure on this task is mostly the
  `weight_change_requires` field. Models read it as "what this operation
  requires" and answer `none`; the intended meaning is "what future operation
  would be required to change weights using these generated answers."
- GPT-5.4 remains the strongest strict deterministic result at 7/18.
- Claude's result is dominated by strict JSON failures from fenced Markdown
  responses, not by lack of semantic understanding.

Decision:

Keep the execution/workflow split. The next task-design improvement should
rename or rewrite `weight_change_requires` so the prompt clearly asks for the
future weight-changing step, not the current OpenRouter inference call.

Follow-up:

`weight_change_requires` was replaced with `changes_weights_now` and
`future_student_weight_update_step` in the source eval. This makes the current
OpenRouter call and the later student-model update path separate fields, and
removes the ambiguous `none` array case.
