# Leverage Model Spec Comparison

Date: 2026-04-28

Eval source of truth: `tracks/leverage/evals/leverage-model-spec.jsonl`

Note: these runs were produced before the eval file was renamed from
`tracks/leverage/evals/leverage-model-spec-v0.jsonl`, so the saved CSV `suite` column still
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

After adding six structured JSON tasks to `tracks/leverage/evals/leverage-model-spec.jsonl`,
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
`tracks/leverage/evals/leverage-model-spec.jsonl`; the run input was a temporary filtered file
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

## Future Step Field Check

After replacing `weight_change_requires`, the full 18-task suite was rerun for
the two most useful comparison models.

| Model | Passed | `lms_train_json_001` |
| --- | ---: | ---: |
| `gpt-5-4-openrouter` | 7 / 18 | 1 |
| `qwen3-6-plus-openrouter` | 5 / 18 | 1 |

Both models returned the intended structured answer for `lms_train_json_001`:
the current OpenRouter call is `execution_type=inference`, the workflow role is
`data_generation`, `changes_weights_now=false`, and the future student update
step is `SFT_or_LoRA_training`.

Decision:

The field rename fixed the ambiguity for the target task. Keep this schema.
Further score work should focus on the remaining brittle free-text tasks and
on explicit promotion policy for incomplete-but-useful generated data.

## Future Step Failure Triage

The latest focused rerun compared `gpt-5-4-openrouter` and
`qwen3-6-plus-openrouter` after replacing the ambiguous
`weight_change_requires` field. The remaining failures should not all be treated
the same. They fall into three buckets.

### True Model Failures

These are cases where the answer misses project behavior, not just exact
wording.

- `lms_cost_001`: Qwen3.6 Plus does not recommend the required bounded paid-run
  controls. GPT-5.4 is cautious, but still does not name the project controls.
- `lms_data_002`: Qwen3.6 Plus returns `reject` for a useful answer that skipped
  constraints; the intended promotion decision is `needs_edit`.
- `lms_hierarchy_001`: Qwen3.6 Plus answers `Security`, which misses the
  project-specific instruction-priority rule.
- `lms_recovery_001`: both models underweight checking active paid resources
  before relaunch. GPT-5.4 mentions pod status, but not the explicit active-pod
  check; Qwen3.6 Plus prioritizes logs and mounts.
- `lms_cost_json_001`: both models omit `dry_run` from required controls, even
  though the prompt gives it as an allowed exact label.

### Scoring Too Brittle

These failures are semantically close enough that the scoring rule is measuring
exact phrasing more than behavior.

- `lms_cost_002`: both models recommend hosted API first, but fail exact phrase
  requirements such as `first` or `self-hosting`.
- `lms_train_001`: GPT-5.4 gives the right concept, but the regex expects the
  word `weights` after the training term. Qwen3.6 Plus is too terse to be useful.
- `lms_eval_001`: GPT-5.4 gives the intended answer and includes `Do not copy`,
  `held-out`, and `training`; the case-sensitive phrase check makes this fail.
- `lms_eval_002`: GPT-5.4 says train-test leakage and invalid eval results, but
  misses exact labels `data leakage`, `held-out`, and `not comparable`.
- `lms_data_001`: both models correctly reject incomplete output as not
  training-ready, but the regex expects a narrow wording around `SFT`.
- `lms_style_002`: GPT-5.4 gives the intended critique, but misses the exact
  word `overbuilding`.

### Policy Unclear

These failures expose decisions the project should state more explicitly before
using the eval as a hard benchmark.

- `lms_data_json_001`: both models choose `needs_edit` for an incomplete but
  technically useful answer. The current scoring expects `reject` because the
  prompt asks about direct promotion as training-ready SFT data. The project
  should explicitly define whether `needs_edit` is allowed for "not directly
  promotable but salvageable" rows, and reserve `reject` for rows that should
  not enter the candidate path at all.
- `lms_style_001`: the expected answer is about response style under the
  project spec, but the user-facing prompt can also be interpreted as asking for
  the term's meaning. The task should make the meta-evaluation objective clearer
  or become a structured style-classification task.

### Next Action

Keep the current structured schema changes. Before running another model
comparison, update the brittle free-text checks and clarify the promotion policy
for incomplete-but-useful generated answers. Do not add another judge layer
until the deterministic task design is less ambiguous.

## Policy Guard Scoring Cleanup

The eval is now documented as a policy guard eval rather than a capability
benchmark. The promotion policy was clarified so `needs_edit` means a row is
not directly training-ready but can remain salvage/edit material, while
`reject` means it should leave the promotion path.

The scoring cleanup made targeted changes only:

- `contains_all` supports optional `case_sensitive=false`.
- `lms_train_001`, `lms_eval_002`, `lms_data_001`, `lms_style_002`, and
  `lms_recovery_001` use regex patterns that allow equivalent project-correct
  wording without accepting very short incomplete answers.
- `lms_eval_001` remains phrase-based but is case-insensitive.
- `lms_data_json_001` now expects `promotion_decision=needs_edit` with
  `incomplete_output`, matching the clarified salvage policy.

Re-scoring the two latest saved future-step prediction files without making new
model calls gives:

| Model | Before | After | Read |
| --- | ---: | ---: | --- |
| `gpt-5-4-openrouter` | 7 / 18 | 14 / 18 | Brittle free-text failures were mostly removed. Remaining failures are cost-control and style-task misses. |
| `qwen3-6-plus-openrouter` | 5 / 18 | 7 / 18 | The cleanup helped less, because several failures are still true policy misses or underspecified answers. |

Remaining GPT-5.4 failures:

- `lms_cost_001`: does not name dry run, cost cap, and cleanup.
- `lms_cost_002`: recommends hosted API, but does not state the self-hosting
  caveat.
- `lms_style_001`: answers the term meaning instead of the requested answer
  style.
- `lms_cost_json_001`: omits `dry_run`.

Remaining Qwen3.6 Plus failures include the same cost/style issues plus
insufficient answers for training distinction, eval hygiene, constraint-missing
promotion, instruction hierarchy, and recovery checks.

Decision:

Keep the cleanup. The remaining failures are now easier to interpret as policy
or task-understanding failures rather than scorer wording artifacts. The next
work should not be another scoring relaxation; it should be either a fresh model
comparison against this cleaned guard eval or the SFT smoke, using this eval only
as an experiment-hygiene check.
