# Reviewed Instruction Growth Plan

Date: 2026-04-29

## Decision

Grow the reviewed instruction dataset toward a general-purpose instruction
dataset, not a project-specific agent policy dataset.

Project-judgment examples remain useful, but they are only one slice of the
dataset. They should not define the whole 300-row target.

## Target Mix

Target reviewed rows: 300

| category | target rows | purpose |
| --- | ---: | --- |
| general_instruction_following | 80 | Basic instruction following, format control, concise answers, and user-intent handling. |
| reasoning_comparison | 60 | Small reasoning tasks, tradeoff comparison, arithmetic, ordering, and consistency checks. |
| coding_repo_reasoning | 50 | Code-reading, small implementation judgment, tests, and repository workflow. |
| tool_use_judgment | 40 | Deciding whether to use local tools, APIs, generation, evaluation, or training. |
| eval_measurement | 30 | Train/test separation, scoring contracts, baselines, and measurement limits. |
| resource_cost_judgment | 20 | Bounded GPU/API use, cleanup, cost caps, and checkpoint thinking. |
| project_specific_policy | 20 | Local leverage-track conventions that are useful but not the dataset center. |

## Rules

- Treat this file as the data-distribution plan for the next capability-seeking
  LoRA attempt.
- Do not use held-out eval prompts as training-generation seeds.
- Do not promote raw generated output directly into reviewed data.
- Use non-self judging before human review or manual promotion.
- Exclude label-only, duplicate, malformed, and environment-specific rows from
  the training export.
- Keep project-specific policy examples below the target allocation unless a
  later committed plan changes the dataset goal.

## Current State

Current reviewed rows: 17

The current reviewed dataset is enough for training-path smoke tests. It is not
large or general enough for a capability claim.

## Relation To Project-Judgment Notes

The project-judgment notes explain a local failure-analysis thread from the
first LoRA smoke. They are historical run records and specific eval notes. They
are not the distribution plan for the 300-row reviewed dataset.

Use project-judgment failures as one source of seeds for the
`project_specific_policy`, `eval_measurement`, `resource_cost_judgment`, and
`coding_repo_reasoning` slices only when they fit the target mix above.
