# Reviewed Instruction Growth Plan

Date: 2026-04-29

## Decision

Grow the reviewed instruction dataset toward a general-purpose instruction
dataset, not a project-specific agent policy dataset.

Project-judgment examples remain useful, but they are only one slice of the
dataset. They should not define the whole 300-row target.

## Target Mix

Target reviewed rows: 300

The table uses the shared `capability` semantics defined in
`data-category-design.md`.

| capability | target rows | purpose |
| --- | ---: | --- |
| instruction_following | 80 | Basic instruction following, format control, concise answers, and user-intent handling. |
| reasoning | 80 | Small reasoning tasks, tradeoff comparison, arithmetic, ordering, and consistency checks. |
| coding | 50 | Code-reading, small implementation judgment, tests, and repository workflow. |
| knowledge_qa | 30 | Factual answers with appropriate specificity and uncertainty handling. |
| summarization_transformation | 25 | Summarization, rewrite, extraction, normalization, and format transformation. |
| tool_use | 35 | Deciding whether to use local tools, APIs, generation, evaluation, or training. |

## Rules

- Treat this file as the data-distribution plan for the next capability-seeking
  LoRA attempt.
- Do not use held-out eval prompts as training-generation seeds.
- Do not promote raw generated output directly into reviewed data.
- Use non-self judging before human review or manual promotion.
- Exclude label-only, duplicate, malformed, and environment-specific rows from
  the training export.
- Keep project-specific policy examples rare and classify them by the broader
  capability they exercise.

## Current State

Current reviewed rows: 17

The current reviewed dataset is enough for training-path smoke tests. It is not
large or general enough for a capability claim.

## Relation To Project-Judgment Notes

The project-judgment notes explain a local failure-analysis thread from the
first LoRA smoke. They are historical run records and specific eval notes. They
are not the distribution plan for the 300-row reviewed dataset.

Use project-judgment failures as one source of seeds only when they fit the
shared capability areas above. Project-specific policy, evaluation critique, and
resource-cost concerns are operational topics, not top-level capability areas.
