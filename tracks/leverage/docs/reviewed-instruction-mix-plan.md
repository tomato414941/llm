# Reviewed Instruction Mix Plan

Date: 2026-04-29

## Purpose

Define the reviewed-instruction mix for `Qwen/Qwen3.5-9B` LoRA readiness and
pilot runs. This file answers what kind of reviewed data to add; the run gate is
defined in `tracks/leverage/docs/lora-sft-runpod.md`.

The dataset should be general-purpose instruction data, not a project-specific
agent-policy dataset. Project-judgment examples are useful, but they are only
one slice of the target mix.

## Baseline Mix

The 300-row readiness target used this baseline distribution. Keep the same
rough capability balance for 1,000-row and 3,000-row targets unless evaluation
failures show a clear reason to rebalance.

Use the shared `capability` semantics defined in
`tracks/leverage/docs/data-category-design.md`.

| capability | target rows | purpose |
| --- | ---: | --- |
| instruction_following | 80 | Basic instruction following, format control, concise answers, and user-intent handling. |
| reasoning | 80 | Small reasoning tasks, tradeoff comparison, arithmetic, ordering, and consistency checks. |
| coding | 50 | Code-reading, small implementation judgment, tests, and repository workflow. |
| knowledge_qa | 30 | Factual answers with appropriate specificity and uncertainty handling. |
| summarization_transformation | 25 | Summarization, rewrite, extraction, normalization, and format transformation. |
| tool_use | 35 | Deciding whether to use local tools, APIs, generation, evaluation, or training. |

Do not let project-specific policy examples dominate any larger dataset stage.

## Current State

Current reviewed rows: 1,216

The current reviewed dataset is enough for training-path smoke tests and a
small pilot LoRA. It is still below the 3,000+ row target for the first serious
capability-seeking run.

The first surface-constraint batch improved some long-answer behavior on a
small IFEval sample, but the 1,216-row adapter still regressed on full IFEval.
See `tracks/leverage/docs/qwen35-9b-adapter-regression-analysis.md`.

## Current Mix Adjustment

The next reviewed-data addition should not chase IFEval prompts directly. Add a
small generalized constraint-accounting slice that covers:

- word-count budgets
- keyword frequency
- letter frequency
- forbidden words
- JSON-only and exact-format answers
- case and section-count constraints

## Rules

- Do not use held-out eval prompts as training-generation seeds.
- Do not promote raw generated output directly into reviewed data.
- Use non-self judging before human review or manual promotion.
- Exclude label-only, duplicate, malformed, and environment-specific rows from
  the training export.
- Keep project-specific policy examples rare and classify them by the broader
  capability they exercise.
