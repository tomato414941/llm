# Reviewed Instruction Mix Plan

Date: 2026-04-29

## Purpose

Define the reviewed-instruction mix needed before the next capability-seeking
LoRA run. This file answers what kind of reviewed data to add; the run gate is
defined in `next-lora-run-gate.md`.

The dataset should be general-purpose instruction data, not a project-specific
agent-policy dataset. Project-judgment examples are useful, but they are only
one slice of the target mix.

## Target Mix

Target reviewed rows: 300

Use the shared `capability` semantics defined in `data-category-design.md`.

| capability | target rows | purpose |
| --- | ---: | --- |
| instruction_following | 80 | Basic instruction following, format control, concise answers, and user-intent handling. |
| reasoning | 80 | Small reasoning tasks, tradeoff comparison, arithmetic, ordering, and consistency checks. |
| coding | 50 | Code-reading, small implementation judgment, tests, and repository workflow. |
| knowledge_qa | 30 | Factual answers with appropriate specificity and uncertainty handling. |
| summarization_transformation | 25 | Summarization, rewrite, extraction, normalization, and format transformation. |
| tool_use | 35 | Deciding whether to use local tools, APIs, generation, evaluation, or training. |

## Current State

Current reviewed rows: 59

The current reviewed dataset is enough for training-path smoke tests. It is not
large or general enough for a capability claim.

## Rules

- Do not use held-out eval prompts as training-generation seeds.
- Do not promote raw generated output directly into reviewed data.
- Use non-self judging before human review or manual promotion.
- Exclude label-only, duplicate, malformed, and environment-specific rows from
  the training export.
- Keep project-specific policy examples rare and classify them by the broader
  capability they exercise.
