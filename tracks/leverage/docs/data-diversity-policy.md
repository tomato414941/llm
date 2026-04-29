# Data Diversity Policy

Date: 2026-04-29

## Purpose

Define how reviewed instruction data should stay diverse while the dataset grows
from smoke-test size to the first `Qwen/Qwen3.5-9B` readiness run.

This document is an operating policy, not a taxonomy research project. The goal
is to prevent obvious data collapse without slowing down data growth.

## Decision

Use a small set of human-readable metadata and lightweight duplicate checks:

- Keep `capability` as the primary distribution-control axis.
- Add `task_shape` as an observational tag, not a hard gate.
- Track source and model provenance so one generator or judge does not dominate
  silently.
- Start with string-based near-duplicate checks before adding embedding-based
  clustering or selection.

Do not use `task_shape` as a target-count system yet. Revisit that after the
300-row readiness dataset exists.

## Why Diversity Matters

Instruction data can fail even when every row looks individually acceptable. A
dataset that repeats the same ability, task form, generator style, or project
context will mostly teach that narrow pattern.

For this project, diversity exists to answer:

- Are we training broad instruction following, or only project-agent behavior?
- Are generated rows surviving review across all planned capabilities?
- Are output styles, task forms, and failure modes varied enough for held-out
  evals to be meaningful?
- Are we distilling one provider or model's habits too strongly?

## External Basis

Public instruction-tuning projects usually combine coarse task categories with
simple deduplication and source controls:

- InstructGPT grouped API prompts into broad use cases such as generation,
  open QA, closed QA, brainstorming, chat, rewriting, summarization,
  classification, extraction, and other. It also used heuristic deduplication
  and limits per user or organization.
- Databricks Dolly 15k asked human contributors to create examples in broad
  categories derived from InstructGPT, plus a free-form category.
- Self-Instruct and Alpaca used seed instructions to generate larger synthetic
  datasets, then filtered invalid or overly similar instructions.
- FLAN emphasized task diversity, template diversity, and mixed prompt settings
  such as zero-shot, few-shot, and chain-of-thought. It also notes that task and
  category definitions differ across projects and are not easily collapsed into
  one universal ontology.

References:

- InstructGPT: https://cdn.openai.com/papers/Training_language_models_to_follow_instructions_with_human_feedback.pdf
- Databricks Dolly 15k: https://huggingface.co/datasets/databricks/databricks-dolly-15k
- Self-Instruct: https://arxiv.org/abs/2212.10560
- Stanford Alpaca: https://crfm.stanford.edu/2023/03/13/alpaca
- FLAN Collection: https://arxiv.org/abs/2301.13688

## Axes

### capability

`capability` is the primary axis. It describes the main ability a row trains or
evaluates. Its definitions live in `tracks/leverage/runs/data-category-design.md`.

Use `capability` for target counts and readiness planning.

### task_shape

`task_shape` describes the form of the task, independent of the primary
capability.

Initial values:

| task_shape | meaning |
| --- | --- |
| direct_answer | Answer directly without a substantial transformation or plan. |
| explanation | Explain a concept, result, or reason. |
| comparison | Compare options, tradeoffs, or alternatives. |
| rewrite | Rewrite, improve, shorten, or restyle text. |
| extraction_transformation | Extract, normalize, summarize, or convert information. |
| planning | Produce steps, a plan, or an execution sequence. |
| debugging | Diagnose or fix an error, inconsistency, or failure. |
| implementation | Produce code, config, commands, or a concrete artifact. |
| decision | Make or justify a choice under constraints. |

`task_shape` is observational for now. It should help spot repeated forms, not
block otherwise useful reviewed rows.

### provenance

Track generator, judge, edit, and historical source fields when available. This
prevents a dataset from quietly becoming a single-model style imitation.

### near-duplicate signal

Use near-duplicate checks as review aids. They should identify rows to inspect,
not delete rows automatically.

Start with string similarity because it is cheap, local, and easy to inspect.
Embedding checks can be added later if string checks miss meaningful semantic
duplicates.

## Current Operating Rules

- Keep adding reviewed rows toward the 300-row readiness target.
- Prefer underrepresented `capability` areas when selecting candidates for
  generation or manual recovery.
- Do not promote rows only because they fill a count.
- Do not let project-specific agent-policy examples dominate the dataset.
- Do not require every `capability` and `task_shape` combination to exist.
- Do not use held-out eval prompts as training-generation seeds.
- Treat high similarity as "review this", not "reject this".
- Revisit `task_shape` after 300 reviewed rows and decide whether it should
  remain observational or become part of mix planning.
