# Data Diversity Policy

Date: 2026-04-29

## Purpose

Define how reviewed instruction data should stay diverse while the dataset grows
from smoke-test size toward serious `Qwen/Qwen3.5-9B` capability-seeking runs.

This document is an operating policy, not a taxonomy research project. The goal
is to prevent obvious data collapse without slowing down data growth.

## Decision

Adopt recurring patterns from public instruction-tuning work as the default
policy for this project:

- Use broad task categories, following InstructGPT and Dolly.
- Track task-form diversity, following FLAN's emphasis on template and prompt
  setting diversity.
- Use simple near-duplicate checks, following InstructGPT and Self-Instruct.
- Track source and model provenance, following InstructGPT's source controls.

For our schema, the adopted policy maps to:

- `capability`: primary distribution-control axis.
- `task_shape`: observational task-form tag, not a hard gate.
- provenance fields: generator, judge, edit, and historical source tracking.
- near-duplicate signal: review aid, starting with string similarity.

See `tracks/leverage/docs/reviewed-instruction-mix-plan.md` for the current
reviewed-data mix plan.

Do not invent a project-specific ontology when a common external pattern covers
the need. Do not use `task_shape` as a target-count system yet. Revisit that
only if evaluation evidence shows `capability` is not enough to explain data
coverage or regressions.

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

## Adopted External Patterns

This project adopts the common public pattern of coarse task categories, task
form diversity, simple deduplication, and source controls:

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

## Adopted Mapping

The project follows external practice by default:

| external pattern | project field or rule | reason |
| --- | --- | --- |
| InstructGPT and Dolly broad use-case categories | `capability` | Keep distribution planning coarse and human-readable. |
| FLAN task and template diversity | `task_shape` | Observe whether task forms are collapsing within a capability. |
| InstructGPT source limits | provenance summaries | Avoid silent domination by one source, generator, judge, or editor. |
| InstructGPT and Self-Instruct heuristic deduplication | near-duplicate signal | Catch obvious repetition before training. |
| FLAN warning that categories differ by project | no universal ontology | Avoid over-designing a taxonomy beyond current use. |

When this document conflicts with a project-specific preference, use the
external pattern unless local evaluation evidence shows that it fails here.

## Axes

### capability

`capability` is the primary axis. It describes the main ability a row trains or
evaluates. Its definitions live in `tracks/leverage/docs/data-category-design.md`.

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

- Keep adding reviewed rows toward the 3,000+ row capability-seeking target.
- Prefer underrepresented `capability` areas when selecting candidates for
  generation or manual recovery.
- Do not promote rows only because they fill a count.
- Do not let project-specific agent-policy examples dominate the dataset.
- Do not require every `capability` and `task_shape` combination to exist.
- Do not use held-out eval prompts as training-generation seeds.
- Treat high similarity as "review this", not "reject this".
- Keep `task_shape` observational unless evaluation evidence shows it needs to
  become part of mix planning.
