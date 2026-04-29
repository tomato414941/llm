# Data Category Design

Date: 2026-04-29

## Decision

Use one shared top-level capability taxonomy across:

- seed prompts
- reviewed instruction rows
- held-out eval tasks

Do not create separate top-level category systems for each layer. Separate
systems would make it hard to compare seed supply, reviewed-data coverage, and
eval coverage.

## Naming

`category` is the current JSONL field name. It is broad and can mean several
things.

For design discussions, use `capability_area` to mean the shared top-level
ability being trained or evaluated. Existing files may continue to use the
field name `category` until a separate schema cleanup is justified.

## Shared Capability Areas

| capability_area | definition |
| --- | --- |
| instruction_following | Follow explicit user constraints, output formats, brevity requirements, and refusal to over-answer. |
| reasoning_comparison | Compare options, do small calculations, preserve ordering, and explain tradeoffs. |
| coding | Understand code or repository tasks, propose small changes, choose tests, and avoid unrelated edits. |
| knowledge_qa | Answer factual questions with appropriate specificity and uncertainty. |
| summarization_transformation | Summarize, rewrite, extract, normalize, or transform text without changing meaning. |
| tool_use_judgment | Decide whether to use local tools, external APIs, generation, evaluation, or training. |
| evaluation_critique | Identify measurement flaws, scoring contracts, baselines, leakage, and evaluation limits. |
| resource_cost_judgment | Reason about bounded compute/API use, cost caps, runtime limits, checkpoints, and cleanup. |
| project_specific_policy | Local leverage-track conventions that should remain a small slice, not the dataset center. |

## Layer-Specific Metadata

The shared `capability_area` should be comparable across layers, but each layer
still needs its own metadata.

Seed prompts:

- `purpose`
- `output_format`
- `constraints`
- optional difficulty or source note

Reviewed instruction rows:

- `source_prompt_id`
- `review.status`
- `review.notes`
- training exclusion flags when needed

Held-out eval tasks:

- `scoring`
- `response_format`
- `expected_behavior`
- `difficulty`
- `tags`

## Rules

- Design new seed batches from `capability_area`, not from project-specific run
  history.
- Keep eval prompts held out from training-generation seeds.
- Use project-specific examples only when they fit the planned allocation.
- Treat existing `category` values as legacy labels until mapped to
  `capability_area`.
- If an audit needs remapping, change the mapping rule first, then rerun the
  count. Do not manually override individual rows after seeing the result.

## Relation To Growth Plan

`reviewed-instruction-growth-plan.md` defines the 300-row target mix. This file
defines the shared category semantics that should govern seed, reviewed-data,
and eval coverage.
