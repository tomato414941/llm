# Data Category Decision

Date: 2026-04-29

## Decision

Use one shared top-level capability classification for seed prompts, reviewed
instruction rows, and held-out eval tasks.

Do not create separate top-level category systems for each layer. The point is
to compare seed supply, reviewed-data coverage, and eval coverage by the same
rough ability area.

## Why Categories Exist

Categories exist only as data-distribution control metadata. They answer:

- Which capability areas are over- or under-supplied at the seed layer?
- Which capability areas fail to survive generation, judging, and review?
- Which capability areas have reviewed training rows but too little held-out
  eval coverage?
- Which capability areas improve or regress after a LoRA run?

They are not labels to teach the model, and they are not quality scores. If we
stop using category counts for planning or analysis, the metadata should be
removed rather than maintained decoratively.

## Naming

`category` is the current JSONL field name. In this document, the intended
meaning is `capability_area`: the primary ability being trained or evaluated.
Existing files may keep the `category` field until a schema cleanup is worth the
extra churn.

## Primary Axis

The top-level axis is capability, not domain, format, lifecycle, difficulty,
source model, or operational topic.

Classify a row by asking:

```text
If this row disappeared, which capability area's coverage would shrink most?
```

Examples:

- A RunPod cost calculation is `resource_cost_judgment`, not `runpod`.
- A JSON-only answer task is `instruction_following`, not `json`.
- A held-out scoring-contract critique is `evaluation_critique`, not `eval`.
- A repository test-selection task is `coding`, not `repo`.

Other useful axes, such as output format, difficulty, domain, source model,
lifecycle stage, or review status, should stay as separate metadata when needed.

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

## Rules

- Design new seed batches from `capability_area`, not from project-specific run
  history.
- Keep eval prompts held out from training-generation seeds.
- Use project-specific examples only when they fit the planned allocation.
- Treat existing `category` values as legacy labels until mapped to
  `capability_area`.
- If an audit needs remapping, change the mapping rule first, then rerun the
  count. Do not manually override individual rows after seeing the result.
