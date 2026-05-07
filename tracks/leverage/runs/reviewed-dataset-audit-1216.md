# Reviewed Dataset Audit 1216

Date: 2026-05-07

## Goal

Check whether `surface-constraint-batch-001` reduced the reviewed dataset's short-answer bias before the next Qwen3.5-9B training decision.

## Scope

- Reviewed dataset: `tracks/leverage/datasets/reviewed-instructions/bootstrap.jsonl`
- Total reviewed rows: 1216
- Previous rows: 1098
- New surface-constraint rows: 118 (`instr_1101` through `instr_1218`)

## Answer Length

All reviewed rows:

- Median words: 22.0
- Average words: 58.9
- At least 140 words: 115
- At least 260 words: 88
- At least 320 words: 70
- `<=50` words: 976

Previous rows only:

- Median words: 19.0
- Average words: 30.2
- At least 140 words: 22
- At least 260 words: 10
- At least 320 words: 8
- `<=50` words: 953

Surface-constraint batch only:

- Median words: 345.5
- Average words: 325.9
- At least 140 words: 93
- At least 260 words: 78
- At least 320 words: 62
- `<=50` words: 23

## Interpretation

The batch did what it was meant to do: it added a meaningful long-answer and exact-constraint signal. The reviewed dataset is still short-answer heavy overall, but the long-answer count moved from 10 rows at `>=260` words to 88 rows.

The remaining imbalance is by capability. Long answers are now mostly in `instruction_following`, `summarization_transformation`, `reasoning`, and `knowledge_qa`. `coding` and `tool_use` still have no `>=260` word accepted answers.

## Contract Check

All promoted surface-constraint rows include the full user-side instruction contract:

- `Output format:`
- `Constraints:`

This matters because the student model must see the same visible constraints that the generator saw.

## Current Risk

Training now is reasonable if the next run is treated as a baseline over the current dataset, not as proof that all capability areas have balanced long-form coverage.

If the goal is specifically long-form constraint following across capability areas, the next data batch should target `coding` and `tool_use` instead of adding more general long-form instruction-following rows.

## Verification

- `uv run python -m llm.leverage.summarize_capabilities`
- One-off answer-length audit over reviewed JSONL
