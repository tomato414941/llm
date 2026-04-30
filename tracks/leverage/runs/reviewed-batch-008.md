# Reviewed Instruction Batch 008

Batch 008 starts from the batch 007 failure mode: accepted rows were mostly good,
but many prompts were exact duplicates and could not be promoted.

## Seeds

Added 200 seeds:

- `reasoning`: 80
- `coding`: 45
- `tool_use`: 30
- `knowledge_qa`: 20
- `summarization_transformation`: 15
- `instruction_following`: 10

The mix prioritizes the remaining reviewed deficits, especially `reasoning`,
while still adding coverage for the smaller deficits.

## Pre-Generation Duplicate Report

Ran the exact-duplicate seed report before any teacher-model call:

```text
selected_seed_count,200
duplicate_seed_count,0
duplicate_row_count,0
```

This checks the full generation user prompt against the selected batch and the
existing reviewed instruction dataset.

## Next Step

Generate candidates for `lt_seed_571` through `lt_seed_770`, then run the
existing filter, judge, and promotion gates.

