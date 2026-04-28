# Prompts

This directory stores committed input prompts for leverage-track data generation.
These files are inputs, not model outputs.

## `leverage-training-seed-v0.jsonl`

Seed prompts for generating candidate instruction-tuning examples with an
OpenAI-compatible provider such as OpenRouter. Keep this file small and reviewed.
Generated outputs should be written under
`experiments/leverage/instruction-outputs/` first, then reviewed before any row
is promoted into a committed dataset.

Required fields:

- `id`: stable unique identifier
- `category`: broad capability or project concern
- `purpose`: why this generated example would be useful
- `system_prompt`: provider-facing instruction
- `prompt`: user-facing input to answer
- `output_format`: expected answer shape
- `constraints`: list of required answer properties

This is intentionally separate from `evals/`. Eval files define scored tests.
Prompt files define generation inputs.

## Lifecycle Boundary

Prompt files are stable inputs for teacher-model generation. They should not
contain model answers, review decisions, scores, or SFT-ready chat rows.

The expected flow is:

```text
prompts/leverage-training-seed-v0.jsonl
  -> experiments/leverage/instruction-outputs/<run>.jsonl
  -> datasets/reviewed-instructions/leverage-v0.jsonl
  -> data/sft/leverage_v0.train.jsonl
```

Only the review step may promote raw generated answers into
`datasets/reviewed-instructions/`. Keep `source_prompt_id` stable so promoted
rows can be traced back to the prompt that produced them.
