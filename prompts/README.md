# Prompts

This directory stores committed input prompts for leverage-track data generation.
These files are inputs, not model outputs.

## `leverage_training_seed_v0.jsonl`

Seed prompts for generating candidate instruction-tuning examples with an
OpenAI-compatible provider such as OpenRouter. Keep this file small and reviewed.
Generated outputs should be written under `experiments/leverage/` first, then
reviewed before any row is promoted into a training dataset.

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
