# Datasets

This directory stores committed dataset candidates that are small enough to
review. Large generated data, raw model outputs, checkpoints, and private data
must stay out of git.

## `sft_candidates/leverage_sft_v0.jsonl`

Reviewed supervised fine-tuning candidates for the leverage track.

Each row contains:

- `id`: stable unique identifier
- `source_prompt_id`: source prompt id from `prompts/` when applicable
- `category`: broad capability area
- `messages`: chat-style SFT messages with `system`, `user`, and `assistant`
  turns
- `review`: lightweight provenance and acceptance notes

These rows are candidates, not proof of model improvement. They should only be
used for training after a held-out evaluation target is selected.
