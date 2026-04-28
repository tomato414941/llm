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

Validate candidate files before adding or training on them:

```bash
uv run python -m llm.leverage.validate_sft_candidates \
  datasets/sft_candidates/leverage_sft_v0.jsonl
```

See `datasets/sft_candidates/review_rules.md` for the review boundary between
raw generated outputs and accepted SFT candidates.

Export a training JSONL file after validation:

```bash
uv run python -m llm.leverage.export_sft_dataset --overwrite
```

The default output is `data/sft/leverage_sft_v0.train.jsonl`. Files under
`data/sft/` are generated training inputs and are ignored by git.
