# Datasets

This directory stores committed, reviewed dataset source files that are small
enough to inspect. Large generated data, raw model outputs, checkpoints, and
private data must stay out of git.

## `reviewed_instructions/leverage_v0.jsonl`

Reviewed instruction/answer examples for the leverage track. This is the
versioned source data that can later be exported into SFT training JSONL.

Each row contains:

- `id`: stable unique identifier
- `source_prompt_id`: source prompt id from `prompts/` when applicable
- `category`: broad capability area
- `messages`: chat-style SFT messages with `system`, `user`, and `assistant`
  turns
- `review`: lightweight provenance and acceptance notes

These rows are reviewed source data, not proof of model improvement. They
should only be exported for training after a held-out evaluation target is
selected.

Validate reviewed instruction files before adding or exporting them:

```bash
uv run python -m llm.leverage.validate_reviewed_instructions \
  datasets/reviewed_instructions/leverage_v0.jsonl
```

See `datasets/reviewed_instructions/review_rules.md` for the review boundary
between raw generated outputs and accepted reviewed instructions.

Export a training JSONL file after validation:

```bash
uv run python -m llm.leverage.export_reviewed_instructions --overwrite
```

The default output is `data/sft/leverage_v0.train.jsonl`. Files under
`data/sft/` are generated training inputs and are ignored by git.
