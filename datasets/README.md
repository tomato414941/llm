# Datasets

This directory stores committed, reviewed dataset source files that are small
enough to inspect. Large generated data, raw model outputs, checkpoints, and
private data must stay out of git.

Raw instruction-generation outputs belong under
`experiments/leverage/instruction-outputs/`. Do not edit those files into shape
and treat them as datasets. Promote only selected rows manually into
`datasets/reviewed-instructions/` after review.

## `reviewed-instructions/leverage-v0.jsonl`

Reviewed instruction/answer examples for the leverage track. This is the
versioned source data that can later be exported into SFT training JSONL.
The current file is bootstrap data for validating the pipeline; it is not meant
to become the primary source of capability through manual polishing.

Each row contains:

- `id`: stable unique identifier
- `source_prompt_id`: source prompt id from `prompts/` when applicable
- `category`: broad capability area
- `messages`: chat-style SFT messages with `system`, `user`, and `assistant`
  turns
- `review`: lightweight provenance and acceptance notes

These rows are reviewed source data, not proof of model improvement. They
should only be exported for training after a held-out evaluation target is
selected. For scale, prefer teacher generation, structural filtering, model
judging, and held-out evaluation over manual row-by-row curation.

Promotion from raw outputs is explicit and row-by-row, but it should stay sparse:

1. Start from a raw generated answer under
   `experiments/leverage/instruction-outputs/`.
2. Verify the answer against `datasets/reviewed-instructions/review-rules.md`.
3. Rewrite or reject anything that is incorrect, generic, private, or detached
   from the source prompt.
4. Add only accepted rows to `reviewed-instructions/leverage-v0.jsonl` with
   stable provenance in `source_prompt_id` and `review`.

Validate reviewed instruction files before adding or exporting them:

```bash
uv run python -m llm.leverage.validate_reviewed-instructions \
  datasets/reviewed-instructions/leverage-v0.jsonl
```

See `datasets/reviewed-instructions/review-rules.md` for the review boundary
between raw generated outputs and accepted reviewed instructions.

Export a training JSONL file after validation:

```bash
uv run python -m llm.leverage.export_reviewed-instructions --overwrite
```

The default output is `data/sft/leverage_v0.train.jsonl`. Files under
`data/sft/` are generated training inputs and are ignored by git.
