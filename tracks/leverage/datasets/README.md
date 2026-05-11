# Datasets

This directory stores committed, reviewed dataset source files that are small
enough to inspect. Large generated data, raw model outputs, checkpoints, and
private data must stay out of git.

Raw instruction-generation outputs belong under
`tracks/leverage/runs/instruction-outputs/`. Do not edit those files into shape
and treat them as datasets. Promote only selected rows manually into
`tracks/leverage/datasets/reviewed-instructions/` after review.

## `reviewed-instructions/bootstrap.jsonl`

Reviewed instruction/answer examples for the leverage track. This is the
versioned source data that can later be exported into SFT training JSONL.
The current file is bootstrap data for validating the pipeline; it is not meant
to become the primary source of capability through manual polishing.

Each row contains:

- `id`: stable unique identifier
- `source_prompt_id`: source prompt id from `tracks/leverage/prompts/` when applicable
- `capability`: broad model capability
- `messages`: chat-style SFT messages with `system`, `user`, and `assistant`
  turns
- `review`: lightweight provenance and acceptance notes

These rows are reviewed source data, not proof of model improvement. They
should only be exported for training after a held-out evaluation target is
selected. For scale, prefer teacher generation, structural filtering, model
judging, and held-out evaluation over manual row-by-row curation.

Promotion from raw outputs is explicit and row-by-row, but it should stay sparse:

1. Start from a raw generated answer under
   `tracks/leverage/runs/instruction-outputs/`.
2. Verify the answer against `tracks/leverage/datasets/reviewed-instructions/review-rules.md`.
3. Rewrite or reject anything that is incorrect, generic, private, or detached
   from the source prompt.
4. Add only accepted rows to `reviewed-instructions/bootstrap.jsonl` with
   stable provenance in `source_prompt_id` and `review`.

Validate reviewed instruction files before adding or exporting them:

```bash
uv run python -m llm.leverage.validate_reviewed_instructions \
  tracks/leverage/datasets/reviewed-instructions/bootstrap.jsonl
```

See `tracks/leverage/datasets/reviewed-instructions/review-rules.md` for the review boundary
between raw generated outputs and accepted reviewed instructions.

Export a training JSONL file after validation:

```bash
uv run python -m llm.leverage.export_reviewed_instructions --overwrite
```

The default output is `tracks/leverage/sft/bootstrap.train.jsonl`. Files under
`tracks/leverage/sft/` are generated training inputs and are ignored by git.

## External SFT Data

External instruction datasets are not reviewed project data. Keep them separate
from `reviewed-instructions/`, record their license, and write generated SFT
exports under ignored `tracks/leverage/sft/` paths.

The first external full-dataset candidate is OpenOrca:

- Dataset: `Open-Orca/OpenOrca`
- License: MIT
- Export path: `tracks/leverage/sft/openorca.train.jsonl`
- Config: `tracks/leverage/configs/leverage-sft-openorca-full.toml`

Export command:

```bash
uv pip install datasets
uv run python -m llm.leverage.import_openorca --overwrite
```

For a quick local shape check, use `--limit`:

```bash
uv run python -m llm.leverage.import_openorca \
  --limit 10 \
  --output tracks/leverage/sft/openorca.sample.train.jsonl \
  --overwrite
```

Do not run the full OpenOrca training config until the export size is known and
the trainer has a safe large-dataset plan. The current SFT smoke trainer loads
all training rows into memory.
