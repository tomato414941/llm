# Project Judgment Seed Generation Plan

Date: 2026-04-29

Purpose: generate teacher-model candidate answers only for the 13 training
seeds added after the `project-judgment` eval review. These seeds target
behaviors classified as `good_eval` in
`tracks/leverage/runs/project-judgment-eval-review.md`.

This run must not use held-out eval prompts as training data. The selected seed
IDs are separate generation prompts:

- `lt_seed_051`
- `lt_seed_052`
- `lt_seed_053`
- `lt_seed_054`
- `lt_seed_055`
- `lt_seed_056`
- `lt_seed_057`
- `lt_seed_058`
- `lt_seed_059`
- `lt_seed_060`
- `lt_seed_061`
- `lt_seed_062`
- `lt_seed_063`

## Dry Run

Verified command shape without requiring an API key or writing outputs:

```bash
uv run python scripts/leverage/openai_compatible_instruction_once.py \
  --dry-run \
  --resume \
  --output tracks/leverage/runs/instruction-outputs/qwen3-5-flash-openrouter-project-judgment-seeds.jsonl \
  --seed-id lt_seed_051 \
  --seed-id lt_seed_052 \
  --seed-id lt_seed_053 \
  --seed-id lt_seed_054 \
  --seed-id lt_seed_055 \
  --seed-id lt_seed_056 \
  --seed-id lt_seed_057 \
  --seed-id lt_seed_058 \
  --seed-id lt_seed_059 \
  --seed-id lt_seed_060 \
  --seed-id lt_seed_061 \
  --seed-id lt_seed_062 \
  --seed-id lt_seed_063
```

The generated command uses:

- seeds: `tracks/leverage/prompts/leverage-training-seed-v0.jsonl`
- provider: OpenRouter-compatible API
- model: `qwen/qwen3.5-flash-02-23`
- model label: `qwen3-5-flash-openrouter`
- output: `tracks/leverage/runs/instruction-outputs/qwen3-5-flash-openrouter-project-judgment-seeds.jsonl`
- mode: `--resume`
- reasoning: `--reasoning-effort none --exclude-reasoning`

## Real Run

Run the same command without `--dry-run` after confirming
`~/.secrets/openrouter` is present.

Do not commit the raw generated output. It belongs under
`tracks/leverage/runs/instruction-outputs/`, which is ignored. Commit only a
short run note, filter summary, or reviewed dataset changes after inspection.

## Follow-Up

After generation:

1. Run structural filtering on the output.
2. Run model judging on candidates that pass structural checks.
3. Promote only reviewed accepted rows to
   `tracks/leverage/datasets/reviewed-instructions/bootstrap.jsonl`.
4. Re-export SFT data.
5. Consider a small LoRA smoke rerun only after the reviewed dataset changes.

## Run Result

Executed on 2026-04-29.

Raw output:

```text
tracks/leverage/runs/instruction-outputs/qwen3-5-flash-openrouter-project-judgment-seeds.jsonl
```

Structural filter:

- total: 13
- `needs_judge`: 13
- `response_too_long`: 4

GPT-5.4 judge result:

- `accept`: 1
- `needs_edit`: 9
- `reject`: 3

Accepted row promoted:

- `lt_seed_052` -> `instr_0011`

Rejected rows were not promoted. `needs_edit` rows were not promoted because
they require editing before they can be treated as reviewed SFT data.

SFT export was regenerated:

```text
tracks/leverage/sft/bootstrap.train.jsonl
```

The reviewed dataset now contains 11 rows.
