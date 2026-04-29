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

## Mutual Generation And Random Non-Self Judging

Executed on 2026-04-29 to reduce generator bias without paying for a full
all-pairs judge matrix.

Generator pool:

- `qwen3-5-flash-openrouter` (`qwen/qwen3.5-flash-02-23`)
- `gpt-5-4-openrouter` (`openai/gpt-5.4`)
- `claude-sonnet-4-6-openrouter` (`anthropic/claude-sonnet-4.6`)

Judge assignment rule:

- Pick one judge per generator.
- The judge must not be the generator.
- Random seed: `20260429`

Assignments:

| generator | judge |
| --- | --- |
| `qwen3-5-flash-openrouter` | `gpt-5-4-openrouter` |
| `gpt-5-4-openrouter` | `claude-sonnet-4-6-openrouter` |
| `claude-sonnet-4-6-openrouter` | `gpt-5-4-openrouter` |

Results:

| generator | judge | accept | needs_edit | reject |
| --- | --- | ---: | ---: | ---: |
| `qwen3-5-flash-openrouter` | `gpt-5-4-openrouter` | 1 | 9 | 3 |
| `gpt-5-4-openrouter` | `claude-sonnet-4-6-openrouter` | 3 | 8 | 2 |
| `claude-sonnet-4-6-openrouter` | `gpt-5-4-openrouter` | 2 | 8 | 3 |

New accepted rows promoted from this mutual pass:

- `lt_seed_054` -> `instr_0012`
- `lt_seed_057` -> `instr_0013`
- `lt_seed_058` -> `instr_0014`
- `lt_seed_060` -> `instr_0015`

`lt_seed_052` was also accepted from Claude generation, but it was already
covered by `instr_0011`, so it was not duplicated.

The reviewed dataset now contains 15 rows. SFT export was regenerated with 15
rows.

## Default Teacher/Judge Decision

Decision on 2026-04-29: retire `qwen3-5-flash-openrouter` as the default
instruction teacher or judge. Use the current OpenRouter Qwen model instead of
the older Flash model.

Reason:

- Qwen3.5 Flash is cheap, but this pass accepted only 1 of 13 generated rows.
- Low API price is not useful if most rows still require review, editing, or
  rejection.
- For reviewed SFT data, optimize for accepted rows per dollar, not raw tokens
  per dollar.

Default going forward:

- instruction teacher: `qwen3-6-plus-openrouter`
- instruction judge: `claude-sonnet-4-6-openrouter`

All default hosted calls use the OpenRouter-compatible API base URL. Qwen3.5
Flash may still be used explicitly for low-cost exploration, but it is not the
default path for reviewed training-data generation.

## Qwen3.6 Plus Teacher Check

Executed on 2026-04-29 with the same 13 project-judgment seeds.

Generator:

- `qwen3-6-plus-openrouter` (`qwen/qwen3.6-plus`)

Judge:

- `claude-sonnet-4-6-openrouter` (`anthropic/claude-sonnet-4.6`)

Structural filter:

- total: 13
- `needs_judge`: 13
- `response_too_long`: 4

Judge result:

- `accept`: 4
- `needs_edit`: 7
- `reject`: 2

Accepted rows:

- `lt_seed_052`: already covered by `instr_0011`
- `lt_seed_053`: promoted to `instr_0016`
- `lt_seed_057`: already covered by `instr_0013`
- `lt_seed_058`: already covered by `instr_0014`

Compared with Qwen3.5 Flash's 1/13 accept rate, Qwen3.6 Plus improved to 4/13
and produced one new unique reviewed instruction row. The reviewed dataset now
contains 16 rows.

## Per-Candidate Random Judge Rule

Decision on 2026-04-29: judge assignment should be random per candidate row,
not one random judge for all rows from the same generator.

Rule:

- Provide a judge candidate pool as repeated `label=model` values.
- For each candidate row, remove the generator's own label from the eligible
  judges.
- Pick one eligible judge with a fixed `--random-seed` for reproducibility.

This keeps cost bounded at one judge call per candidate while reducing the
chance that a whole generator's score is an artifact of one judge model.

Executed on 2026-04-29 for the Qwen3.6 Plus project-judgment candidates with
random seed `20260429`.

Judge pool:

- `qwen3-6-plus-openrouter` (`qwen/qwen3.6-plus`)
- `gpt-5-4-openrouter` (`openai/gpt-5.4`)
- `claude-sonnet-4-6-openrouter` (`anthropic/claude-sonnet-4.6`)

Qwen3.6 Plus generated every row, so Qwen was excluded and each row was judged
by either GPT-5.4 or Claude Sonnet 4.6.

Assignment/result:

| seed | judge | decision |
| --- | --- | --- |
| `lt_seed_051` | `gpt-5-4-openrouter` | `accept` |
| `lt_seed_052` | `claude-sonnet-4-6-openrouter` | `accept` |
| `lt_seed_053` | `claude-sonnet-4-6-openrouter` | `accept` |
| `lt_seed_054` | `gpt-5-4-openrouter` | `needs_edit` |
| `lt_seed_055` | `claude-sonnet-4-6-openrouter` | `reject` |
| `lt_seed_056` | `claude-sonnet-4-6-openrouter` | `needs_edit` |
| `lt_seed_057` | `gpt-5-4-openrouter` | `accept` |
| `lt_seed_058` | `gpt-5-4-openrouter` | `accept` |
| `lt_seed_059` | `gpt-5-4-openrouter` | `needs_edit` |
| `lt_seed_060` | `claude-sonnet-4-6-openrouter` | `needs_edit` |
| `lt_seed_061` | `claude-sonnet-4-6-openrouter` | `reject` |
| `lt_seed_062` | `gpt-5-4-openrouter` | `needs_edit` |
| `lt_seed_063` | `gpt-5-4-openrouter` | `reject` |

Summary:

- `accept`: 5
- `needs_edit`: 5
- `reject`: 3
- `gpt-5-4-openrouter` judged 7 rows
- `claude-sonnet-4-6-openrouter` judged 6 rows

New accepted row promoted:

- `lt_seed_051` -> `instr_0017`

`lt_seed_052`, `lt_seed_053`, `lt_seed_057`, and `lt_seed_058` were already
covered by existing reviewed rows. The reviewed dataset now contains 17 rows.
