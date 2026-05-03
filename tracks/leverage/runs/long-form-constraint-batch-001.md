# Long-Form Constraint Batch 001

Date: 2026-05-03

## Purpose

Check whether the existing seed schema can support long-form surface-constraint
data without adding `constraint_tags`.

This run used the existing seed fields only:

- `capability`
- `purpose`
- `output_format`
- `constraints`

## Inputs

- Seeds: `lt_seed_1471` through `lt_seed_1490`
- Seed count: 20
- Generation pool: default random OpenRouter generator pool
- Judge pool: default random OpenRouter judge pool with self-judge exclusion
- Random seed: `1490`
- Reasoning: `reasoning_effort=none`, `exclude_reasoning=true`

## Raw Artifacts

Raw artifacts are intentionally ignored and remain under:

```text
tracks/leverage/runs/instruction-outputs/
```

Files:

- `long-form-constraint-batch-001-raw.jsonl`
- `long-form-constraint-batch-001-filter.csv`
- `long-form-constraint-batch-001-filter-summary.csv`
- `long-form-constraint-batch-001-candidates.jsonl`
- `long-form-constraint-batch-001-judgments.jsonl`
- `long-form-constraint-batch-001-judgments.csv`
- `long-form-constraint-batch-001-judgments-summary.csv`

## Generation Result

- generated rows: 20
- raw rows: 20
- finish reason `stop`: 20
- generation errors: 0
- observed generation time: about 4.5 minutes
- generation prompt tokens: 1,767
- generation completion tokens: 10,494

Generator distribution:

| generator | rows |
| --- | ---: |
| `qwen3-6-plus-openrouter` | 7 |
| `claude-sonnet-4-6-openrouter` | 4 |
| `glm-5-1-openrouter` | 4 |
| `gpt-5-5-openrouter` | 4 |
| `deepseek-v4-pro-openrouter` | 1 |

## Filter Result

- total rows: 20
- `needs_judge`: 18
- `reject`: 2

Rejects:

| seed | generator | issues |
| --- | --- | --- |
| `lt_seed_1477` | `glm-5-1-openrouter` | `json_markdown_fence`; `invalid_json` |
| `lt_seed_1482` | `qwen3-6-plus-openrouter` | `punctuation_forbidden`; `word_count_not_120` |

## Judge Result

- judged rows: 18
- accepted rows: 15
- needs edit rows: 3
- self-judge rows: 0
- observed judge time: about 1 minute

Decision distribution:

| decision | rows |
| --- | ---: |
| `accept` | 15 |
| `needs_edit` | 3 |

Judge distribution:

| judge | rows |
| --- | ---: |
| `claude-sonnet-4-6-openrouter` | 5 |
| `gpt-5-4-openrouter` | 3 |
| `deepseek-v4-pro-openrouter` | 2 |
| `glm-5-1-openrouter` | 2 |
| `gpt-5-5-openrouter` | 2 |
| `kimi-k2-6-openrouter` | 2 |
| `qwen3-6-plus-openrouter` | 2 |

Needs-edit rows:

| seed | generator | judge | reason |
| --- | --- | --- | --- |
| `lt_seed_1481` | `qwen3-6-plus-openrouter` | `gpt-5-4-openrouter` | Extra opening paragraph violated the exact section/paragraph structure. |
| `lt_seed_1484` | `qwen3-6-plus-openrouter` | `gpt-5-5-openrouter` | Added unsupported workflow details while expanding notes. |
| `lt_seed_1489` | `claude-sonnet-4-6-openrouter` | `kimi-k2-6-openrouter` | Added an unsupported sixth checklist item and may be below the word target. |

## Promotion Result

Accepted rows were promoted after an extra local constraint spot check covering
minimum word count, exact word count, forbidden punctuation, uppercase-only
letters, forbidden terms, JSON validity, and Markdown JSON fences.

- promoted rows: 15
- reviewed dataset size after promotion: 1,098 rows
- SFT export size after promotion: 1,098 rows
- promoted id range: `instr_1086` through `instr_1100`

Promoted seeds:

```text
lt_seed_1471
lt_seed_1472
lt_seed_1473
lt_seed_1474
lt_seed_1475
lt_seed_1476
lt_seed_1478
lt_seed_1479
lt_seed_1480
lt_seed_1483
lt_seed_1485
lt_seed_1486
lt_seed_1487
lt_seed_1488
lt_seed_1490
```

## Interpretation

The existing schema was sufficient to run the batch. The pipeline could generate,
filter, and judge these rows without adding `constraint_tags`.

The weak point is auditability, not generation. The filter can catch a few
deterministic failures, but distribution questions such as "how many accepted
rows exercise long minimum length plus punctuation constraints" still require
reading or ad hoc string searches over `constraints`.

Do not add `constraint_tags` yet based on this run alone. The next useful check
is whether promoting or planning a larger long-form constraint batch becomes
hard to audit with natural-language `constraints` only.
