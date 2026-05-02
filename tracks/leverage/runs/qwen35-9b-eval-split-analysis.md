# Qwen3.5-9B Eval Split Analysis

Date: 2026-05-02

## Goal

Explain why the `Qwen/Qwen3.5-9B` LoRA adapter improved the small project-owned
eval but regressed on full IFEval.

This is a local artifact analysis. It did not run new paid inference or
training.

## Inputs

- Training run note:
  `tracks/leverage/runs/qwen35-9b-full-gradient-checkpointing.md`
- Full IFEval run note:
  `tracks/leverage/runs/lm-harness-ifeval-full.md`
- Training dataset:
  `tracks/leverage/datasets/reviewed-instructions/bootstrap.jsonl`
- Project-owned eval outputs:
  `outputs/leverage-sft-qwen35-9b/post-training-scores.csv`
  `outputs/leverage-sft-qwen35-9b/post-training-predictions.jsonl`
- Full IFEval outputs:
  `outputs/leverage-lm-harness-ifeval-full/.../results_*.json`

## Known Scores

Project-owned held-out eval:

| model | overall | leverage-smoke | project-judgment |
| --- | ---: | ---: | ---: |
| base | 15/30 | 9/12 | 6/18 |
| adapter | 17/30 | 9/12 | 8/18 |

Full IFEval:

| model | prompt strict | prompt loose | instruction strict | instruction loose |
| --- | ---: | ---: | ---: | ---: |
| base | 0.8410 | 0.8817 | 0.8885 | 0.9185 |
| adapter | 0.7819 | 0.8133 | 0.8501 | 0.8717 |

## Project-Owned Eval Delta

Only four project-owned tasks changed pass/fail status:

| task | suite | capability | change | reason |
| --- | --- | --- | --- | --- |
| `summary_mission` | leverage-smoke | summarization_transformation | fail -> pass | adapter included required phrase `two tracks` |
| `coding_sql_count` | leverage-smoke | coding | pass -> fail | adapter wrapped SQL in a fenced code block |
| `pj_exp_001` | project-judgment | reasoning | fail -> pass | adapter matched the expected dataset-shard blocking issue |
| `pj_exp_003` | project-judgment | reasoning | fail -> pass | adapter returned `REPEAT` exactly |

The net project-owned gain is therefore small and brittle: three wins and one
loss, for a net +2 tasks.

## Training-Data Overlap Signal

Two of the project-owned wins are close to explicit training rows:

- `pj_exp_003` asks for `SHIP`, `REPEAT`, or `DROP` when a promising ablation has
  only one seed and no held-out evaluation. Training row `instr_0016` from
  `lt_seed_053` teaches the same decision shape and answer: `REPEAT`.
- `pj_exp_001` asks whether a result should be promoted when it used a smaller
  dataset shard. Training row `instr_0210` from `lt_seed_373` teaches the same
  causal caution pattern: the dataset changed, so causality is not proven.

This means the project-owned improvement is not strong evidence of broad
capability. It is plausibly evidence that the adapter learned local project
phrasing and decision patterns.

## Output-Format Regression Signal

The adapter failed `coding_sql_count` by returning:

````text
```sql
SELECT COUNT(*) FROM users;
```
````

The base model returned:

```text
SELECT COUNT(*) FROM users;
```

The training data contains 104 assistant messages with fenced code blocks.
Those are concentrated in coding:

| slice | rows | fenced rows |
| --- | ---: | ---: |
| all training rows | 1083 | 104 |
| coding capability | 191 | 100 |
| implementation task_shape | 117 | 97 |

So the SQL regression is consistent with the adapter learning a code-answer
style that is often appropriate in the training data but wrong for strict
`SQL only` or exact-regex tasks.

## IFEval Limitation

The full IFEval run was executed without `--log-samples`, so the saved artifacts
contain aggregate metrics but not per-sample base-vs-adapter failures. That
means the exact IFEval regression mode cannot be classified from current
artifacts.

The aggregate regression is still meaningful because it is broad across both
prompt-level and instruction-level metrics:

- prompt strict: -0.0591
- prompt loose: -0.0684
- instruction strict: -0.0384
- instruction loose: -0.0468

## Interpretation

The two evals are measuring different things:

- The project-owned eval is small and partly close to the training distribution.
  It can detect whether the adapter learned local project conventions, but it
  is not a strong generalization test.
- IFEval is broader instruction-following coverage. The adapter regressed there,
  so it should be treated as an external guardrail failure.

Current conclusion: the full LoRA run worked mechanically, but the current
training set is too local and too formatting-biased to claim a general
instruction-following improvement.

## Next Step

Do not start another full LoRA run just to chase the small project-owned gain.

The next useful step is to improve eval observability before changing training:

- run a small IFEval sample comparison with `--log-samples`
- compare base and adapter failures manually
- classify whether regressions are formatting, verbosity, refusal, instruction
  omission, or semantic errors

That would be cost-bearing work and should be proposed for
`tracks/leverage/docs/execution-costs.md` before execution.
