# Leverage SFT Smoke Diff

This report compares raw decoded scoring with Qwen final-response scoring.

## Summary

| model | raw passed | qwen-final passed | total |
| --- | ---: | ---: | ---: |
| `qwen3-0.6b-base` | 3 | 8 | 30 |
| `qwen3-0.6b-lora-smoke` | 2 | 8 | 30 |

## Recovered By Qwen Final Parse

- `qa_capital_france`
- `instruction_json`
- `reasoning_order`
- `coding_sql_count`
- `qa_author`
- `instruction_lowercase`
- `pj_repo_001`

## Adapter Only

- none

## Base Only

- `summary_mission`

## Both Pass

- `summary_runpod`
- `coding_python_function`

## Both Fail

- `qa_water_freezing`
- `instruction_bullets`
- `reasoning_arithmetic`
- `pj_exp_001`
- `pj_exp_002`
- `pj_exp_003`
- `pj_cost_001`
- `pj_cost_002`
- `pj_cost_003`
- `pj_track_001`
- `pj_track_002`
- `pj_track_003`
- `pj_loss_001`
- `pj_loss_002`
- `pj_loss_003`
- `pj_repo_002`
- `pj_repo_003`
- `pj_eval_001`
- `pj_eval_002`
- `pj_eval_003`

## Next Data Target

Prioritize `both_fail` project-judgment tasks before another training run.
