# Leverage Post-Training Compare

Compares raw decoded responses with Qwen final-response parsing.

## Summary

| raw passed | qwen-final passed | total |
| ---: | ---: | ---: |
| 8 | 19 | 60 |

## Adapter Only

- `qa_author`
- `pj_repo_001`

## Base Only

- `qa_capital_france`
- `summary_mission`
- `pj_eval_001`

## Both Pass

- `summary_runpod`
- `instruction_json`
- `reasoning_order`
- `coding_python_function`
- `coding_sql_count`
- `instruction_lowercase`
- `pj_loss_001`

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
- `pj_loss_002`
- `pj_loss_003`
- `pj_repo_002`
- `pj_repo_003`
- `pj_eval_002`
- `pj_eval_003`

## Details

## coding

| model | suite | task | raw | qwen-final | status |
| --- | --- | --- | ---: | ---: | --- |
| `qwen3-0.6b-base` | `leverage-smoke` | `coding_python_function` | 1 | 1 | `pass` |
| `qwen3-0.6b-lora-smoke` | `leverage-smoke` | `coding_python_function` | 1 | 1 | `pass` |
| `qwen3-0.6b-base` | `leverage-smoke` | `coding_sql_count` | 0 | 1 | `parsed_only` |
| `qwen3-0.6b-lora-smoke` | `leverage-smoke` | `coding_sql_count` | 0 | 1 | `parsed_only` |

## coding

| model | suite | task | raw | qwen-final | status |
| --- | --- | --- | ---: | ---: | --- |
| `qwen3-0.6b-base` | `project-judgment` | `pj_repo_001` | 0 | 0 | `fail` |
| `qwen3-0.6b-lora-smoke` | `project-judgment` | `pj_repo_001` | 0 | 1 | `parsed_only` |
| `qwen3-0.6b-base` | `project-judgment` | `pj_repo_002` | 0 | 0 | `fail` |
| `qwen3-0.6b-lora-smoke` | `project-judgment` | `pj_repo_002` | 0 | 0 | `fail` |
| `qwen3-0.6b-base` | `project-judgment` | `pj_repo_003` | 0 | 0 | `fail` |
| `qwen3-0.6b-lora-smoke` | `project-judgment` | `pj_repo_003` | 0 | 0 | `fail` |

## reasoning

| model | suite | task | raw | qwen-final | status |
| --- | --- | --- | ---: | ---: | --- |
| `qwen3-0.6b-base` | `project-judgment` | `pj_eval_001` | 1 | 1 | `pass` |
| `qwen3-0.6b-lora-smoke` | `project-judgment` | `pj_eval_001` | 0 | 0 | `fail` |
| `qwen3-0.6b-base` | `project-judgment` | `pj_eval_002` | 0 | 0 | `fail` |
| `qwen3-0.6b-lora-smoke` | `project-judgment` | `pj_eval_002` | 0 | 0 | `fail` |
| `qwen3-0.6b-base` | `project-judgment` | `pj_eval_003` | 0 | 0 | `fail` |
| `qwen3-0.6b-lora-smoke` | `project-judgment` | `pj_eval_003` | 0 | 0 | `fail` |

## reasoning

| model | suite | task | raw | qwen-final | status |
| --- | --- | --- | ---: | ---: | --- |
| `qwen3-0.6b-base` | `project-judgment` | `pj_exp_001` | 0 | 0 | `fail` |
| `qwen3-0.6b-lora-smoke` | `project-judgment` | `pj_exp_001` | 0 | 0 | `fail` |
| `qwen3-0.6b-base` | `project-judgment` | `pj_exp_002` | 0 | 0 | `fail` |
| `qwen3-0.6b-lora-smoke` | `project-judgment` | `pj_exp_002` | 0 | 0 | `fail` |
| `qwen3-0.6b-base` | `project-judgment` | `pj_exp_003` | 0 | 0 | `fail` |
| `qwen3-0.6b-lora-smoke` | `project-judgment` | `pj_exp_003` | 0 | 0 | `fail` |

## instruction

| model | suite | task | raw | qwen-final | status |
| --- | --- | --- | ---: | ---: | --- |
| `qwen3-0.6b-base` | `leverage-smoke` | `instruction_json` | 0 | 1 | `parsed_only` |
| `qwen3-0.6b-lora-smoke` | `leverage-smoke` | `instruction_json` | 0 | 1 | `parsed_only` |
| `qwen3-0.6b-base` | `leverage-smoke` | `instruction_bullets` | 0 | 0 | `fail` |
| `qwen3-0.6b-lora-smoke` | `leverage-smoke` | `instruction_bullets` | 0 | 0 | `fail` |
| `qwen3-0.6b-base` | `leverage-smoke` | `instruction_lowercase` | 0 | 1 | `parsed_only` |
| `qwen3-0.6b-lora-smoke` | `leverage-smoke` | `instruction_lowercase` | 0 | 1 | `parsed_only` |

## reasoning

| model | suite | task | raw | qwen-final | status |
| --- | --- | --- | ---: | ---: | --- |
| `qwen3-0.6b-base` | `project-judgment` | `pj_loss_001` | 1 | 1 | `pass` |
| `qwen3-0.6b-lora-smoke` | `project-judgment` | `pj_loss_001` | 1 | 1 | `pass` |
| `qwen3-0.6b-base` | `project-judgment` | `pj_loss_002` | 0 | 0 | `fail` |
| `qwen3-0.6b-lora-smoke` | `project-judgment` | `pj_loss_002` | 0 | 0 | `fail` |
| `qwen3-0.6b-base` | `project-judgment` | `pj_loss_003` | 0 | 0 | `fail` |
| `qwen3-0.6b-lora-smoke` | `project-judgment` | `pj_loss_003` | 0 | 0 | `fail` |

## qa

| model | suite | task | raw | qwen-final | status |
| --- | --- | --- | ---: | ---: | --- |
| `qwen3-0.6b-base` | `leverage-smoke` | `qa_capital_france` | 0 | 1 | `parsed_only` |
| `qwen3-0.6b-lora-smoke` | `leverage-smoke` | `qa_capital_france` | 0 | 0 | `fail` |
| `qwen3-0.6b-base` | `leverage-smoke` | `qa_water_freezing` | 0 | 0 | `fail` |
| `qwen3-0.6b-lora-smoke` | `leverage-smoke` | `qa_water_freezing` | 0 | 0 | `fail` |
| `qwen3-0.6b-base` | `leverage-smoke` | `qa_author` | 0 | 0 | `fail` |
| `qwen3-0.6b-lora-smoke` | `leverage-smoke` | `qa_author` | 0 | 1 | `parsed_only` |

## reasoning

| model | suite | task | raw | qwen-final | status |
| --- | --- | --- | ---: | ---: | --- |
| `qwen3-0.6b-base` | `leverage-smoke` | `reasoning_arithmetic` | 0 | 0 | `fail` |
| `qwen3-0.6b-lora-smoke` | `leverage-smoke` | `reasoning_arithmetic` | 0 | 0 | `fail` |
| `qwen3-0.6b-base` | `leverage-smoke` | `reasoning_order` | 0 | 1 | `parsed_only` |
| `qwen3-0.6b-lora-smoke` | `leverage-smoke` | `reasoning_order` | 0 | 1 | `parsed_only` |

## reasoning

| model | suite | task | raw | qwen-final | status |
| --- | --- | --- | ---: | ---: | --- |
| `qwen3-0.6b-base` | `project-judgment` | `pj_cost_001` | 0 | 0 | `fail` |
| `qwen3-0.6b-lora-smoke` | `project-judgment` | `pj_cost_001` | 0 | 0 | `fail` |
| `qwen3-0.6b-base` | `project-judgment` | `pj_cost_002` | 0 | 0 | `fail` |
| `qwen3-0.6b-lora-smoke` | `project-judgment` | `pj_cost_002` | 0 | 0 | `fail` |
| `qwen3-0.6b-base` | `project-judgment` | `pj_cost_003` | 0 | 0 | `fail` |
| `qwen3-0.6b-lora-smoke` | `project-judgment` | `pj_cost_003` | 0 | 0 | `fail` |

## summarization

| model | suite | task | raw | qwen-final | status |
| --- | --- | --- | ---: | ---: | --- |
| `qwen3-0.6b-base` | `leverage-smoke` | `summary_mission` | 1 | 1 | `pass` |
| `qwen3-0.6b-lora-smoke` | `leverage-smoke` | `summary_mission` | 0 | 0 | `fail` |
| `qwen3-0.6b-base` | `leverage-smoke` | `summary_runpod` | 1 | 1 | `pass` |
| `qwen3-0.6b-lora-smoke` | `leverage-smoke` | `summary_runpod` | 1 | 1 | `pass` |

## reasoning

| model | suite | task | raw | qwen-final | status |
| --- | --- | --- | ---: | ---: | --- |
| `qwen3-0.6b-base` | `project-judgment` | `pj_track_001` | 0 | 0 | `fail` |
| `qwen3-0.6b-lora-smoke` | `project-judgment` | `pj_track_001` | 0 | 0 | `fail` |
| `qwen3-0.6b-base` | `project-judgment` | `pj_track_002` | 0 | 0 | `fail` |
| `qwen3-0.6b-lora-smoke` | `project-judgment` | `pj_track_002` | 0 | 0 | `fail` |
| `qwen3-0.6b-base` | `project-judgment` | `pj_track_003` | 0 | 0 | `fail` |
| `qwen3-0.6b-lora-smoke` | `project-judgment` | `pj_track_003` | 0 | 0 | `fail` |
