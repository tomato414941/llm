# OpenRouter Frontier Eval

Date: 2026-05-02

Goal: compare the Qwen3.5-9B base and LoRA results against current hosted
frontier and strong hosted models on the same 30 held-out eval tasks.

## Models

The OpenRouter model list was checked before running. The evaluated hosted
models were:

- `openai/gpt-5.5`
- `anthropic/claude-opus-4.7`
- `google/gemini-3.1-pro-preview`
- `deepseek/deepseek-v4-pro`
- `qwen/qwen3.6-plus`

## Run

Command shape:

```bash
OPENAI_BASE_URL=https://openrouter.ai/api/v1 \
OPENAI_API_KEY=... \
uv run python -m llm.leverage.evaluate_openrouter \
  --tasks tracks/leverage/evals/leverage-smoke.jsonl \
  --tasks tracks/leverage/evals/project-judgment.jsonl \
  --output-root outputs/leverage-openrouter-eval \
  --max-tokens 512 \
  --overwrite
```

Shared evaluation settings:

- `temperature`: `0.0`
- `max_tokens`: `512`
- `system_prompt`: `Return only the requested answer. Do not include hidden reasoning.`
- `thinking_mode`: `none`
- `thinking_param`: `chat_template_kwargs`
- `reasoning_effort`: `provider_default`
- `exclude_reasoning`: `true`
- `timeout_seconds`: `120.0`

The first run hit an upstream 429 from Alibaba while evaluating
`qwen/qwen3.6-plus`. The CLI was given `--resume`, then only the missing
predictions were regenerated.

Generated outputs:

- `outputs/leverage-openrouter-eval/openrouter-predictions.jsonl`
- `outputs/leverage-openrouter-eval/openrouter-scores.csv`
- `outputs/leverage-openrouter-eval/openrouter-summary.csv`
- `outputs/leverage-openrouter-eval/openrouter-run.json`

These are generated artifacts and are not committed.

## Results

| model | overall | leverage-smoke | project-judgment |
|---|---:|---:|---:|
| `deepseek-v4-pro-openrouter` | 22/30 | 12/12 | 10/18 |
| `gpt-5-5-openrouter` | 20/30 | 11/12 | 9/18 |
| `claude-opus-4-7-openrouter` | 19/30 | 11/12 | 8/18 |
| `qwen3-6-plus-openrouter` | 19/30 | 12/12 | 7/18 |
| `gemini-3-1-pro-preview-openrouter` | 16/30 | 9/12 | 7/18 |
| `qwen3.5-9b-lora-full-batch2` | 17/30 | 9/12 | 8/18 |
| `qwen3.5-9b-base` | 15/30 | 9/12 | 6/18 |

## Cost

OpenRouter reported these costs in the usage metadata:

| model | cost |
|---|---:|
| `claude-opus-4-7-openrouter` | `$0.067245` |
| `deepseek-v4-pro-openrouter` | `$0.021538` |
| `gemini-3-1-pro-preview-openrouter` | `$0.110538` |
| `gpt-5-5-openrouter` | `$0.084190` |
| `qwen3-6-plus-openrouter` | `$0.055968` |
| total | `$0.339478` |

## Interpretation

The Qwen3.5-9B LoRA adapter sits between Gemini 3.1 Pro Preview and the base
Qwen3.5-9B on this strict 30-task eval, but below DeepSeek V4 Pro, GPT-5.5,
Claude Opus 4.7, and Qwen3.6 Plus.

This should not be treated as a broad benchmark. The eval set is small and uses
strict exact, regex, and contains-all scoring. It is still useful as a project
regression harness because every model was scored through the same local
contract.

Gemini 3.1 Pro Preview and Qwen3.6 Plus consumed many reasoning tokens under
OpenRouter despite the short task answers, and Gemini had length-truncated
responses. Future hosted eval runs may need model-specific reasoning controls,
but this run intentionally kept one simple shared OpenRouter path.
