# LM Harness GSM8K Limit 50

Date: 2026-05-10

## Goal

Add a second external benchmark axis for the Qwen3.5-9B LoRA adapter. IFEval
showed instruction-following regression, but the project-owned eval is low
trust, so this run checks whether the adapter also damages basic math
reasoning on GSM8K.

Run both no-thinking and thinking-on modes so later readers do not confuse a
mode mismatch with a model-quality result.

## Setup

- Benchmark: `gsm8k`
- Harness: `lm-evaluation-harness` from GitHub, installed as
  `lm_eval[math] @ git+https://github.com/EleutherAI/lm-evaluation-harness.git`
- Model: `Qwen/Qwen3.5-9B`
- Adapter: `outputs/leverage-sft-qwen35-9b/lora-adapter`
- Config: `tracks/leverage/configs/leverage-sft-qwen35-9b.toml`
- Limit: 50 requests
- Few-shot: 5, harness default for GSM8K
- Batch size: 4
- Backend: `lm-evaluation-harness` `hf`
- Hardware: RunPod Secure Cloud, `NVIDIA GeForce RTX 4090`
- Image: `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`
- CUDA filter: 12.8, 12.9, 13.0

The first dependency attempt used `uv pip install lm-evaluation-harness` and
failed because that package name was not available in the environment. The
successful runs installed the harness from GitHub with the `math` extra.

## Results

| variant | thinking | flexible exact match | strict exact match | wall time | command time | generation time | cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | off | 0.82 | 0.80 | `669.705s` | `377.532s` | `132.171s` | about `$0.13` |
| adapter | off | 0.92 | 0.90 | `557.370s` | `294.709s` | `160.284s` | about `$0.11` |
| base | on | 0.00 | 0.00 | `508.688s` | `296.310s` | `132.217s` | about `$0.10` |
| adapter | on | 0.02 | 0.00 | `521.719s` | `344.180s` | `214.676s` | about `$0.10` |

Cost formula: `wall_seconds / 3600 * $0.69/hr`.

## Interpretation

In no-thinking visible-answer mode, the adapter did not show a broad GSM8K
reasoning collapse on this 50-request sample. It outperformed the base by
`+0.10` exact match on both flexible and strict extraction.

This does not override the full IFEval regression. It narrows the failure
hypothesis: the current 1,216-row adapter appears more likely to have an
instruction-following or response-format regression than a general reasoning
regression visible on this GSM8K sample.

The thinking-on rows are not valid model-quality comparisons. Logged prompts
end with `<|im_start|>assistant\n<think>\n`, so `enable_thinking=True` did
affect the chat template. However, logged completions never reached the
configured `</think>` terminator. Tokenizing the saved completions with the
Qwen tokenizer showed that 88/100 base sample rows and 94/100 adapter sample
rows ended at exactly 256 generated tokens, matching the lm-evaluation-harness
default `max_gen_toks`.

This points to a generation-budget/configuration failure before any LoRA
quality claim: the model began the Qwen thinking block, but the harness cut off
generation before the model closed the block and exposed a final answer. The
current command also used GSM8K's default greedy generation
(`do_sample=False`, `temperature=0.0`), while Qwen's guidance recommends
sampling for thinking mode. A valid thinking-on comparison needs a separate
probe with a larger `max_gen_toks` and Qwen-compatible sampling settings before
recording quality scores.

## Cleanup

All created RunPod pods were deleted after output sync. Final pod list was
empty.
