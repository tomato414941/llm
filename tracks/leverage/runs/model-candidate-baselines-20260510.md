# Model Candidate Baselines

Date: 2026-05-10

## Goal

Check whether an alternative no-thinking instruction model looks better than
`Qwen/Qwen3.5-9B` before moving the leverage track away from the current base.

Use only base models. Do not train adapters in this pass.

## Setup

- Benchmarks: IFEval limit 50, GSM8K limit 50
- Harness: `lm-evaluation-harness` from GitHub
- Backend: `hf`
- Chat template: enabled
- Thinking: not enabled
- Batch size: 4
- Hardware: RunPod Secure Cloud, `NVIDIA GeForce RTX 4090`
- Image: `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`
- CUDA filter: 12.8, 12.9, 13.0

## Results

| model | status | IFEval prompt strict | IFEval prompt loose | IFEval instruction strict | IFEval instruction loose | GSM8K flexible | GSM8K strict | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `Qwen/Qwen3.5-9B` | existing baseline | 0.92 | 0.90 | 0.9342 | 0.9342 | 0.82 | 0.80 | IFEval row is the existing limit-50 base sample diagnosis; GSM8K row is the 2026-05-10 no-thinking base run. |
| `Qwen/Qwen2.5-7B-Instruct` | completed | 0.66 | 0.72 | 0.7237 | 0.7632 | 0.68 | 0.24 | Weaker than Qwen3.5 on both axes. |
| `mistralai/Mistral-7B-Instruct-v0.3` | completed | 0.48 | 0.48 | 0.5658 | 0.5789 | 0.44 | 0.44 | Weaker than Qwen3.5 and Qwen2.5 on this sample. |
| `meta-llama/Llama-3.1-8B-Instruct` | blocked | n/a | n/a | n/a | n/a | n/a | n/a | Hugging Face gated repo returned 401 unauthenticated. |

## Timing And Cost

| model | workload | wall time | IFEval command | IFEval generation | GSM8K command | GSM8K generation | cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Qwen/Qwen2.5-7B-Instruct` | IFEval 50 + GSM8K 50 | `543.375s` | `244.447s` | `143.604s` | `145.188s` | `75.766s` | about `$0.10` |
| `mistralai/Mistral-7B-Instruct-v0.3` | IFEval 50 + GSM8K 50 | `1137.572s` | `466.129s` | `238.617s` | `267.132s` | `85.249s` | about `$0.22` |
| `meta-llama/Llama-3.1-8B-Instruct` | access check through IFEval startup | `293.875s` | n/a | n/a | n/a | n/a | about `$0.06` |

Cost formula: `wall_seconds / 3600 * $0.69/hr`.

## Interpretation

Do not move the mainline from `Qwen/Qwen3.5-9B` to either tested open
alternative based on this pass. Qwen2.5-7B-Instruct avoids the Qwen3.5 thinking
confusion, but as a no-thinking base it is materially weaker on the two
external sample checks. Mistral-7B-Instruct-v0.3 is weaker still and slower on
IFEval in this setup.

`meta-llama/Llama-3.1-8B-Instruct` remains an interesting non-Qwen comparison,
but it requires authenticated Hugging Face access before it can be evaluated in
this RunPod workflow.

Current recommendation: keep `Qwen/Qwen3.5-9B` as a no-thinking-only baseline
for now. If adding another candidate, prefer an accessible stronger open model
over Qwen2.5-7B or Mistral-7B v0.3.

## Cleanup

All pods created for this run were deleted. Final RunPod list was empty.

