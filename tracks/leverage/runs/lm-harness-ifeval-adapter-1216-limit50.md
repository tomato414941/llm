# LM Harness IFEval 1216-Row Adapter Limit 50

Date: 2026-05-09

## Goal

Run a small IFEval sample on the LoRA adapter produced by
`qwen35-9b-baseline-1216.md`, then compare it with the previous limit-50 base
and adapter samples.

This is a diagnostic limited run, not a benchmark claim.

## Setup

- Task: `ifeval`
- Limit: 50
- Thinking mode: `--no-enable-thinking`
- Batch size: 4
- Backend: EleutherAI `lm-evaluation-harness` `hf`
- Base model: `Qwen/Qwen3.5-9B`
- Adapter: `outputs/leverage-sft-qwen35-9b/lora-adapter`
- GPU: `NVIDIA GeForce RTX 4090`
- Cloud: RunPod Secure Cloud
- Location: `RO`
- Image: `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`
- CUDA filter: `12.8`, `12.9`, `13.0`
- Cost rate: `$0.69/hr`

## Result

| model | prompt strict | prompt loose | instruction strict | instruction loose |
| --- | ---: | ---: | ---: | ---: |
| previous base sample | 0.9200 | 0.9000 | 0.9342 | 0.9342 |
| previous adapter sample | 0.8400 | 0.8200 | 0.8684 | 0.8684 |
| 1,216-row adapter sample | 0.8800 | 0.8600 | 0.9079 | 0.8947 |

Delta, 1,216-row adapter minus previous adapter:

- prompt strict: +0.0400
- prompt loose: +0.0400
- instruction strict: +0.0395
- instruction loose: +0.0263

Delta, 1,216-row adapter minus previous base:

- prompt strict: -0.0400
- prompt loose: -0.0400
- instruction strict: -0.0263
- instruction loose: -0.0395

## Interpretation

The 1,216-row adapter partially recovers the earlier adapter regression on this
limit-50 IFEval sample, but it still trails the base sample.

This is a useful signal, not a final conclusion. The result is sampled, and the
base numbers are reused from the previous A40 run because the base model,
benchmark task, thinking mode, and batch size are unchanged.

## Timing

From
`outputs/leverage-lm-harness-ifeval-samples-20260509/adapter-limit50/runpod-timings.json`:

- Status: `passed`
- Total RunPod wall time: `1254.422s`
- Pod create: `1.306s`
- SSH info wait: `117.338s`
- SSH ready wait: `1.572s`
- Setup: `41.053s`
- CUDA smoke: `25.776s`
- Package install command: `33.074s`
- Import smoke command: `101.065s`
- Benchmark command step: `914.137s`
- Output sync: `2.523s`
- Approximate cost: `1254.422 / 3600 * $0.69 = about $0.24`

From
`outputs/leverage-lm-harness-ifeval-samples-20260509/adapter-limit50/adapter-limit50-timing.json`:

- Harness elapsed time: `910.965s`
- Generation started after: `244.245s`
- Last generation progress seen after: `904.148s`
- Observed generation interval: `659.903s`

## Artifacts

Ignored output files:

- `outputs/leverage-lm-harness-ifeval-samples-20260509/adapter-limit50/adapter/outputs__leverage-sft-qwen35-9b__lora-adapter/results_2026-05-09T15-44-21.098729.json`
- `outputs/leverage-lm-harness-ifeval-samples-20260509/adapter-limit50/adapter/outputs__leverage-sft-qwen35-9b__lora-adapter/samples_ifeval_2026-05-09T15-44-21.098729.jsonl`
- `outputs/leverage-lm-harness-ifeval-samples-20260509/adapter-limit50/adapter-limit50-timing.json`
- `outputs/leverage-lm-harness-ifeval-samples-20260509/adapter-limit50/runpod-timings.json`

## Cleanup

- First RunPod create attempt returned HTTP 500 and did not create a pod.
- Successful pod id: `254nhse6wg9tjx`
- Successful pod name:
  `llm-leverage-ifeval-adapter-limit50-20260509-20260509-152334`
- The runner deleted the pod.
- Final `runpodctl pod list -o json` returned `[]`.

## Next Step

Do not treat this as proof that the adapter is externally stronger than the
base model. The next useful step is either a fuller IFEval run or a second
external benchmark, depending on whether the immediate goal is regression
diagnosis or broader capability coverage.
