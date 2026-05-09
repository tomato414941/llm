# LM Harness IFEval 1216-Row Adapter Full

Date: 2026-05-09

## Goal

Run full EleutherAI `lm-evaluation-harness` IFEval for the LoRA adapter produced
by `qwen35-9b-baseline-1216.md`, then compare it with the existing full base
and old adapter results.

This is an external benchmark result. It is not a replacement for project-owned
`leverage-smoke` or `project-judgment` evaluations.

## Setup

- Task: `ifeval`
- Request count: 541
- Thinking mode: `--no-enable-thinking`
- Batch size: 4
- Backend: EleutherAI `lm-evaluation-harness` `hf`
- Base model: `Qwen/Qwen3.5-9B`
- Adapter: `outputs/leverage-sft-qwen35-9b/lora-adapter`
- GPU: `NVIDIA A40`
- Cloud: RunPod Secure Cloud
- Location: `CA`
- Image: `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`
- CUDA filter: `12.8`, `12.9`, `13.0`
- Cost rate: `$0.44/hr`

## Command

```bash
uv run python -u -m llm.leverage.evaluate_lm_harness \
  --config tracks/leverage/configs/leverage-sft-qwen35-9b.toml \
  --task ifeval \
  --variant adapter \
  --no-enable-thinking \
  --batch-size 4 \
  --output-root outputs/leverage-lm-harness-ifeval-full-20260509-batch4 \
  --timing-output outputs/leverage-lm-harness-ifeval-full-20260509-batch4/adapter-full-timing.json
```

## Results

| model | prompt strict | prompt loose | instruction strict | instruction loose |
| --- | ---: | ---: | ---: | ---: |
| existing base full | 0.8410 | 0.8817 | 0.8885 | 0.9185 |
| old adapter full | 0.7819 | 0.8133 | 0.8501 | 0.8717 |
| 1,216-row adapter full | 0.7800 | 0.8041 | 0.8429 | 0.8609 |

Delta, 1,216-row adapter minus existing base:

- prompt strict: -0.0610
- prompt loose: -0.0776
- instruction strict: -0.0456
- instruction loose: -0.0576

Delta, 1,216-row adapter minus old adapter:

- prompt strict: -0.0019
- prompt loose: -0.0092
- instruction strict: -0.0072
- instruction loose: -0.0108

The same result is recorded as machine-readable one-result-per-line metadata in
`tracks/leverage/runs/benchmark-results.jsonl`.

## Timing

From
`outputs/leverage-lm-harness-ifeval-full-20260509-batch4/runpod-timings.json`:

- Status: `passed`
- Pod id: `zl46g6tnw7bxgm`
- Pod name: `llm-leverage-ifeval-adapter-full-batch4-20260509-20260509-175743`
- Total RunPod wall time: `5897.653s`
- Pod create: `1.202s`
- SSH info wait: `23.004s`
- SSH ready wait: `0.413s`
- Setup: `34.775s`
- CUDA smoke: `24.792s`
- Package install command: `35.838s`
- Import smoke command: `96.848s`
- Benchmark command step: `5674.234s`
- Output sync: `0.566s`
- Approximate cost: `5897.653 / 3600 * $0.44 = about $0.72`

From
`outputs/leverage-lm-harness-ifeval-full-20260509-batch4/adapter-full-timing.json`:

- Harness elapsed time: `5672.374s`
- Generation started after: `199.284s`
- Last generation progress seen after: `5664.310s`
- Observed generation interval: `5465.026s`

## Aborted Auto-Batch Attempt

Before the successful batch-4 run, an A40 run using `--batch-size auto` was
started and stopped manually:

- Pod id: `tm09ozhz5vgeas`
- Batch size: `auto`, detected effectively as slow single-request progress
- Stopped at 97/541 requests after about 31 minutes of generation
- Reason: projected runtime was close to or beyond the 180-minute run window
- Cleanup: pod was deleted

The operational conclusion is simple: use explicit `--batch-size 4` for this
full IFEval adapter path unless a future run proves a better value.

## Artifacts

Ignored output files:

- `outputs/leverage-lm-harness-ifeval-full-20260509-batch4/adapter/outputs__leverage-sft-qwen35-9b__lora-adapter/results_2026-05-09T19-35-54.542104.json`
- `outputs/leverage-lm-harness-ifeval-full-20260509-batch4/adapter-full-timing.json`
- `outputs/leverage-lm-harness-ifeval-full-20260509-batch4/runpod-timings.json`

## Interpretation

The latest 1,216-row adapter did not recover the full IFEval regression. It is
slightly worse than the old adapter on all four full IFEval metrics and remains
meaningfully below the base model.

This conflicts with the limit-50 sample, where the latest adapter looked better
than the old adapter. The practical reading is that the limit-50 sample was not
representative enough to claim recovery.

The held-out project eval still improved for the adapter, so the model may be
learning project-local behavior while preserving or worsening general
instruction-following. For the next data iteration, IFEval should stay as an
external guardrail rather than a training target.

## Cleanup

- Successful pod `zl46g6tnw7bxgm` was deleted by the runner.
- Aborted pod `tm09ozhz5vgeas` was deleted after interruption.
- Final `runpodctl pod list -o json` returned `[]`.
