# LM Harness IFEval Full

## Goal

Run full EleutherAI `lm-evaluation-harness` IFEval for both the
`Qwen/Qwen3.5-9B` base model and the trained LoRA adapter.

This is an external benchmark result, not a replacement for project-owned
`leverage-smoke` or `project-judgment` evaluations.

## Setup

- Task: `ifeval`
- Request count: 541
- Thinking mode: `--no-enable-thinking`
- Batch size: `auto`, detected as `1`
- GPU: `NVIDIA A40`
- Cloud: RunPod Secure Cloud
- Image: `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`
- CUDA filter: `12.8`, `12.9`, `13.0`
- Cost rate: `$0.44/hr`

The base and adapter runs were split into two RunPod jobs because a single
base+adapter job would not fit comfortably in the original runtime window.

## Commands

Base:

```bash
uv run python -u -m llm.leverage.evaluate_lm_harness \
  --config tracks/leverage/configs/leverage-sft-qwen35-9b.toml \
  --task ifeval \
  --variant base \
  --no-enable-thinking \
  --output-root outputs/leverage-lm-harness-ifeval-full
```

Adapter:

```bash
uv run python -u -m llm.leverage.evaluate_lm_harness \
  --config tracks/leverage/configs/leverage-sft-qwen35-9b.toml \
  --task ifeval \
  --variant adapter \
  --no-enable-thinking \
  --output-root outputs/leverage-lm-harness-ifeval-full
```

## Results

| model | prompt strict | prompt loose | instruction strict | instruction loose |
| --- | ---: | ---: | ---: | ---: |
| `Qwen/Qwen3.5-9B` | 0.8410 | 0.8817 | 0.8885 | 0.9185 |
| `Qwen/Qwen3.5-9B` + LoRA adapter | 0.7819 | 0.8133 | 0.8501 | 0.8717 |

On full IFEval, the adapter is worse than the base model:

- prompt strict: -0.0591
- prompt loose: -0.0684
- instruction strict: -0.0384
- instruction loose: -0.0468

The same result is recorded as machine-readable one-result-per-line metadata in
`tracks/leverage/runs/benchmark-results.jsonl`.

## Artifacts

Ignored output files:

- `outputs/leverage-lm-harness-ifeval-full/base/Qwen__Qwen3.5-9B/results_2026-05-02T12-22-27.844411.json`
- `outputs/leverage-lm-harness-ifeval-full/adapter/outputs__leverage-sft-qwen35-9b__lora-adapter/results_2026-05-02T14-10-39.330181.json`
- `outputs/leverage-lm-harness-ifeval-full/runpod-timings.json`

## Timing

Base:

- Generation progress completed in `2:18:45`.
- The first RunPod job was manually stopped after the base result was copied
  locally, because the adapter run would not fit safely in the same runtime
  window.

Adapter:

- Total RunPod wall time: `1:45:06`
- Remote benchmark command: `6205.941s`
- SSH readiness: `23.002s`
- Setup: `37.434s`
- CUDA smoke: `14.662s`
- Output sync: `0.566s`

Approximate cost:

```text
base:    about 2.43h * $0.44/hr = about $1.07
adapter: 1.75h * $0.44/hr       = about $0.77
total:                              about $1.84
```

## Interpretation

This result is a negative signal for the current LoRA adapter on general
instruction-following. The adapter may still help the project-owned task
distribution, but it does not improve full IFEval.

The result supports keeping IFEval as an external guardrail: it catches
instruction-following regressions that a small project-owned smoke can miss.

## Cleanup

- Base pod `qt0jfl4zuoy7yk` was deleted after the base output was copied.
- Adapter pod `f2x198jn33c6mb` was deleted by the runner.
- Final `runpodctl pod list -o json` returned `[]`.
