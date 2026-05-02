# LM Harness IFEval Timing Limit-10

## Goal

Verify that `--timing-output` records useful benchmark timing on RunPod before
using it for GPU or backend comparisons.

This is a timing probe, not a benchmark score.

## Setup

- Task: `ifeval`
- Limit: `10`
- Model variant: base
- Base model: `Qwen/Qwen3.5-9B`
- Thinking mode: `--no-enable-thinking`
- Backend: `lm-evaluation-harness` `hf`
- Batch size: `4`
- Device: `cuda:0`
- GPU: `NVIDIA A40`
- Cloud: RunPod Secure Cloud
- Image: `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`
- CUDA filter: `12.8`, `12.9`, `13.0`
- Pod location: `SE`
- Cost rate: `$0.44/hr`

## Command

```bash
uv run python scripts/runpod/run_once.py \
  --name lm-harness-timing-probe \
  --gpu-type "NVIDIA A40" \
  --secure-cloud \
  --max-cost 1.0 \
  --allowed-cuda-version 12.8 \
  --allowed-cuda-version 12.9 \
  --allowed-cuda-version 13.0 \
  --sync tracks/leverage/configs \
  --output outputs/leverage-lm-harness-timing-probe \
  --remote 'mkdir -p outputs/leverage-lm-harness-timing-probe && uv pip install lm-eval peft transformers accelerate datasets sentencepiece protobuf langdetect immutabledict && uv run python -m llm.leverage.evaluate_lm_harness --config tracks/leverage/configs/leverage-sft-qwen35-9b.toml --task ifeval --variant base --limit 10 --batch-size 4 --no-enable-thinking --timing-output outputs/leverage-lm-harness-timing-probe/base-limit10-batch4-timing.json --output-root outputs/leverage-lm-harness-timing-probe' \
  --max-runtime-minutes 30 \
  --allow-existing-pods
```

The first attempt failed because `lm_eval` was not installed in the RunPod
environment. The successful run installed the benchmark dependencies in the
remote command.

## Results

`--timing-output` worked on RunPod.

From
`outputs/leverage-lm-harness-timing-probe/base-limit10-batch4-timing.json`:

- Benchmark command time: `226.772s`
- Generation started after: `181.795s`
- Last generation progress seen after: `222.557s`
- Observed generation time: `40.762s`
- Return code: `0`

Score on the limited 10-request subset:

- prompt strict: `0.9000`
- prompt loose: `0.9000`
- instruction strict: `0.9444`
- instruction loose: `0.9444`

From `outputs/leverage-lm-harness-timing-probe/runpod-timings.json`:

- Total RunPod wall time: `534.698s`
- SSH info wait: `207.775s`
- SSH ready wait: `1.474s`
- Setup: `41.651s`
- CUDA smoke: `18.374s`
- Remote command, including benchmark dependency install and benchmark:
  `254.042s`
- Output sync: `2.222s`

Approximate cost:

```text
534.698s / 3600 * $0.44/hr = about $0.07
```

Benchmark-command-only cost equivalent:

```text
226.772s / 3600 * $0.44/hr = about $0.03
```

Generation-only cost equivalent:

```text
40.762s / 3600 * $0.44/hr = about $0.005
```

## Interpretation

The timing recorder is usable for future speed comparisons. For actual RunPod
cost, use total wall time. For backend, GPU, or batch-size comparisons, compare
benchmark command time and generation time separately.

This run had a large SSH info wait, so wall time is not a clean proxy for model
inference speed. It is still the right number for actual spend.

## Artifacts

Ignored output files:

- `outputs/leverage-lm-harness-timing-probe/base-limit10-batch4-timing.json`
- `outputs/leverage-lm-harness-timing-probe/runpod-timings.json`
- `outputs/leverage-lm-harness-timing-probe/base/Qwen__Qwen3.5-9B/results_2026-05-02T15-51-52.217630.json`

## Cleanup

The RunPod pod was deleted. Final pod list was empty.
