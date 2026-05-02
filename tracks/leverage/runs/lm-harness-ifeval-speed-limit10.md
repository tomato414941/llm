# LM Harness IFEval Speed Limit-10

## Goal

Measure whether EleutherAI `lm-evaluation-harness` IFEval speed is sensitive to
`--batch-size` on the `Qwen/Qwen3.5-9B` base model.

This is a speed probe, not a benchmark score.

## Setup

- Task: `ifeval`
- Limit: `10`
- Model variant: `Qwen/Qwen3.5-9B` base
- Thinking mode: `--no-enable-thinking`
- Backend: `lm-evaluation-harness` `hf`
- GPU: `NVIDIA A40`
- Cloud: RunPod Secure Cloud
- Image: `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`
- CUDA filter: `12.8`, `12.9`, `13.0`
- Cost rate: `$0.44/hr`

The three batch sizes were run inside one pod, but each condition invoked a
separate `lm_eval` process and therefore reloaded the model.

## Results

| batch size | seconds | requests/sec | relative speed |
| --- | ---: | ---: | ---: |
| `auto` | 287.059 | 0.034836 | 1.00x |
| `2` | 199.272 | 0.050183 | 1.44x |
| `4` | 174.127 | 0.057429 | 1.65x |

`batch_size=auto` detected largest batch size `1`.

The score was identical across all three runs on this 10-request subset:

- prompt strict: `0.9000`
- prompt loose: `0.9000`
- instruction strict: `0.9444`
- instruction loose: `0.9444`

## Timing

From `outputs/leverage-lm-harness-speed-ifeval-limit10/runpod-timings.json`:

- Total RunPod wall time: `758.978s`
- Remote benchmark command: `679.041s`
- SSH readiness: `23.227s`
- Setup: `33.466s`
- CUDA smoke: `16.818s`
- Output sync: `0.566s`

Approximate cost:

```text
758.978s / 3600 * $0.44/hr = about $0.09
```

## Interpretation

Fixed `batch_size=4` is the best current setting for IFEval through the HF
backend on one A40. The default `auto` path is too conservative here because it
settles on batch size `1`.

This improves speed, but not enough to make full IFEval cheap. The full run is
still fundamentally generation-heavy, and each separate model variant run pays
model initialization and loading cost.

## Artifacts

Ignored output files:

- `outputs/leverage-lm-harness-speed-ifeval-limit10/speed-results.json`
- `outputs/leverage-lm-harness-speed-ifeval-limit10/runpod-timings.json`
- `outputs/leverage-lm-harness-speed-ifeval-limit10/batch-auto/base/Qwen__Qwen3.5-9B/results_2026-05-02T14-50-54.985269.json`
- `outputs/leverage-lm-harness-speed-ifeval-limit10/batch-2/base/Qwen__Qwen3.5-9B/results_2026-05-02T14-54-15.628955.json`
- `outputs/leverage-lm-harness-speed-ifeval-limit10/batch-4/base/Qwen__Qwen3.5-9B/results_2026-05-02T14-57-09.484606.json`

## Cleanup

- Pod `dobq6uwib3ymeg` was deleted by the runner.
- Final `runpodctl pod list -o json` returned `[]`.

## Next Decision

Use `--batch-size 4` for future IFEval HF-backend runs unless a larger model
variant fails memory checks.

If benchmark speed remains the blocker after this, the next probe should compare
HF backend against a serving backend such as vLLM on a small fixed limit.
