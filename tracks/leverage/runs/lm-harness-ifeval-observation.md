# LM Harness IFEval Observation

## Goal

Measure why limited `ifeval` was slow before changing benchmark strategy.

This run observes:

- actual generated token count
- GPU utilization and VRAM
- whether CUDA was used
- fixed `batch_size=1` behavior

## Command

RunPod:

- GPU: `NVIDIA A40`
- Cloud: Secure
- Image: `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`
- CUDA filter: `12.8`
- Cost rate: `$0.44/hr`

Remote command:

```bash
mkdir -p outputs/leverage-lm-harness-observe/logs &&
uv pip install lm-eval peft transformers accelerate datasets sentencepiece protobuf langdetect immutabledict &&
(
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 1 \
    > outputs/leverage-lm-harness-observe/logs/gpu-samples.csv &
  monitor=$!
  trap "kill $monitor 2>/dev/null || true" EXIT
  uv run python -m llm.leverage.evaluate_lm_harness \
    --config tracks/leverage/configs/leverage-sft-qwen35-9b.toml \
    --task ifeval \
    --variant base \
    --batch-size 1 \
    --limit 2 \
    --log-samples \
    --output-root outputs/leverage-lm-harness-observe
)
```

## Result

The run completed and wrote samples.

`lm_eval` log:

```text
Using device 'cuda:0'
cuda_available=True
cuda_device=NVIDIA A40
batch_size: 1
Running generate_until requests: 2/2 [01:54, 57.36s/it]
```

The run was not CPU inference.

## Generated Length

Both examples hit the `max_gen_toks=1280` cap.

| sample | prompt tokens | response tokens | response words | prompt strict | instruction strict |
| --- | ---: | ---: | ---: | --- | --- |
| 1 | 87 | 1280 | 765 | false | `[false, true, true]` |
| 2 | 48 | 1280 | 806 | false | `[false]` |

The logged prompts ended with:

```text
<|im_start|>assistant
<think>
```

The generated responses began with `Thinking Process:`. This means the run was
not merely slow because `ifeval` allows long outputs; the model actually filled
the generation budget with thinking-style text.

## GPU Observation

From `outputs/leverage-lm-harness-observe/logs/gpu-samples.csv`:

- Samples: 358
- All-sample GPU util: avg 26.83%, max 92%
- Active samples with memory used over 1GB: 133
- Active GPU util: avg 72.23%, max 92%
- Active VRAM: avg 17.52GB, max 17.54GB
- Active power: avg 180.59W, max 211.46W

Interpretation:

- CUDA was used.
- VRAM was healthy on A40.
- During active generation, GPU utilization was not zero.
- End-to-end utilization looks low because setup, model download, task setup,
  and idle periods are included.

## Timing

From `outputs/leverage-lm-harness-observe/runpod-timings.json`:

- Total wall time: 491.093s
- Remote command: 391.875s
- SSH readiness: 23.214s
- Setup: 41.837s
- CUDA smoke: 25.495s

Approximate RunPod cost:

```text
491.093s / 3600 * $0.44/hr = $0.06
```

## Finding

The main issue is not CPU fallback. The stronger explanation is:

```text
lm-eval + Qwen chat template is causing thinking-style generation,
and the responses hit the 1280-token cap.
```

This also explains why `ifeval` looked much slower than training. The
evaluation generated 1280 autoregressive tokens per example at batch size 1.

## Cleanup

- `runpodctl pod delete qi1p2ac8gpcic4` returned deleted.
- `runpodctl pod list -o json` returned `[]`.

## Next Decision

Before judging `ifeval` as unusable, test whether the Qwen thinking mode can be
disabled through the lm-evaluation-harness path. If it cannot be disabled
cleanly, prefer either:

- a non-thinking rendered prompt path for Qwen, or
- a lighter non-generative benchmark for the first external benchmark.
