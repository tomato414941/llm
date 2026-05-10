# Execution Costs

Use this document as the source of truth for recording speed and cost
measurements in the leverage track.

The goal is not accounting precision. The goal is to make the next execution
decision better: run or skip, small or full, which GPU, which backend, which
model, which batch size, and whether caching or setup work is worth it.

## Scope

Record time and cost when the result can change a future decision:

- benchmark and evaluation runs
- local or RunPod inference
- generated-data API calls
- judge API calls
- LoRA or SFT training
- RunPod pod startup, setup, model download, and model load

Do not record timing just because it is available. If it will not affect a
future decision, leave it in the raw ignored output.

## Where To Record

Keep policy here:

```text
tracks/leverage/docs/execution-costs.md
```

Keep individual observations in run notes:

```text
tracks/leverage/runs/*.md
```

Keep raw artifacts under ignored output paths:

```text
outputs/
```

Keep a compact record table in this document. Keep details, caveats, and
interpretation in the linked run note.

## What To Record

For any measured run, record the smallest useful set:

- purpose of the run
- workload: task, data count, limit or full-run status
- model target: base model, adapter path when used, model variant, thinking or
  reasoning mode
- execution path: backend, batch size, device, GPU type, cloud, image, CUDA
  filter, and pod location when relevant
- score or quality result, if the run produced one
- total wall time for actual spend
- command time for comparing execution paths
- generation or training-loop time when available
- setup, SSH readiness, dependency install, model download, model load, and
  output sync time only when they materially affect the decision
- cost rate and approximate cost formula

Use `scale` for the compact execution size. For training runs, include enough
training scale to compare unlike runs, such as rows, epochs, optimizer steps,
batch size, and gradient accumulation. Keep detailed loss curves, LoRA config,
and VRAM notes in the linked run note.

## Which Time To Use

Use total wall time for actual spend.

Use command time when comparing backends, batch sizes, model variants, or GPUs.

Use generation or training-loop time only when it is directly measured. If it is
missing, say that it is unavailable. Do not infer it from total time.

For RunPod runs, keep startup and setup separate from benchmark or training
time. Startup and setup affect cost, but they are not model speed.

## Estimates

Write estimates as formulas with assumptions.

Example:

```text
534.698s / 3600 * $0.44/hr = about $0.07
```

When projecting from a limited run, label it as an estimate and state what is
being scaled. Do not present limited-run quality scores as real benchmark
results.

## Current Tools

For `lm-evaluation-harness`, use `--timing-output` from
`llm.leverage.evaluate_lm_harness`. It records command time and, when visible
in harness output, the observed `generate_until` interval.

For RunPod one-off jobs, use `runpod-timings.json` from
`scripts/runpod/run_once.py`. It records pod creation, SSH readiness, setup,
remote command, output sync, and total wall time.

For benchmark and evaluation records, use one row per model variant. If a run
is part of a base-vs-adapter comparison, record the base and adapter as
separate rows and link them through the same run note or purpose.

## Planned Records

| planned date | kind | workload | model | variant | provider / hardware | scale | estimated wall time | estimated command time | estimated generation time | estimated cost | purpose | note |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 2026-05-02 | sample diagnosis | IFEval limit 50 with `--log-samples` | `Qwen/Qwen3.5-9B` | base | RunPod Secure Cloud / `NVIDIA A40` | 50 | about `15-25m` | unknown | unknown | about `$0.11-$0.18` | Fixed baseline sample set for adapter regression diagnosis. | Estimate based on the limit-10 timing probe; setup and model load do not scale linearly. |
| 2026-05-02 | sample diagnosis | IFEval limit 50 with `--log-samples` | `Qwen/Qwen3.5-9B` + LoRA adapter | adapter | RunPod Secure Cloud / `NVIDIA A40` | 50 | about `15-25m` | unknown | unknown | about `$0.11-$0.18` | Diagnose adapter-only IFEval regressions. | Compare against the planned base sample set. |
| 2026-05-04 | LoRA training smoke | reviewed-data short run after long-form constraint promotion | `Qwen/Qwen3.5-9B` | adapter | RunPod Secure Cloud / `NVIDIA RTX 4090` | 1,098 rows, 1ep, 549 steps, ~138 opt steps, bs2, acc4 | about `30-60m` | about `20-45m` | n/a | about `$0.25-$0.50` | Check whether the 1,098-row dataset with new long-form constraint rows trains without instability before scaling data further. | `tracks/leverage/runs/qwen35-9b-lora-long-form-constraint-smoke.md`; config `tracks/leverage/configs/leverage-sft-qwen35-9b-long-form-constraint.toml` |
| 2026-05-07 | LoRA training baseline | full reviewed data after surface-constraint batch 001 | `Qwen/Qwen3.5-9B` | adapter | RunPod Secure Cloud / `NVIDIA RTX 4090` | 1,216 rows, 1ep, 608 steps, ~152 opt steps, bs2, acc4, ~190k-196k tokens estimated | about `35-50m` | about `30-40m` | n/a | about `$0.40-$0.58` | Train the current reviewed-data Qwen3.5-9B baseline after confirming the dataset now includes a stronger long-form constraint signal. | Estimate scales from `tracks/leverage/runs/qwen35-9b-lora-long-form-constraint-smoke.md`; prior 1,098-row run processed 109,860 tokens at 116.303 tokens/sec and cost about `$0.21`; new rows are much longer, so token estimate uses exported text size rather than row count. |

## Benchmark / Evaluation Records

| date | kind | workload | model | variant | provider / hardware | scale | wall time | command time | generation time | cost | note |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-05-02 | benchmark timing probe | IFEval limit 10 | `Qwen/Qwen3.5-9B` | base | RunPod Secure Cloud / `NVIDIA A40` | 10 | `534.698s` | `226.772s` | `40.762s` | about `$0.07` | Historical timing probe; raw run note removed. |
| 2026-05-02 | sample diagnosis | IFEval limit 50 with `--log-samples` | `Qwen/Qwen3.5-9B` | base | RunPod Secure Cloud / `NVIDIA A40` | 50 | `648.600s` | `542.230s` | `373.265s` | about `$0.08` | `tracks/leverage/runs/lm-harness-ifeval-sample-diagnosis.md` |
| 2026-05-02 | sample diagnosis | IFEval limit 50 with `--log-samples` | `Qwen/Qwen3.5-9B` + LoRA adapter | adapter | RunPod Secure Cloud / `NVIDIA A40` | 50 | `774.844s` | `629.557s` | `369.489s` | about `$0.09` | `tracks/leverage/runs/lm-harness-ifeval-sample-diagnosis.md` |
| 2026-05-09 | held-out eval | configured leverage eval tasks after 1,216-row LoRA training | `Qwen/Qwen3.5-9B` and LoRA adapter | base + adapter | RunPod Secure Cloud / `NVIDIA RTX 4090` | 30 tasks, base + adapter | `640.257s` | `438.917s` | unavailable | about `$0.12` | `tracks/leverage/runs/qwen35-9b-baseline-1216-eval.md` |
| 2026-05-09 | sample diagnosis | IFEval limit 50 with `--log-samples` after 1,216-row LoRA training | `Qwen/Qwen3.5-9B` + LoRA adapter | adapter | RunPod Secure Cloud / `NVIDIA RTX 4090` | 50 | `1254.422s` | `910.965s` | `659.903s` | about `$0.24` | `tracks/leverage/runs/lm-harness-ifeval-adapter-1216-limit50.md` |
| 2026-05-09 | external benchmark | full IFEval after 1,216-row LoRA training | `Qwen/Qwen3.5-9B` + LoRA adapter | adapter | RunPod Secure Cloud / `NVIDIA A40` | 541 | `5897.653s` | `5672.374s` | `5465.026s` | about `$0.72` | `tracks/leverage/runs/lm-harness-ifeval-adapter-1216-full.md` |
| 2026-05-10 | external benchmark | GSM8K limit 50, no-thinking | `Qwen/Qwen3.5-9B` | base | RunPod Secure Cloud / `NVIDIA RTX 4090` | 50 | `669.705s` | `377.532s` | `132.171s` | about `$0.13` | `tracks/leverage/runs/lm-harness-gsm8k-limit50.md` |
| 2026-05-10 | external benchmark | GSM8K limit 50, no-thinking | `Qwen/Qwen3.5-9B` + LoRA adapter | adapter | RunPod Secure Cloud / `NVIDIA RTX 4090` | 50 | `557.370s` | `294.709s` | `160.284s` | about `$0.11` | `tracks/leverage/runs/lm-harness-gsm8k-limit50.md` |
| 2026-05-10 | diagnostic benchmark | GSM8K limit 50, thinking-on | `Qwen/Qwen3.5-9B` | base | RunPod Secure Cloud / `NVIDIA RTX 4090` | 50 | `508.688s` | `296.310s` | `132.217s` | about `$0.10` | `tracks/leverage/runs/lm-harness-gsm8k-limit50.md` |
| 2026-05-10 | diagnostic benchmark | GSM8K limit 50, thinking-on | `Qwen/Qwen3.5-9B` + LoRA adapter | adapter | RunPod Secure Cloud / `NVIDIA RTX 4090` | 50 | `521.719s` | `344.180s` | `214.676s` | about `$0.10` | `tracks/leverage/runs/lm-harness-gsm8k-limit50.md` |
| 2026-05-10 | diagnostic benchmark | GSM8K limit 5, thinking-on, `max_gen_toks=16384` | `Qwen/Qwen3.5-9B` | base | RunPod Secure Cloud / `NVIDIA RTX 4090` | 5 | `981.984s` | `827.198s` | `711.089s` | about `$0.19` | `tracks/leverage/runs/lm-harness-gsm8k-limit50.md` |

## Data Generation / Judge Records

| date | kind | workload | model | variant | provider / hardware | scale | wall time | command time | generation time | cost | note |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-05-03 | data generation | long-form constraint seeds `lt_seed_1471`-`lt_seed_1490` | default random OpenRouter generator pool | raw candidates | OpenRouter | 20 | about `4.5m` | not measured exactly | not measured separately | not recorded | `tracks/leverage/runs/long-form-constraint-batch-001.md` |
| 2026-05-03 | model judge | long-form constraint candidates | default random OpenRouter judge pool | non-self judge | OpenRouter | 18 | about `1m` | not measured exactly | n/a | not recorded | `tracks/leverage/runs/long-form-constraint-batch-001.md` |

## LoRA / SFT Training Records

| date | kind | workload | model | variant | provider / hardware | scale | wall time | command time | training time | cost | note |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-05-04 | LoRA training smoke | reviewed-data short run after long-form constraint promotion | `Qwen/Qwen3.5-9B` | adapter | RunPod Secure Cloud / `NVIDIA RTX 4090` | 1,098 rows, 1ep, 549 steps, 138 opt steps, bs2, acc4 | `1100.261s` | `995.435s` | `944.599s` | about `$0.21` | `tracks/leverage/runs/qwen35-9b-lora-long-form-constraint-smoke.md` |
| 2026-05-09 | LoRA training baseline | full reviewed data after surface-constraint batch 001 | `Qwen/Qwen3.5-9B` | adapter | RunPod Secure Cloud / `NVIDIA RTX 4090` | 1,216 rows, 1ep, 608 steps, 152 opt steps, bs2, acc4 | `749.380s` | `566.667s` | `492.657s` | about `$0.14` | `tracks/leverage/runs/qwen35-9b-baseline-1216.md` |
