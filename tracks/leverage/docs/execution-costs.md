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

## Records

| date | kind | workload | model | variant | provider / hardware | units | wall time | command time | generation time | cost | note |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-05-02 | benchmark timing probe | IFEval limit 10 | `Qwen/Qwen3.5-9B` | base | RunPod Secure Cloud / `NVIDIA A40` | 10 | `534.698s` | `226.772s` | `40.762s` | about `$0.07` | `tracks/leverage/runs/lm-harness-ifeval-timing-limit10.md` |
