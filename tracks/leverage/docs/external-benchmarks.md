# External Benchmarks

Use external benchmarks as a usability-first supplement to the project-owned
evals. They do not replace `leverage-smoke` or `project-judgment`.

## Adoption Criteria

- The same wrapper can evaluate each model variant with the same result schema.
- Results are written under ignored `outputs/` paths.
- The benchmark runner owns task formatting and scoring.
- The adapter remains the source artifact; merged models are only temporary
  derived artifacts when another tool cannot load PEFT adapters.

## Current Benchmark Set

Start with EleutherAI `lm-evaluation-harness`.

Current benchmarks:

- `ifeval`: primary instruction-following guardrail for formatting and explicit
  constraint adherence.
- `gsm8k`: second external axis for checking whether adapter changes damage
  basic reasoning beyond IFEval-style surface constraints.

Why this set:

- `lm-evaluation-harness` owns task formatting and scoring.
- The same wrapper can load PEFT adapters with
  `pretrained=<base>,peft=<adapter>`.
- `ifeval` catches instruction-following regressions that project-owned evals
  can miss.
- `gsm8k` gives a simple automatic reasoning signal without adding judge API
  cost.

## Command

Install `lm-evaluation-harness` in the RunPod environment. IFEval also needs
`langdetect` and `immutabledict` in this environment. Then dry-run the project
wrapper:

```bash
uv run python -m llm.leverage.evaluate_lm_harness \
  --config tracks/leverage/configs/leverage-sft-qwen35-9b.toml \
  --task ifeval \
  --variant base \
  --limit 10 \
  --no-enable-thinking \
  --dry-run
```

Run the benchmark by removing `--dry-run`:

```bash
uv run python -m llm.leverage.evaluate_lm_harness \
  --config tracks/leverage/configs/leverage-sft-qwen35-9b.toml \
  --task ifeval \
  --variant base \
  --limit 10 \
  --no-enable-thinking \
  --timing-output outputs/leverage-lm-harness/base-timing.json
```

The wrapper builds model arguments from the selected config:

- base: `pretrained=<base_model>`
- adapter: `pretrained=<base_model>,peft=<evaluation.adapter_dir>`

Run each model variant as a separate command. Use `--variant base` for the base
model and `--variant adapter` for the LoRA adapter. Do not combine variants in
one long benchmark job.

Outputs go under:

```text
outputs/leverage-lm-harness/
```

## Artifact Policy

Keep raw benchmark outputs under ignored `outputs/` paths. Commit concise run
notes and machine-readable summaries only when the result affects future
decisions.

Treat full benchmark runs as decision records. Treat partial runs as smoke,
timing, or sample-level diagnosis; do not use partial-run scores as benchmark
claims.

For stable base models, keep one canonical baseline per benchmark
configuration. Re-run the base only when the benchmark, model revision,
tokenizer, chat template, thinking mode, or decoding settings change.

For adapters, record benchmark results per training run because each adapter is
a new evaluated model variant.

The selected config is the source of truth for the adapter under evaluation.
Record the config path and the training run note with adapter benchmark results
so the result can be traced back to the exact training run.

If raw artifacts become obsolete, they may be deleted after the run note records
the important result, configuration, and interpretation. Mark superseded run
notes explicitly instead of relying on old raw outputs.

Use `--timing-output` when comparing GPUs or backend speed. It records total
command time and, when the lm-evaluation-harness progress output exposes it,
the observed `generate_until` interval. Treat generation timing as unavailable
when the field is `null`; do not infer it from total command time.

For generative benchmarks, if latency looks abnormal, inspect generated sample
length, stop behavior, and thinking traces before assuming CPU fallback.

Record speed and cost according to `tracks/leverage/docs/execution-costs.md`.

Use a small `--limit` first. Full `ifeval` has 541 generation requests and can
be too slow for a first usability check on a single A40. For `gsm8k`, start
with `--limit 50` for both base and adapter before considering a larger run.

For Qwen, use `--no-enable-thinking` for IFEval by default. IFEval scores the
visible response against formatting constraints, and thinking traces can distort
those metrics. `--enable-thinking --think-end-token "</think>"` is available as
a diagnostic mode, but it is not the default.

## Adapter Policy

Keep the LoRA adapter as the source of truth. Use it directly when the
benchmark runner supports PEFT.

Create a merged model only when a benchmark tool cannot load PEFT adapters or
when a serving backend needs a complete model directory. Do not commit merged
models.

## Thinking Mode Policy

Choose thinking mode per benchmark. Do not use one global setting.

| benchmark | default mode | status | note |
| --- | --- | --- | --- |
| `ifeval` | `--no-enable-thinking` | Verified for Qwen. | IFEval scores the visible response against explicit formatting and instruction constraints. Thinking traces distort the metric. |
| `gsm8k` | `--no-enable-thinking` for the first base-vs-adapter comparison. | Planned. | Use the same visible-answer mode as IFEval first; compare thinking mode only if the no-thinking result is ambiguous or unexpectedly poor. |
| `hellaswag` | Undecided | Not yet verified. | Run a small smoke before choosing a default. |
| `arc_easy` | Undecided | Not yet verified. | Run a small smoke before choosing a default. |
| Project-owned smoke evals | `--no-enable-thinking` | Project policy. | Smoke runs should be cheap and verify the visible answer contract. Enable thinking only when the smoke is specifically checking reasoning-mode behavior. |

If thinking mode is enabled, log samples first and verify that the model reaches
the thinking terminator. If it does not reach `</think>` within the generation
limit, the final answer is missing and the benchmark result is not reliable.
