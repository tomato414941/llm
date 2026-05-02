# External Benchmarks

Use external benchmarks as a usability-first supplement to the project-owned
evals. They do not replace `leverage-smoke` or `project-judgment`.

## Adoption Criteria

- One command can evaluate the base model and the LoRA adapter.
- Results are written under ignored `outputs/` paths.
- The benchmark runner owns task formatting and scoring.
- The adapter remains the source artifact; merged models are only temporary
  derived artifacts when another tool cannot load PEFT adapters.

## First Benchmark

Start with EleutherAI `lm-evaluation-harness` and `ifeval`.

Why:

- `ifeval` checks instruction following, which is the closest external signal
  for this SFT/LoRA loop.
- `lm-evaluation-harness` can load PEFT adapters with
  `pretrained=<base>,peft=<adapter>`.
- The same runner can later add `gsm8k`, `hellaswag`, or `arc_easy` without
  new project-owned scoring code.

## Command

Install `lm-evaluation-harness` in the RunPod environment, then dry-run the
project wrapper:

```bash
uv run python -m llm.leverage.evaluate_lm_harness \
  --config tracks/leverage/configs/leverage-sft-qwen35-9b.toml \
  --task ifeval \
  --run both \
  --limit 10 \
  --no-enable-thinking \
  --dry-run
```

Run the benchmark by removing `--dry-run`:

```bash
uv run python -m llm.leverage.evaluate_lm_harness \
  --config tracks/leverage/configs/leverage-sft-qwen35-9b.toml \
  --task ifeval \
  --run both \
  --limit 10 \
  --no-enable-thinking
```

The wrapper runs:

- base: `pretrained=Qwen/Qwen3.5-9B`
- adapter: `pretrained=Qwen/Qwen3.5-9B,peft=outputs/leverage-sft-qwen35-9b/lora-adapter`

Outputs go under:

```text
outputs/leverage-lm-harness/
```

Use a small `--limit` first. Full `ifeval` has 541 generation requests and can
be too slow for a first usability check on a single A40.

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
| `gsm8k` | Undecided | Not yet verified. | Compare `--no-enable-thinking` against `--enable-thinking --think-end-token "</think>"` before choosing a default. |
| `hellaswag` | Undecided | Not yet verified. | Run a small smoke before choosing a default. |
| `arc_easy` | Undecided | Not yet verified. | Run a small smoke before choosing a default. |
| Project-owned smoke evals | `--no-enable-thinking` | Project policy. | Smoke runs should be cheap and verify the visible answer contract. Enable thinking only when the smoke is specifically checking reasoning-mode behavior. |

If thinking mode is enabled, log samples first and verify that the model reaches
the thinking terminator. If it does not reach `</think>` within the generation
limit, the final answer is missing and the benchmark result is not reliable.
