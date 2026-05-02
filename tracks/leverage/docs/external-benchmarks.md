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
  --dry-run
```

Run the benchmark by removing `--dry-run`:

```bash
uv run python -m llm.leverage.evaluate_lm_harness \
  --config tracks/leverage/configs/leverage-sft-qwen35-9b.toml \
  --task ifeval \
  --run both \
  --limit 10
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

## Adapter Policy

Keep the LoRA adapter as the source of truth. Use it directly when the
benchmark runner supports PEFT.

Create a merged model only when a benchmark tool cannot load PEFT adapters or
when a serving backend needs a complete model directory. Do not commit merged
models.
