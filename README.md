# llm

Small LLM learning lab.

This project is a from-scratch path toward understanding decoder-only language
models. The current main route is:

1. train a BPE tokenizer
2. prepare token tensors
3. train a small Transformer language model
4. generate text from a checkpoint

The goal is not hyperparameter tuning yet. The goal is to keep the implementation
small enough to read, run, and change while building the core pieces of a GPT-style
model.

See `ROADMAP.md` for the split between the from-scratch learning track and the
open-model leverage track.

## Setup

```bash
uv sync --extra dev
```

## Current Workflow

Put local text data under `tracks/from-scratch/data/raw/`. Data files are intentionally ignored by git.

Train a BPE tokenizer:

```bash
uv run python -m llm.tokenize \
  --input tracks/from-scratch/data/raw/tinyshakespeare.txt \
  --output tracks/from-scratch/data/processed/bpe_tokenizer.json \
  --vocab-size 500
```

Prepare reusable token tensors:

```bash
uv run python -m llm.prepare_data \
  --input tracks/from-scratch/data/raw/tinyshakespeare.txt \
  --tokenizer tracks/from-scratch/data/processed/bpe_tokenizer.json \
  --output tracks/from-scratch/data/processed/tinyshakespeare_bpe_500.pt
```

Train the current model:

```bash
uv run python -m llm.train \
  --tokens tracks/from-scratch/data/processed/tinyshakespeare_bpe_500.pt \
  --checkpoint tracks/from-scratch/checkpoints/mini-gpt.pt \
  --metrics-output tracks/from-scratch/runs/metricstinyshakespeare-bpe-500.csv
```

`--metrics-output` is optional. When set, eval steps are appended as CSV rows with
train/validation loss and perplexity.

You can also run training from a TOML config. CLI arguments override config
values, which is useful for smoke runs:

```bash
uv run python -m llm.train \
  --config tracks/from-scratch/configs/tinyshakespeare-bpe-500-small.toml
```

Resume a training run from a checkpoint that includes optimizer and RNG state:

```bash
uv run python -m llm.train \
  --config tracks/from-scratch/configs/tinyshakespeare-bpe-500-small.toml \
  --resume tracks/from-scratch/checkpoints/tinyshakespeare-bpe-500-small.pt
```

Generate from a checkpoint:

```bash
uv run python -m llm.generate \
  --checkpoint tracks/from-scratch/checkpoints/mini-gpt.pt \
  --prompt "KING:" \
  --max-new-tokens 200 \
  --seed 1337 \
  --samples 2
```

Evaluate a checkpoint on the validation split:

```bash
uv run python -m llm.evaluate \
  --checkpoint tracks/from-scratch/checkpoints/mini-gpt.pt \
  --tokens tracks/from-scratch/data/processed/tinyshakespeare_bpe_500.pt
```

A useful observation loop is to evaluate the checkpoint first, then generate a
few seeded samples with the same checkpoint.

Write a checkpoint observation report:

```bash
uv run python -m llm.observe \
  --checkpoint tracks/from-scratch/checkpoints/first-observation.pt \
  --tokens tracks/from-scratch/data/processed/tinyshakespeare_bpe_500.pt \
  --prompt "KING:" \
  --output tracks/from-scratch/runs/observations/first-observation.md \
  --summary-output tracks/from-scratch/runs/summaries/observations.csv \
  --eval-iters 5 \
  --batch-size 32 \
  --max-new-tokens 200 \
  --temperature 1.0 \
  --top-k 20 \
  --seed 1337 \
  --samples 3
```

Run the same observation flow across a fixed JSONL prompt set:

```bash
uv run python -m llm.observe \
  --checkpoint tracks/from-scratch/checkpoints/first-observation.pt \
  --tokens tracks/from-scratch/data/processed/tinyshakespeare_bpe_500.pt \
  --prompt-file tracks/from-scratch/evals/tinyshakespeare.jsonl \
  --output tracks/from-scratch/runs/observations/first-observation.md \
  --summary-output tracks/from-scratch/runs/summaries/observations.csv
```

For leverage experiments, use the no-dependency JSONL evaluator only after model
outputs already exist. It reads saved local JSONL predictions, applies
deterministic scoring rules, and writes local CSV summaries. The evaluator is an
offline scoring step only: it does not run models, start RunPod, call external
APIs, download weights, or fetch datasets.

The committed eval layers are:

- `tracks/leverage/evals/leverage-smoke.jsonl`: small smoke checks for evaluator wiring and
  prediction format.
- `tracks/leverage/evals/project-judgment-v0.jsonl`: project-specific judgment checks for the
  leverage track once saved predictions exist.
- `tracks/leverage/evals/leverage-model-spec.jsonl`: scenario checks derived from
  `tracks/leverage/model-spec.md`.

Run both layers against saved predictions by passing `--tasks` more than once:

```bash
uv run python -m llm.leverage.evaluate \
  --tasks tracks/leverage/evals/leverage-smoke.jsonl \
  --tasks tracks/leverage/evals/project-judgment-v0.jsonl \
  --predictions tracks/leverage/runs/two-layer.example.jsonl \
  --output /tmp/leverage_scores.csv \
  --summary-output /tmp/leverage_summary.csv
```

Task JSONL records use `id`, `category`, `prompt`, and `scoring`. Prediction
JSONL records use `task_id`, `model`, and `response`. Prediction files must be
created by a separate inference or manual collection step before this command is
run.

Collect predictions through any OpenAI-compatible chat completions API:

```bash
export OPENAI_BASE_URL="https://<provider>/v1"
export OPENAI_API_KEY="..."

uv run python -m llm.leverage.collect_openai \
  --tasks tracks/leverage/evals/leverage-smoke.jsonl \
  --tasks tracks/leverage/evals/project-judgment-v0.jsonl \
  --model <provider-model-id> \
  --model-label <local_label> \
  --output tracks/leverage/runs/<local_label>.jsonl
```

The collector only calls the configured API endpoint and writes saved
predictions. It does not use RunPod, download weights, or fine-tune models.

The current practical path should start with an external OpenAI-compatible
provider instead of a self-hosted RunPod pod. For OpenRouter, store the API key
outside the repository at `~/.secrets/openrouter`, then dry-run the complete
collection and scoring command before making a paid request:

```bash
uv run python scripts/leverage/openai_compatible_eval_once.py --dry-run
```

Run the same command without `--dry-run` only when the selected model and cost
are intentional. The default is `qwen/qwen3.5-flash-02-23` through
`https://openrouter.ai/api/v1`; it writes predictions and evaluator CSVs under
`tracks/leverage/runs/`. Lower-cost or stronger Qwen comparisons can be selected
without changing code:

```bash
uv run python scripts/leverage/openai_compatible_eval_once.py \
  --model qwen/qwen3.5-9b \
  --model-label qwen3-5-9b-openrouter \
  --output tracks/leverage/runs/qwen3-5-9b-openrouter.jsonl \
  --scores-output tracks/leverage/runs/qwen3-5-9b-openrouter-scores.csv \
  --summary-output tracks/leverage/runs/qwen3-5-9b-openrouter-summary.csv
```

This path is inference and evaluation only. It is the right next step for
measuring model behavior cheaply; it is not a fine-tuning or weight-changing
workflow.

For training-data drafting, fix the generation inputs before collecting model
outputs. The committed seed inputs live in
`tracks/leverage/prompts/leverage-training-seed-v0.jsonl`. They are not eval tasks and they are
not training data yet. Treat generated answers as experiment artifacts until
they have been reviewed and promoted deliberately.

The leverage track is organized around a scalable loop:

```text
teacher generation -> structural filter -> model judge -> student training -> held-out eval
```

The target behavior for this loop is defined in
`tracks/leverage/model-spec.md`. Use that spec as the reference for generation
prompts, judge rubrics, reviewed-instruction promotion, and held-out eval
coverage.

Avoid treating small hand-written or manually reviewed data as the source of
capability. The current reviewed instructions are bootstrap material for wiring
the loop and testing SFT export, not a final dataset.

The leverage data lifecycle is:

```text
tracks/leverage/prompts/
  fixed generation inputs for teacher models

tracks/leverage/runs/
  run artifacts for leverage experiments

tracks/leverage/runs/instruction-outputs/
  raw teacher-model answers, filtered candidates, judgments, and summaries;
  local run artifacts, not committed dataset source

tracks/leverage/datasets/reviewed-instructions/
  manually promoted instruction/answer source data after review

tracks/leverage/sft/
  generated training JSONL exports; ignored by git
```

Keep `tracks/leverage/evals/` separate from this flow. Eval files are held-out scoring tasks,
not training-data generation inputs. Do not copy held-out eval prompts into
instruction generation seeds or reviewed instruction rows.

Promotion into `tracks/leverage/datasets/reviewed-instructions/` is manual but should be sparse:
use it for bootstrap examples, spot checks, or deliberately accepted rows. A
reviewer should read the raw answer, verify correctness and usefulness, remove
any private or environment-specific content, preserve provenance with
`source_prompt_id`, and only then add an accepted row to the reviewed dataset.
Exported files under `tracks/leverage/sft/` are derived training inputs; they are not the
source of truth.

Before model judging or promotion, run the structural filter over raw
instruction outputs:

```bash
uv run python -m llm.leverage.filter_instruction_outputs \
  --input tracks/leverage/runs/instruction-outputs/qwen3-5-flash-openrouter.jsonl \
  --output tracks/leverage/runs/instruction-outputs/qwen3-5-flash-openrouter-filter.csv \
  --candidates-output tracks/leverage/runs/instruction-outputs/qwen3-5-flash-openrouter-candidates.jsonl \
  --summary-output tracks/leverage/runs/instruction-outputs/qwen3-5-flash-openrouter-filter-summary.csv
```

This filter performs only structural hygiene checks such as schema, source
prompt, empty output, secret markers, and obvious length issues. It is not a
quality scorer, does not replace a model judge, and does not promote rows into
`tracks/leverage/datasets/`.

Run a model judge over filtered candidates with an explicit small limit first:

```bash
uv run python scripts/leverage/openai_compatible_instruction_judge_once.py \
  --limit 2
```

Judge outputs use separate `generator_model` and `judge_model` fields so the
same schema can later support generator-by-judge evaluation matrices. The judge
step is an automated scorer, not automatic promotion into `tracks/leverage/datasets/`.

The first weight-changing leverage experiment is specified in
`tracks/leverage/configs/leverage-sft-smoke.toml` and `tracks/leverage/docs/sft-smoke.md`. It is a
small LoRA/SFT wiring smoke over the reviewed instruction data, not a quality
claim.

For a RunPod self-hosted spike, start an OpenAI-compatible model server, point
`OPENAI_BASE_URL` at that server, run the same collector command, then score the
saved predictions with `llm.leverage.evaluate`. The first spike should be
inference-only, use the committed eval files, keep the context length modest,
sync back only JSONL/CSV results, and remove the pod as soon as the run finishes
or fails clearly.

The preferred automated RunPod path treats the pod as a temporary
OpenAI-compatible HTTP model server. It starts the server container, runs the
collector and evaluator locally, then removes the pod. It defaults to a small
FP8 Qwen model, but the model, label, image, GPU, and output paths are CLI
options:

```bash
uv run python scripts/runpod/runpod_openai_api_once.py --dry-run
```

Defaults:

- model: `Qwen/Qwen3-14B-FP8` unless overridden with `--model`
- GPU: 1x `NVIDIA GeForce RTX 4090`
- image: `vllm/vllm-openai:latest`
- cost ceiling: `$2.00`
- memory request: `29 GB`
- context: `8192`
- workload: committed leverage eval prompts only
- cleanup: remove the pod after success or failure

For Qwen3.5 and newer model spikes, do not treat `latest` as a stable
compatibility contract. A failed A4000 debug run showed that installing current
vLLM can pull a PyTorch/CUDA stack newer than the host NVIDIA driver supports.
Pin the RunPod image or vLLM/PyTorch/CUDA combination explicitly before
spending on larger runs, and use the script's readiness diagnostics to capture
the last pod `PORTS` state and proxy HTTP status.

The most cost-controlled path is to build and publish a fixed image before
creating another paid pod. See `docker/runpod-vllm-qwen/README.md`. The image
wraps a pinned vLLM base image with a stable entrypoint, a GPU smoke test, and
Qwen3.5-4B defaults so RunPod startup does not spend time installing packages.
Publish it with the `publish-runpod-vllm-qwen` GitHub Actions workflow, then
use `ghcr.io/<owner>/llm-runpod-vllm-qwen:cu129-qwen35` in RunPod.

The next Qwen3.5 compatibility check should prefer a prebuilt vLLM CUDA 12.9
nightly image instead of installing vLLM inside the pod. Keep the first run
short and inference-only; the goal is to prove that `/v1/models` becomes ready
and that the committed eval pipeline can produce JSONL/CSV artifacts:

```bash
uv run python scripts/runpod/runpod_openai_api_once.py \
  --model Qwen/Qwen3.5-4B \
  --served-model-name Qwen/Qwen3.5-4B \
  --model-label qwen3_5_4b \
  --image vllm/vllm-openai:cu129-nightly \
  --gpu-type "NVIDIA L40S" \
  --max-cost 0.8 \
  --max-allowed-cost 0.8 \
  --mem 120 \
  --vcpu 12 \
  --container-disk-size 120 \
  --volume-size 120 \
  --max-model-len 4096 \
  --max-tokens 256 \
  --reasoning-parser qwen3 \
  --wait-seconds 180 \
  --api-wait-seconds 600 \
  --max-runtime-minutes 20 \
  --output tracks/leverage/runs/qwen3-5-4b-runpod.jsonl \
  --scores-output tracks/leverage/runs/qwen3-5-4b-scores.csv \
  --summary-output tracks/leverage/runs/qwen3-5-4b-summary.csv
```

If HTTP ports never appear, stop treating it as a model-quality problem. First
debug RunPod port publication or switch to the SSH-based path. If ports appear
but `/v1/models` stays unavailable, treat it as an image/model/vLLM
compatibility issue and inspect container logs before trying a larger GPU.

If the prebuilt `vllm/vllm-openai` image never exposes ports, use the SSH-based
path on a RunPod PyTorch CUDA 12.8 image so the vLLM log is available. This path
installs the pinned vLLM `0.20.0+cu129` wheel directly, avoiding the PyPI
default CUDA 13 wheel:

```bash
uv run python scripts/runpod/runpod_qwen_eval_once.py \
  --model Qwen/Qwen3.5-4B \
  --model-label qwen3_5_4b \
  --image runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404 \
  --install-vllm \
  --gpu-type "NVIDIA L40S" \
  --max-cost 0.8 \
  --mem 120 \
  --vcpu 12 \
  --container-disk-size 120 \
  --volume-size 120 \
  --max-model-len 4096 \
  --max-tokens 256 \
  --wait-seconds 240 \
  --ssh-wait-seconds 180 \
  --max-runtime-minutes 30 \
  --output tracks/leverage/runs/qwen3-5-4b-runpod.jsonl \
  --scores-output tracks/leverage/runs/qwen3-5-4b-scores.csv \
  --summary-output tracks/leverage/runs/qwen3-5-4b-summary.csv
```

The script is not Qwen-specific. Override `--model`, `--model-label`, `--image`,
and server args for any RunPod image that exposes an OpenAI-compatible API.

Use `--dry-run` first to inspect the exact commands without creating a pod. Do
not start a paid pod unless the objective, cost ceiling, stopping condition, and
cleanup plan are explicit. Generated prediction JSONL and score CSV outputs are
local experiment artifacts and are ignored by git.

Or use the run config for the same observation settings:

```bash
uv run python -m llm.observe \
  --config tracks/from-scratch/configs/tinyshakespeare-bpe-500-small.toml
```

Append or update a model-size comparison row:

```bash
uv run python -m llm.scaling \
  --checkpoint tracks/from-scratch/checkpoints/tinyshakespeare-bpe-500-small.pt \
  --summary tracks/from-scratch/runs/summaries/observations.csv \
  --output tracks/from-scratch/runs/summaries/scaling.csv
```

## Project Shape

- `src/llm/tokenizer.py`: char and BPE tokenizers.
- `src/llm/tokenize.py`: trains and saves a BPE tokenizer.
- `src/llm/prepare_data.py`: encodes text into saved token tensors.
- `src/llm/models/`: Transformer components and language model.
- `src/llm/train.py`: training loop, loss/perplexity reporting, checkpoint saving.
- `src/llm/checkpoint.py`: checkpoint loading and validation.
- `src/llm/config.py`: TOML config loading for reproducible runs.
- `src/llm/generate.py`: checkpoint loading and sampling.
- `src/llm/evaluate.py`: checkpoint evaluation on prepared token data.
- `src/llm/observe.py`: checkpoint evaluation, seeded generation, and report writing.
- `src/llm/scaling.py`: scaling comparison CSV generation.
- `tests/`: focused tests for reusable model and tokenizer behavior.

## Repository Rules

Do not commit datasets, checkpoints, secrets, or generated experiment artifacts.
The intended committed surface is source code, tests, project config, and concise
documentation.

Use `uv` for Python dependency and environment management.

## Verification

```bash
uv run ruff check .
uv run pytest
```

## License

MIT
