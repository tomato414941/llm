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

Put local text data under `data/raw/`. Data files are intentionally ignored by git.

Train a BPE tokenizer:

```bash
uv run python -m llm.tokenize \
  --input data/raw/tinyshakespeare.txt \
  --output data/processed/bpe_tokenizer.json \
  --vocab-size 500
```

Prepare reusable token tensors:

```bash
uv run python -m llm.prepare_data \
  --input data/raw/tinyshakespeare.txt \
  --tokenizer data/processed/bpe_tokenizer.json \
  --output data/processed/tinyshakespeare_bpe_500.pt
```

Train the current model:

```bash
uv run python -m llm.train \
  --tokens data/processed/tinyshakespeare_bpe_500.pt \
  --checkpoint checkpoints/mini_gpt.pt \
  --metrics-output experiments/runs/tinyshakespeare_bpe_500.csv
```

`--metrics-output` is optional. When set, eval steps are appended as CSV rows with
train/validation loss and perplexity.

You can also run training from a TOML config. CLI arguments override config
values, which is useful for smoke runs:

```bash
uv run python -m llm.train \
  --config configs/tinyshakespeare_bpe_500_small.toml
```

Resume a training run from a checkpoint that includes optimizer and RNG state:

```bash
uv run python -m llm.train \
  --config configs/tinyshakespeare_bpe_500_small.toml \
  --resume checkpoints/tinyshakespeare_bpe_500_small.pt
```

Generate from a checkpoint:

```bash
uv run python -m llm.generate \
  --checkpoint checkpoints/mini_gpt.pt \
  --prompt "KING:" \
  --max-new-tokens 200 \
  --seed 1337 \
  --samples 2
```

Evaluate a checkpoint on the validation split:

```bash
uv run python -m llm.evaluate \
  --checkpoint checkpoints/mini_gpt.pt \
  --tokens data/processed/tinyshakespeare_bpe_500.pt
```

A useful observation loop is to evaluate the checkpoint first, then generate a
few seeded samples with the same checkpoint.

Write a checkpoint observation report:

```bash
uv run python -m llm.observe \
  --checkpoint checkpoints/first_observation.pt \
  --tokens data/processed/tinyshakespeare_bpe_500.pt \
  --prompt "KING:" \
  --output experiments/observations/first_observation.md \
  --summary-output experiments/summaries/observations.csv \
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
  --checkpoint checkpoints/first_observation.pt \
  --tokens data/processed/tinyshakespeare_bpe_500.pt \
  --prompt-file eval_prompts/tinyshakespeare.jsonl \
  --output experiments/observations/first_observation.md \
  --summary-output experiments/summaries/observations.csv
```

For leverage experiments, use the no-dependency JSONL evaluator only after model
outputs already exist. It reads saved local JSONL predictions, applies
deterministic scoring rules, and writes local CSV summaries. The evaluator is an
offline scoring step only: it does not run models, start RunPod, call external
APIs, download weights, or fetch datasets.

The committed eval layers are:

- `evals/leverage_smoke.jsonl`: small smoke checks for evaluator wiring and
  prediction format.
- `evals/project_judgment_v0.jsonl`: project-specific judgment checks for the
  leverage track once saved predictions exist.

Run both layers against saved predictions by passing `--tasks` more than once:

```bash
uv run python -m llm.leverage.evaluate \
  --tasks evals/leverage_smoke.jsonl \
  --tasks evals/project_judgment_v0.jsonl \
  --predictions experiments/leverage/two_layer.example.jsonl \
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
  --tasks evals/leverage_smoke.jsonl \
  --tasks evals/project_judgment_v0.jsonl \
  --model <provider-model-id> \
  --model-label <local_label> \
  --output experiments/leverage/<local_label>.jsonl
```

The collector only calls the configured API endpoint and writes saved
predictions. It does not use RunPod, download weights, or fine-tune models.

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
  --output experiments/leverage/qwen3_5_4b_runpod.jsonl \
  --scores-output experiments/leverage/qwen3_5_4b_scores.csv \
  --summary-output experiments/leverage/qwen3_5_4b_summary.csv
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
  --output experiments/leverage/qwen3_5_4b_runpod.jsonl \
  --scores-output experiments/leverage/qwen3_5_4b_scores.csv \
  --summary-output experiments/leverage/qwen3_5_4b_summary.csv
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
  --config configs/tinyshakespeare_bpe_500_small.toml
```

Append or update a model-size comparison row:

```bash
uv run python -m llm.scaling \
  --checkpoint checkpoints/tinyshakespeare_bpe_500_small.pt \
  --summary experiments/summaries/observations.csv \
  --output experiments/summaries/scaling.csv
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
