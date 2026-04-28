#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-4B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL_ID}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-1}"

if [[ "${RUNPOD_SMOKE_ONLY:-0}" == "1" ]]; then
  nvidia-smi
  python - <<'PY'
import torch
import vllm

print(f"torch={torch.__version__}")
print(f"vllm={vllm.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_device_count={torch.cuda.device_count()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
PY
  exit 0
fi

if [[ "$#" -gt 0 ]]; then
  exec python -m vllm.entrypoints.openai.api_server "$@"
fi

args=(
  --model "$MODEL_ID"
  --served-model-name "$SERVED_MODEL_NAME"
  --host "$HOST"
  --port "$PORT"
  --max-model-len "$MAX_MODEL_LEN"
)

if [[ "$LANGUAGE_MODEL_ONLY" == "1" ]]; then
  args+=(--language-model-only)
fi

if [[ -n "$REASONING_PARSER" ]]; then
  args+=(--reasoning-parser "$REASONING_PARSER")
fi

if [[ -n "${API_KEY:-}" ]]; then
  args+=(--api-key "$API_KEY")
fi

exec python -m vllm.entrypoints.openai.api_server "${args[@]}"
