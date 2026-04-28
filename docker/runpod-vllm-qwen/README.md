# RunPod vLLM Qwen Image

This image fixes the RunPod runtime surface for Qwen OpenAI-compatible
inference. It avoids installing vLLM inside a paid pod and owns the container
entrypoint so RunPod only needs to start the image and expose HTTP port `8000`.

## Build

```bash
docker build \
  -t llm-runpod-vllm-qwen:cu129-qwen35 \
  docker/runpod-vllm-qwen
```

The default base image is pinned to the current amd64 digest for
`vllm/vllm-openai:cu129-nightly`:

```text
vllm/vllm-openai:cu129-nightly@sha256:96d12f16910e0b1087edad26988e04b2e48a680f0999b952a7ec58bc5d0ac18a
```

If you intentionally want to refresh vLLM, inspect Docker tags and override the
base image explicitly:

```bash
docker build \
  --build-arg VLLM_IMAGE=vllm/vllm-openai:cu129-nightly \
  -t llm-runpod-vllm-qwen:cu129-nightly \
  docker/runpod-vllm-qwen
```

## Smoke Test

Run this only on a GPU host:

```bash
docker run --rm --gpus all \
  -e RUNPOD_SMOKE_ONLY=1 \
  llm-runpod-vllm-qwen:cu129-qwen35
```

The smoke test must print `cuda_available=True`. Do not run a model download
until this passes.

## Runtime Configuration

The entrypoint reads these environment variables:

- `MODEL_ID`: default `Qwen/Qwen3.5-4B`
- `SERVED_MODEL_NAME`: default `MODEL_ID`
- `HOST`: default `0.0.0.0`
- `PORT`: default `8000`
- `MAX_MODEL_LEN`: default `4096`
- `REASONING_PARSER`: default `qwen3`
- `LANGUAGE_MODEL_ONLY`: default `1`
- `API_KEY`: optional OpenAI-compatible API bearer token

Additional container arguments are appended to the vLLM server command.

## RunPod Use

Publish the image to a registry first, then use it with
`scripts/runpod/runpod_openai_api_once.py`:

```bash
uv run python scripts/runpod/runpod_openai_api_once.py \
  --model Qwen/Qwen3.5-4B \
  --served-model-name Qwen/Qwen3.5-4B \
  --model-label qwen3_5_4b \
  --image <registry>/llm-runpod-vllm-qwen:cu129-qwen35 \
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
  --wait-seconds 240 \
  --api-wait-seconds 900 \
  --max-runtime-minutes 25 \
  --output experiments/leverage/qwen3_5_4b_runpod.jsonl \
  --scores-output experiments/leverage/qwen3_5_4b_scores.csv \
  --summary-output experiments/leverage/qwen3_5_4b_summary.csv
```

Stop immediately if port `8000/http` does not appear or `/v1/models` does not
become ready within the configured timeout.
