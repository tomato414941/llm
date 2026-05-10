# gpt-oss-20b Serving Probe

Date: 2026-05-10

## Goal

Check whether serving `openai/gpt-oss-20b` exposes the final answer more
structurally than the raw `transformers.pipeline` path.

This was a serving compatibility probe, not a quality benchmark.

## Setup

- Model: `openai/gpt-oss-20b`
- GPU: `NVIDIA GeForce RTX 5090`
- Cloud: RunPod Secure Cloud
- Template id: `runpod-torch-v280`
- Image: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
- CUDA filter: `12.8`, `12.9`, `13.0`
- Cost rate: `$0.99/h`

Cleanup completed after each attempt:

```text
runpodctl pod list -o json
[]
```

## vLLM Attempt

The OpenAI Cookbook vLLM path was not runnable as-is in this environment.

Install command:

```bash
uv pip install --pre vllm==0.10.1+gptoss \
  --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
  --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
  --index-strategy unsafe-best-match openai
```

Result:

```text
No solution found
vllm==0.10.1+gptoss depends on torch==2.9.0.dev20250804+cu128
```

This is an environment/package resolution issue, not evidence about model
quality or Harmony handling.

## Transformers Serve Attempts

The official-looking minimal command was also not enough in the project venv:

```bash
uv pip install -U transformers accelerate triton==3.4 kernels openai
uv run transformers serve
```

Observed blockers:

- `transformers` CLI required `requests`.
- `transformers serve` required `fastapi` and `uvicorn` for
  `is_serve_available()`.
- `openai==2.36.0` did not match the type imports used by
  `transformers==5.8.0`; pinning `openai<2` avoided that issue.
- `/v1/models` returned 500 when the Hugging Face cache was empty, so it is not
  a reliable readiness probe before first model load.
- `/v1/responses` rejected the documented payload shape in this environment:
  `Unexpected fields in the request: {'messages', 'max_tokens'}`.

The working serving command used:

```bash
uv pip install -U transformers accelerate triton==3.4 kernels \
  requests fastapi uvicorn 'openai<2'
uv run transformers serve
```

The probe then called `/v1/chat/completions` with:

```json
{
  "model": "openai/gpt-oss-20b",
  "messages": [
    {"role": "system", "content": "Reasoning: low"},
    {"role": "user", "content": "What is 17 + 25? Answer with just the number."}
  ],
  "temperature": 0,
  "max_tokens": 128,
  "stream": false
}
```

## Result

The `/v1/chat/completions` serving path returned HTTP 200, but the assistant
content was still raw Harmony-style text:

```text
analysisThe user asks: "What is 17 + 25? Answer with just the number." So we need to compute 17+25 = 42. Just output "42".assistantfinal42
```

From `outputs/gpt-oss-20b-transformers-chat-serving-probe/chat.raw`:

```json
{
  "model": "openai/gpt-oss-20b@main",
  "object": "chat.completion",
  "choices": [
    {
      "finish_reason": "stop",
      "message": {
        "role": "assistant",
        "content": "analysis...assistantfinal42"
      }
    }
  ],
  "usage": {
    "completion_tokens": 48,
    "prompt_tokens": 93,
    "total_tokens": 141
  }
}
```

## Interpretation

Serving through Transformers does not remove the need for a project-owned
Harmony final-answer extractor. For now, treat `gpt-oss-20b` as operationally
usable only when the evaluation path explicitly extracts the final channel from
raw Harmony-style output.

Do not rely on `/v1/responses` with the current dependency set. If we revisit a
structured serving path, test it separately after pinning a known-good
Transformers/OpenAI SDK combination.
