# gpt-oss-20b Transformers Probe

Date: 2026-05-10

## Goal

Check whether `openai/gpt-oss-20b` can load and generate a tiny sample on
RunPod before adding it to any benchmark or LoRA/SFT workflow.

This was a compatibility probe, not a benchmark.

## Setup

- Model: `openai/gpt-oss-20b`
- Backend: `transformers.pipeline`
- GPU: `NVIDIA GeForce RTX 5090`
- Cloud: RunPod Secure Cloud
- Template id: `runpod-torch-v280`
- Image: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
- CUDA filter: `12.8`, `12.9`, `13.0`
- Cost rate: `$0.99/h`
- Output path: `outputs/leverage-gpt-oss-20b-probe`
- Pod: `llm-leverage-gpt-oss-20b-probe-20260510-173014`
- Pod id: `eiaalid9nlndih`

Cleanup completed:

```text
runpodctl pod delete eiaalid9nlndih
runpodctl pod list -o json
[]
```

## Dependencies

The project setup installed the locked runtime first:

```text
torch=2.8.0+cu128
cuda_available=True
cuda_device=NVIDIA GeForce RTX 5090
```

The probe then installed the model runtime packages:

```bash
uv pip install -U transformers accelerate triton==3.4 kernels
```

The resulting `transformers` package was `5.8.0`.

## Result

The probe passed.

From `outputs/leverage-gpt-oss-20b-probe/probe.json`:

```json
{
  "model": "openai/gpt-oss-20b",
  "backend": "transformers.pipeline",
  "torch": "2.8.0+cu128",
  "cuda_available": true,
  "cuda_device": "NVIDIA GeForce RTX 5090",
  "status": "passed",
  "load_seconds": 33.64,
  "memory_allocated_gb_after_load": 12.83,
  "max_memory_allocated_gb_after_load": 14.988,
  "total_seconds": 50.838,
  "max_memory_allocated_gb": 14.988
}
```

The two tiny generation prompts both answered `42`.

The returned assistant content included Harmony-style channel markers inside
the visible string, for example:

```text
analysis...assistantfinal42
```

This means the model loaded and generated, but the output contract still needs
explicit post-processing or a serving path that exposes reasoning/final fields
cleanly before using the model in quality benchmarks.

## Timing

From `outputs/leverage-gpt-oss-20b-probe/runpod-timings.json`:

- Total RunPod wall time: 180.090 seconds
- SSH info wait: 56.112 seconds
- Setup: 30.560 seconds
- CUDA smoke: 8.693 seconds
- Runtime package install: 3.124 seconds
- Model load and generation probe: 76.064 seconds
- Output sync: 0.416 seconds

Approximate cost: `$0.05`.

## Interpretation

`openai/gpt-oss-20b` is viable enough for the next compatibility step on an RTX
5090 using the official RunPod PyTorch 2.8.0 template. The peak allocated GPU
memory was about 15GB, consistent with the MXFP4 path rather than bf16 loading.

Do not run full benchmarks yet. The next step should decide how to parse or
serve Harmony output:

- use the Transformers chat template and add a project-owned final-answer
  extractor,
- use `transformers serve` or vLLM and inspect structured response behavior,
- then run a tiny GSM8K/IFEval sample only after the visible-answer contract is
  explicit.
