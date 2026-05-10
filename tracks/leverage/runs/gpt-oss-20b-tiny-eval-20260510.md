# gpt-oss-20b Tiny Eval Probe

Date: 2026-05-10

## Goal

Verify that `openai/gpt-oss-20b` raw Harmony generations can be converted into
visible final answers and scored by the project-owned eval path before using
the model in larger benchmarks.

This was an extraction and scoring probe, not a capability benchmark.

## Setup

- Model: `openai/gpt-oss-20b`
- Backend: `transformers.pipeline`
- System prompt: `Reasoning: low`
- Task suite: first 4 tasks from `tracks/leverage/evals/leverage-smoke.jsonl`
- GPU: `NVIDIA GeForce RTX 5090`
- Cloud: RunPod Secure Cloud
- Template id: `runpod-torch-v280`
- Image: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
- CUDA filter: `12.8`, `12.9`, `13.0`
- Cost rate: `$0.99/h`
- Output path: `outputs/leverage-gpt-oss-20b-tiny-eval`
- Pod: `llm-leverage-gpt-oss-20b-tiny-eval-20260510-175042`
- Pod id: `n4rkypu4dgpg6p`

Cleanup completed:

```text
runpodctl pod delete n4rkypu4dgpg6p
runpodctl pod list -o json
[]
```

## Command

```bash
uv run python scripts/runpod/run_once.py \
  --name llm-leverage-gpt-oss-20b-tiny-eval \
  --secure-cloud \
  --gpu-type 'NVIDIA GeForce RTX 5090' \
  --allowed-cuda-version 12.8 \
  --allowed-cuda-version 12.9 \
  --allowed-cuda-version 13.0 \
  --container-disk-size 120 \
  --volume-size 120 \
  --mem 32 \
  --max-runtime-minutes 45 \
  --sync tracks/leverage/evals \
  --output outputs/leverage-gpt-oss-20b-tiny-eval \
  --remote 'uv pip install -U transformers accelerate triton==3.4 kernels' \
  --remote 'uv run python -u -m llm.leverage.gpt_oss_tiny_eval --task tracks/leverage/evals/leverage-smoke.jsonl --limit 4 --output-root outputs/leverage-gpt-oss-20b-tiny-eval'
```

## Result

The probe passed operationally. Harmony final-answer extraction worked on all
four tasks.

From `outputs/leverage-gpt-oss-20b-tiny-eval/summary.harmony-final.csv`:

| suite | capability | task count | passed | pass rate |
| --- | --- | ---: | ---: | ---: |
| overall | overall | 4 | 3 | 0.750 |
| leverage-smoke | knowledge_qa | 2 | 2 | 1.000 |
| leverage-smoke | summarization_transformation | 2 | 1 | 0.500 |

Final extracted predictions:

```jsonl
{"task_id":"qa_capital_france","response":"Paris"}
{"task_id":"qa_water_freezing","response":"0"}
{"task_id":"summary_mission","response":"The lab pursues two parallel tracks: one that builds small language models from scratch to dissect their inner workings, and another that evaluates and refines existing open models to better emulate practical, general-purpose LLM behavior."}
{"task_id":"summary_runpod","response":"RunPod, a paid external compute resource, requires you to document the objective, expected runtime, cost ceiling, uploaded files, stopping condition, and expected outputs before you use it."}
```

`summary_mission` failed the current `contains_all` scoring because the expected
literal phrase is `two tracks`, while the model wrote `two parallel tracks`.
This is useful evidence that the extraction path works and that the tiny smoke
scoring can be phrase-brittle. Do not treat the 3/4 as a stable quality
benchmark.

## Timing

From `outputs/leverage-gpt-oss-20b-tiny-eval/runpod-timings.json`:

- Total RunPod wall time: 176.318 seconds
- SSH info wait: 45.557 seconds
- Setup: 45.003 seconds
- CUDA smoke: 3.976 seconds
- Runtime package install: 2.322 seconds
- Tiny eval command: 73.459 seconds
- Output sync: 0.918 seconds

From `outputs/leverage-gpt-oss-20b-tiny-eval/metadata.json`:

- Model load time inside tiny eval: 22.131 seconds
- Tiny eval total command time: 64.501 seconds
- Peak allocated GPU memory: 14.988GB
- Runtime: `torch 2.8.0+cu128`
- CUDA device: `NVIDIA GeForce RTX 5090`

Approximate cost: `$0.05`.

## Interpretation

The gpt-oss path is ready for a slightly larger tiny benchmark only after one
more choice: keep using the project-owned extractor path, or switch to a
serving path that exposes Harmony final content more structurally. The current
`lm-eval` path should not be used directly until it can score only the final
answer channel.
