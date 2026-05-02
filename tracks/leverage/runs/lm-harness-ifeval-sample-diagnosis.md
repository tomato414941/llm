# LM Harness IFEval Sample Diagnosis

Date: 2026-05-02

## Goal

Compare `Qwen/Qwen3.5-9B` base and the current LoRA adapter on a small IFEval
sample with `--log-samples`, so the adapter regression seen in full IFEval can
be inspected at the sample level.

This is a diagnostic limited run, not a benchmark claim.

## Setup

- Task: `ifeval`
- Limit: 50
- Thinking mode: `--no-enable-thinking`
- Batch size: 4
- Backend: EleutherAI `lm-evaluation-harness` `hf`
- GPU: `NVIDIA A40`
- Cloud: RunPod Secure Cloud
- Image: `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`
- CUDA filter: `12.8`, `12.9`, `13.0`
- Cost rate: `$0.44/hr`

## Results

| model | prompt strict | prompt loose | instruction strict | instruction loose |
| --- | ---: | ---: | ---: | ---: |
| `Qwen/Qwen3.5-9B` | 0.9200 | 0.9000 | 0.9342 | 0.9342 |
| `Qwen/Qwen3.5-9B` + LoRA adapter | 0.8400 | 0.8200 | 0.8684 | 0.8684 |

Delta, adapter minus base:

- prompt strict: -0.0800
- prompt loose: -0.0800
- instruction strict: -0.0658
- instruction loose: -0.0658

The limited sample matches the full IFEval direction: the adapter is worse.

## Sample-Level Delta

Prompt-level strict changes:

- base pass -> adapter fail: 5 samples
- base fail -> adapter pass: 1 sample

Adapter-only failures:

| doc_id | instruction types | primary failure mode | evidence |
| ---: | --- | --- | --- |
| 0 | no comma, markdown highlighted sections, at least 300 words | length under-run | adapter produced 140 words; base produced 371 words |
| 6 | all caps, two labeled sections | case violation | adapter used lowercase text; base used no lowercase letters |
| 9 | required keywords, at least 500 words, no comma | length and punctuation under-run | adapter produced 278 words and 5 commas; base produced 633 words and no commas |
| 45 | JSON format, forbidden word `nickname` | forbidden keyword in JSON key | adapter returned `{"nickname": "Staffy"}`; base avoided the word |
| 49 | at least 600 words | length under-run | adapter produced 573 words; base produced 942 words |

Adapter-only pass:

| doc_id | instruction types | likely reason |
| ---: | --- | --- |
| 23 | letter `t` at most once | adapter answered much shorter, with one `t`; base exceeded the limit |

## Interpretation

The sample does not show a broad semantic failure. It points to stricter
surface-constraint regressions:

- length constraints
- case constraints
- punctuation constraints
- forbidden keyword constraints

The adapter tends to produce shorter and more direct answers. That helps on one
letter-frequency case, but hurts length-heavy IFEval prompts and exact surface
constraints.

This is consistent with the project-owned eval split analysis: the adapter
learned local answer style and some local decision patterns, but did not improve
general instruction-following.

## Timing

Base:

- Total RunPod wall time: `648.600s`
- Benchmark command: `542.230s`
- Observed generation interval: `373.265s`
- Approximate cost: `648.600 / 3600 * $0.44 = about $0.08`

Adapter:

- Total RunPod wall time: `774.844s`
- Benchmark command: `629.557s`
- Observed generation interval: `369.489s`
- Approximate cost: `774.844 / 3600 * $0.44 = about $0.09`

Combined approximate cost: about `$0.17`.

The prior estimate of `15-25m` per variant was conservative. Actual wall time
was about `10.8m` for base and `12.9m` for adapter.

## Artifacts

Ignored output files:

- `outputs/leverage-lm-harness-ifeval-samples/base-limit50/base/Qwen__Qwen3.5-9B/results_2026-05-02T17-20-28.246317.json`
- `outputs/leverage-lm-harness-ifeval-samples/base-limit50/base/Qwen__Qwen3.5-9B/samples_ifeval_2026-05-02T17-20-28.246317.jsonl`
- `outputs/leverage-lm-harness-ifeval-samples/base-limit50/base-limit50-timing.json`
- `outputs/leverage-lm-harness-ifeval-samples/base-limit50/runpod-timings.json`
- `outputs/leverage-lm-harness-ifeval-samples/adapter-limit50/adapter/outputs__leverage-sft-qwen35-9b__lora-adapter/results_2026-05-02T17-33-33.345518.json`
- `outputs/leverage-lm-harness-ifeval-samples/adapter-limit50/adapter/outputs__leverage-sft-qwen35-9b__lora-adapter/samples_ifeval_2026-05-02T17-33-33.345518.jsonl`
- `outputs/leverage-lm-harness-ifeval-samples/adapter-limit50/adapter-limit50-timing.json`
- `outputs/leverage-lm-harness-ifeval-samples/adapter-limit50/runpod-timings.json`

## Cleanup

- Base pod `t0asbowaneg4it` was deleted by the runner.
- Adapter pod `q8l2x7ssbyrrb9` was deleted by the runner.
- Final `runpodctl pod list -o json` returned `[]`.

## Next Step

Do not start another LoRA run yet.

The next useful change is data-side: reduce training bias toward short local
answers and add or preserve examples that teach exact surface constraints
without overfitting to IFEval.
