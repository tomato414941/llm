# Roadmap

This project has two tracks. They serve different goals and should not be judged
with the same success criteria.

## Goal

Build practical understanding and tooling for moving toward general-purpose
language models under limited compute.

This does not mean training a GPT-3-scale model from scratch. It means learning
the core mechanics, building reliable experiment infrastructure, and using
existing open models when the goal is capability rather than implementation
understanding.

## Track 1: From Scratch

Purpose: understand how decoder-only language models work by implementing and
training small models directly.

Good work for this track:

- tokenizer and data preparation
- causal attention and GPT-style Transformer blocks
- checkpoint, resume, sampling, evaluation, and observation tools
- pico or nano GPT-2 scale experiments
- small controlled comparisons of architecture, tokenizer, and training choices

Success criteria:

- code is readable and tested
- runs are reproducible
- experiments answer a narrow question
- RunPod is avoided unless a short, bounded run answers a specific question

Non-goals:

- claiming general-purpose capability
- long training runs without a comparison hypothesis
- scaling just because more compute is available

## Track 2: Leverage

Purpose: work with models that already have broad language capability, then
evaluate, adapt, distill, or operate them under practical constraints.

The leverage track should favor scalable learning loops over hand-crafted
rules. The intended loop is:

```text
teacher generation -> structural filter -> model judge -> student training -> held-out eval
```

Small reviewed datasets are bootstrap material for this loop, not the main
source of capability.

Good work for this track:

- evaluate small open models on simple QA, instruction following, summarization,
  reasoning, and coding-style tasks
- compare prompting strategies and decoding settings
- run quantized local inference when feasible
- try LoRA or SFT on a narrow dataset
- distill outputs from a stronger model into a smaller one
- design task-specific evals that make model changes measurable
- add model-judge and preference steps that scale beyond manual row review

Success criteria:

- capability is measured on tasks closer to general LLM use
- experiments are cheaper than training from scratch
- data, prompts, and evaluation rules are versioned
- improvements are compared against a baseline model
- human effort focuses on seed distribution, spot checks, and metric review
  rather than hand-labeling most rows

No-dependency JSONL evaluator:

- input is one or more task JSONL files plus a saved prediction JSONL file
- `tracks/leverage/evals/leverage-smoke.jsonl` is the smoke layer for evaluator wiring and
  prediction format checks
- `tracks/leverage/evals/project-judgment.jsonl` is the project-judgment layer for more
  meaningful leverage-track comparisons
- pass multiple layers with repeated `--tasks` arguments
- output includes a local CSV scoring file and, when requested, a
  `--summary-output` CSV for rollups suitable for review and comparison
- it scores saved predictions only; it does not run inference, use RunPod,
  download models, call APIs, or fetch datasets
- use it before any paid or networked run to make the eval target explicit

OpenAI-compatible collection:

- collect predictions through any OpenAI-compatible chat completions endpoint
  with `llm.leverage.collect_openai`
- RunPod is only one possible host for that endpoint; hosted APIs and local
  servers use the same saved-prediction flow
- provider-specific API terms, authentication, and costs are separate from this
  repository and must stay out of commits

One-shot RunPod spike:

- objective: verify that a selected open model can serve this project's eval
  prompts and produce saved JSONL predictions
- workload: inference only; no training or fine-tuning
- model server: temporary OpenAI-compatible HTTP API on RunPod
- default model artifact: `Qwen/Qwen3-14B-FP8`, override with `--model` when a
  different compatible model is the target
- default GPU target: 1x `NVIDIA GeForce RTX 4090`
- default cost ceiling: `$2.00`
- initial context: keep the server context well below the model maximum unless
  a long-context test is the explicit objective
- input files: committed eval JSONL files only
- output files: local prediction JSONL file and evaluator CSV summaries
- stopping condition: all committed leverage eval tasks have saved predictions,
  or the serving setup fails clearly
- cleanup: remove the pod immediately after syncing results and verify no active
  pods remain

Non-goals:

- treating an open model as an opaque demo only
- spending GPU time before the eval target is clear
- mixing leverage results with from-scratch claims
- polishing a tiny hand-written dataset as if it were the source of capability

## RunPod Policy

RunPod is a paid external resource. Use it only when the expected value is clear.

Before starting a RunPod job, write down:

- objective
- expected runtime
- GPU type and cost ceiling
- files that will be uploaded
- stopping condition
- expected output files

After every RunPod job:

- sync results back
- remove the pod
- verify that no pod remains
- record the result or failure

## Current Baseline

The current from-scratch baseline is `pico_gpt2_tinyshakespeare`:

- checkpoint: `runs/from-scratch/pico-gpt2-tinyshakespeare/checkpoint.pt`
- data: `data/from-scratch/processed/tinyshakespeare_bpe_500.pt`
- parameters: 235,892
- checkpoint step: 2,999
- validation loss: about 3.34
- validation perplexity: about 28.28

This baseline demonstrates the training pipeline and Shakespeare-style local
structure. It is not a general-purpose LLM.

## Next Decisions

1. From-scratch next step: define a nano GPT-2 smoke config and estimate cost
   before any paid run.
2. Leverage next step: run the OpenAI-compatible collection path against the
   committed eval layers through either a hosted API or a short RunPod inference
   spike.
3. Shared next step: keep results comparable with explicit baselines, configs,
   and observation notes.
