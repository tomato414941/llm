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

Good work for this track:

- evaluate small open models on simple QA, instruction following, summarization,
  reasoning, and coding-style tasks
- compare prompting strategies and decoding settings
- run quantized local inference when feasible
- try LoRA or SFT on a narrow dataset
- distill outputs from a stronger model into a smaller one
- design task-specific evals that make model changes measurable

Success criteria:

- capability is measured on tasks closer to general LLM use
- experiments are cheaper than training from scratch
- data, prompts, and evaluation rules are versioned
- improvements are compared against a baseline model

No-dependency JSONL evaluator:

- input is one or more task JSONL files plus a saved prediction JSONL file
- `evals/leverage_smoke.jsonl` is the smoke layer for evaluator wiring and
  prediction format checks
- `evals/project_judgment_v0.jsonl` is the project-judgment layer for more
  meaningful leverage-track comparisons
- pass multiple layers with repeated `--tasks` arguments
- output includes a local CSV scoring file and, when requested, a
  `--summary-output` CSV for rollups suitable for review and comparison
- it scores saved predictions only; it does not run inference, use RunPod,
  download models, call APIs, or fetch datasets
- use it before any paid or networked run to make the eval target explicit

First real-model target:

- `Qwen/Qwen3.5-35B-A3B` is the first planned leverage target because it is a
  strong Qwen open-weight model with Apache 2.0 licensing on Hugging Face
- collect predictions through an OpenAI-compatible API with
  `llm.leverage.collect_openai`
- when using RunPod, host the same model behind an OpenAI-compatible endpoint
  and run only the committed eval prompts first
- provider-specific API terms, authentication, and costs are separate from this
  repository and must stay out of commits

First RunPod spike:

- objective: verify that `Qwen/Qwen3.5-35B-A3B` can serve this project's eval
  prompts and produce saved JSONL predictions
- workload: inference only; no training or fine-tuning
- model server: official `vllm/vllm-openai` OpenAI-compatible API image
- first model artifact: `Qwen/Qwen3.5-35B-A3B-FP8`
- first GPU target: 1x `NVIDIA A100 80GB PCIe`
- first cost ceiling: `$5.00`
- initial context: keep the server context well below the model maximum unless
  a long-context test is the explicit objective
- input files: committed eval JSONL files only
- output files: one prediction JSONL file and evaluator CSV summaries
- stopping condition: all committed leverage eval tasks have saved predictions,
  or the serving setup fails clearly
- cleanup: remove the pod immediately after syncing results and verify no active
  pods remain

Non-goals:

- treating an open model as an opaque demo only
- spending GPU time before the eval target is clear
- mixing leverage results with from-scratch claims

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

- checkpoint: `checkpoints/pico_gpt2_tinyshakespeare.pt`
- data: `data/processed/tinyshakespeare_bpe_500.pt`
- parameters: 235,892
- checkpoint step: 2,999
- validation loss: about 3.34
- validation perplexity: about 28.28

This baseline demonstrates the training pipeline and Shakespeare-style local
structure. It is not a general-purpose LLM.

## Next Decisions

1. From-scratch next step: define a nano GPT-2 smoke config and estimate cost
   before any paid run.
2. Leverage next step: run the Qwen3.5 collection path against the committed
   eval layers through either a hosted API or a short RunPod inference spike.
3. Shared next step: keep results comparable with explicit baselines, configs,
   and observation notes.
