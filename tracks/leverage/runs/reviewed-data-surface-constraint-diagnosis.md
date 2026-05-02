# Reviewed Data Surface Constraint Diagnosis

Date: 2026-05-02

## Goal

Check whether the reviewed SFT dataset contains data-side biases that explain
the `Qwen/Qwen3.5-9B` LoRA adapter regression on IFEval.

This is a local dataset analysis. It did not run paid inference or training.

## Input

- Dataset:
  `tracks/leverage/datasets/reviewed-instructions/bootstrap.jsonl`
- Rows: 1083
- Related benchmark note:
  `tracks/leverage/runs/lm-harness-ifeval-sample-diagnosis.md`

## Assistant Length Distribution

Assistant response word counts:

| statistic | words |
| --- | ---: |
| min | 1 |
| p10 | 4 |
| p25 | 10 |
| median | 20 |
| p75 | 36 |
| p90 | 58 |
| max | 205 |
| mean | 27.46 |

Threshold counts:

| threshold | rows | share |
| --- | ---: | ---: |
| <= 20 words | 591 | 54.6% |
| <= 40 words | 853 | 78.8% |
| <= 60 words | 991 | 91.5% |
| <= 100 words | 1059 | 97.8% |
| >= 300 words | 0 | 0.0% |
| >= 500 words | 0 | 0.0% |

Interpretation: the reviewed dataset strongly trains short answers. It does
not teach long-form instruction following. This matches the IFEval sample
diagnosis, where adapter-only failures included 300-, 500-, and 600-word
minimum constraints.

## Length By Capability

| capability | rows | mean words | median words | <= 60 words |
| --- | ---: | ---: | ---: | ---: |
| coding | 191 | 26.3 | 21 | 97.9% |
| instruction_following | 204 | 7.5 | 6 | 100.0% |
| knowledge_qa | 153 | 40.9 | 37 | 77.8% |
| reasoning | 257 | 40.2 | 33 | 84.0% |
| summarization_transformation | 139 | 17.6 | 19 | 100.0% |
| tool_use | 139 | 29.8 | 23 | 90.6% |

The shortest area is `instruction_following`. That is useful for exact-answer
training, but it is a poor proxy for IFEval's multi-constraint long-form
instruction following.

## Length By Task Shape

| task_shape | rows | mean words | median words | <= 60 words |
| --- | ---: | ---: | ---: | ---: |
| comparison | 23 | 32.1 | 25 | 91.3% |
| debugging | 57 | 35.6 | 32 | 96.5% |
| decision | 201 | 44.6 | 36 | 82.1% |
| direct_answer | 253 | 9.8 | 8 | 100.0% |
| explanation | 156 | 52.6 | 45 | 72.4% |
| extraction_transformation | 102 | 9.0 | 7 | 100.0% |
| implementation | 117 | 19.2 | 14 | 98.3% |
| planning | 52 | 38.7 | 42 | 86.5% |
| rewrite | 122 | 17.5 | 19 | 100.0% |

`direct_answer`, `extraction_transformation`, `implementation`, and `rewrite`
are almost entirely short-output shapes. The current mix does not provide much
pressure to satisfy long outputs while maintaining constraints.

## Surface Constraint Coverage

Prompt mentions that resemble IFEval surface constraints:

| constraint family | rows | share |
| --- | ---: | ---: |
| exact length | 42 | 3.9% |
| max length | 2 | 0.2% |
| no punctuation or comma | 44 | 4.1% |
| case constraint | 30 | 2.8% |
| forbidden word | 29 | 2.7% |
| JSON format | 68 | 6.3% |
| Markdown or highlighted format | 36 | 3.3% |
| long minimum length | 0 | 0.0% |

The dataset has some exact-format and JSON rows, but almost no long-form
surface-constraint rows. The important missing combination is not "JSON" or
"punctuation" by itself; it is multiple constraints over a longer answer.

## Output Format Signals

Assistant output features:

| feature | rows | share |
| --- | ---: | ---: |
| JSON-like output | 60 | 5.5% |
| fenced code block | 104 | 9.6% |
| contains comma | 647 | 59.7% |
| all-caps answer with at least 20 words | 0 | 0.0% |
| <= 60 words | 991 | 91.5% |
| >= 300 words | 0 | 0.0% |
| >= 500 words | 0 | 0.0% |

This supports the IFEval failure pattern:

- the dataset can teach short JSON extraction
- the dataset can teach short exact answers
- it does not teach long all-caps answers
- it does not teach long no-comma answers
- it does not teach long minimum-word-count answers

## Examples

Longest reviewed assistant answers are still short relative to IFEval:

| row | capability | task_shape | words | note |
| --- | --- | --- | ---: | --- |
| `instr_0063` | reasoning | explanation | 205 | GPU SSH readiness risk explanation |
| `instr_0527` | coding | debugging | 205 | external API timeout explanation |
| `instr_0635` | knowledge_qa | explanation | 194 | LoRA explanation |
| `instr_0488` | reasoning | decision | 179 | randomized nonprofit design choice |
| `instr_0493` | reasoning | decision | 161 | mobile app rollout decision |

Representative short surface-constraint rows:

| row | prompt type | assistant words |
| --- | --- | ---: |
| `instr_0018` | exact lowercase word, no punctuation | 1 |
| `instr_0109` | exactly four words, no punctuation | 4 |
| `instr_0115` | exactly three words, no punctuation | 3 |
| `instr_0033` | JSON only | 4 |
| `instr_0114` | JSON, no Markdown fence | 4 |

These are useful tests of exactness, but they do not cover IFEval-style
long-form constraint retention.

## Interpretation

The dataset diagnosis is consistent with the benchmark diagnosis.

The adapter likely learned a short, direct local-answer style. That is useful
for project-owned exact-answer tasks, but it weakens general instruction
following when the task requires long output while preserving surface
constraints.

The current data does not support a claim that the LoRA should improve general
IFEval behavior. It has too little long-form instruction-following coverage and
too few examples that combine length, punctuation, case, forbidden-word, and
format constraints.

## Recommendation

Do not start another full LoRA run yet.

Before the next capability-seeking LoRA, add reviewed training data that covers:

- long minimum-length answers, starting around 300-600 words
- long answers with no-comma or no-punctuation constraints
- long all-caps or case-constrained answers
- JSON answers that avoid prompt-forbidden words in both keys and values
- multi-constraint prompts where satisfying one constraint cannot be achieved
  by making the answer very short

Keep this data general-purpose. Do not copy IFEval prompts or tune directly to
IFEval samples.
