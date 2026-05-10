# Qwen3.5-9B Adapter Regression Analysis

Date: 2026-05-10

## Goal

Explain why `Qwen/Qwen3.5-9B` LoRA adapters improved the project-owned eval
while regressing on IFEval, and what changed between the old adapter and the
1,216-row adapter.

This is a local artifact analysis. It did not run new paid inference, training,
or benchmarks.

## Inputs

- Old full IFEval note:
  `tracks/leverage/runs/lm-harness-ifeval-full.md`
- 1,216-row full IFEval note:
  `tracks/leverage/runs/lm-harness-ifeval-adapter-1216-full.md`
- Old limit-50 sample note:
  `tracks/leverage/runs/lm-harness-ifeval-sample-diagnosis.md`
- 1,216-row limit-50 sample note:
  `tracks/leverage/runs/lm-harness-ifeval-adapter-1216-limit50.md`
- Old adapter training note:
  `tracks/leverage/runs/qwen35-9b-full-gradient-checkpointing.md`
- 1,216-row adapter project-owned eval note:
  `tracks/leverage/runs/qwen35-9b-baseline-1216-eval.md`
- Saved limit-50 samples:
  - `outputs/leverage-lm-harness-ifeval-samples/base-limit50/base/Qwen__Qwen3.5-9B/samples_ifeval_2026-05-02T17-20-28.246317.jsonl`
  - `outputs/leverage-lm-harness-ifeval-samples/adapter-limit50/adapter/outputs__leverage-sft-qwen35-9b__lora-adapter/samples_ifeval_2026-05-02T17-33-33.345518.jsonl`
  - `outputs/leverage-lm-harness-ifeval-samples-20260509/adapter-limit50/adapter/outputs__leverage-sft-qwen35-9b__lora-adapter/samples_ifeval_2026-05-09T15-44-21.098729.jsonl`

## Full IFEval Scores

| model | prompt strict | prompt loose | instruction strict | instruction loose |
| --- | ---: | ---: | ---: | ---: |
| base | 0.8410 | 0.8817 | 0.8885 | 0.9185 |
| old adapter | 0.7819 | 0.8133 | 0.8501 | 0.8717 |
| 1,216-row adapter | 0.7800 | 0.8041 | 0.8429 | 0.8609 |

The full benchmark does not support a recovery claim. The 1,216-row adapter is
slightly worse than the old adapter on all four full IFEval metrics.

## Project-Owned Eval Signal

The project-owned held-out eval moved in the opposite direction from IFEval.

| model | overall | leverage-smoke | project-judgment |
| --- | ---: | ---: | ---: |
| base before old adapter | 15/30 | 9/12 | 6/18 |
| old adapter | 17/30 | 9/12 | 8/18 |
| base before 1,216-row adapter | 15/30 | 9/12 | 6/18 |
| 1,216-row adapter | 18/30 | 9/12 | 9/18 |

The project-owned gain is real in the local harness, but it is small and
distribution-specific. In the old adapter analysis, only four project-owned
tasks changed pass/fail status:

| task | suite | capability | change | reason |
| --- | --- | --- | --- | --- |
| `summary_mission` | leverage-smoke | summarization_transformation | fail -> pass | adapter included required phrase `two tracks` |
| `coding_sql_count` | leverage-smoke | coding | pass -> fail | adapter wrapped SQL in a fenced code block |
| `pj_exp_001` | project-judgment | reasoning | fail -> pass | adapter matched the expected dataset-shard blocking issue |
| `pj_exp_003` | project-judgment | reasoning | fail -> pass | adapter returned `REPEAT` exactly |

Two of those wins are close to explicit reviewed training rows:

- `pj_exp_003` asks for `SHIP`, `REPEAT`, or `DROP` when a promising ablation has
  only one seed and no held-out evaluation. Training row `instr_0016` from
  `lt_seed_053` teaches the same decision shape and answer: `REPEAT`.
- `pj_exp_001` asks whether a result should be promoted when it used a smaller
  dataset shard. Training row `instr_0210` from `lt_seed_373` teaches the same
  causal caution pattern: the dataset changed, so causality is not proven.

This means the project-owned improvement is not strong evidence of broad
capability. It is plausibly evidence that the adapter learned local project
phrasing and decision patterns.

## Output-Format Signal

The old adapter failed `coding_sql_count` by returning fenced SQL:

````text
```sql
SELECT COUNT(*) FROM users;
```
````

The base model returned plain SQL:

```text
SELECT COUNT(*) FROM users;
```

At that point, the training data contained 104 assistant messages with fenced
code blocks. Those were concentrated in coding:

| slice | rows | fenced rows |
| --- | ---: | ---: |
| all training rows | 1083 | 104 |
| coding capability | 191 | 100 |
| implementation task_shape | 117 | 97 |

So the SQL regression is consistent with the adapter learning a code-answer
style that is often appropriate in the training data but wrong for strict
`SQL only` or exact-regex tasks.

## Limit-50 Instruction Failures

The saved limit-50 samples cover the same 50 prompts for base, old adapter, and
1,216-row adapter. Prompt-level strict pass counts:

| model | prompt strict |
| --- | ---: |
| base | 46/50 |
| old adapter | 42/50 |
| 1,216-row adapter | 44/50 |

Instruction-level strict pass counts:

| model | instruction strict |
| --- | ---: |
| base | 71/76 |
| old adapter | 66/76 |
| 1,216-row adapter | 69/76 |

On this sample only, the 1,216-row adapter partially recovers the old adapter
regression. This conflicts with the full benchmark, so the sample should be
used only to inspect failure modes.

## Failure Families

Instruction-level failures by family on the limit-50 sample:

| family | base failures | old adapter failures | 1,216-row failures |
| --- | ---: | ---: | ---: |
| `length_constraints` | 1 | 5 | 2 |
| `keywords` | 2 | 2 | 4 |
| `change_case` | 1 | 2 | 1 |
| `punctuation` | 0 | 1 | 0 |
| `startend` | 1 | 0 | 0 |

The old adapter's most visible sample regression was short output. It failed
four of eight `length_constraints:number_words` instructions. The 1,216-row
adapter recovered two of those word-count failures, but it introduced more
keyword-counting failures.

## Paired Prompt Changes

Prompt-level strict changes on the 50 shared prompts:

| comparison | pass -> fail | fail -> pass |
| --- | ---: | ---: |
| base -> old adapter | 5 | 1 |
| old adapter -> 1,216-row adapter | 2 | 4 |
| base -> 1,216-row adapter | 2 | 0 |

The 1,216-row adapter improves over the old adapter on the sample, but it still
does not add any prompt-level wins over base. Its two base-to-adapter losses are
both surface-constraint failures:

- prompt 9: at least 500 words, required keywords, no commas
- prompt 36: use `war` at least eight times and `peace` at least ten times

## Representative Examples

Prompt 0 asks for a 300+ word comma-free summary with three highlighted
sections. The old adapter produced 140 words and failed the word-count
constraint. The 1,216-row adapter produced 308 words and passed.

Prompt 49 asks for at least 600 words in a presidential style. The old adapter
produced 564 words and failed. The 1,216-row adapter produced 654 words and
passed.

Prompt 36 asks for the words `war` and `peace` at least eight and ten times. The
old adapter passed by repeating the target words heavily. The 1,216-row adapter
returned a more natural answer and failed the `peace` frequency requirement.

Prompt 23 asks for a logic quiz where the letter `t` appears at most once. The
old adapter passed by giving a very short answer. The 1,216-row adapter returned
a long normal quiz and failed the letter-frequency constraint.

## Interpretation

The 1,216-row surface-constraint batch did move the adapter in the intended
direction on some long-answer constraints. It reduced the old adapter's
short-output failure mode on the small sample.

The full IFEval result still got worse, so the data addition was not enough.
The likely issue is not simply row count. The adapter is learning a more natural
project-answering style, but IFEval rewards exact constraint accounting even
when the answer becomes repetitive or unnatural.

Current conclusion:

- Do not claim that more reviewed data is improving general instruction
  following.
- Do not use project-owned eval improvement alone as the training success
  criterion.
- Keep IFEval as a guardrail, but do not tune directly to IFEval prompts.
- The next data or training change should explicitly target generalized
  constraint accounting: word-count budgets, keyword frequency, letter
  frequency, forbidden words, case constraints, and format constraints.

## Next Step

Before another full Qwen3.5-9B LoRA run, make one controlled change and measure
it against both project-owned evals and IFEval:

- add a small synthetic-but-general constraint-accounting slice that does not
  copy IFEval prompts, or
- reduce LoRA/SFT strength and check whether IFEval recovers without losing the
  project-owned gains.

If choosing another benchmark axis, prefer a small reasoning or coding benchmark
after this IFEval diagnosis. That would test whether the regression is limited
to surface constraints or reflects broader instruction-tuning damage.
