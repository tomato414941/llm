# Leverage SFT Smoke Failure Triage

Date: 2026-04-28

Input artifacts:

- `outputs/leverage-sft-smoke/post-training-predictions.jsonl`
- `outputs/leverage-sft-smoke/post-training-scores.csv`
- `outputs/leverage-sft-smoke/post-training-summary.csv`

These generated artifacts are intentionally ignored by git. This note is the
tracked triage record.

## Summary

The first LoRA smoke did not improve capability. More importantly, the eval
failures show that the next blocker is not "train longer". The dominant issue
is output formatting from the local generation path.

Overall scores:

```csv
model,task_count,passed_count,pass_rate
qwen3-0.6b-base,30,3,0.100
qwen3-0.6b-lora-smoke,30,2,0.067
```

The cleanup effect was verified by rescoring the existing saved predictions:

```bash
uv run python -m llm.leverage.evaluate_sft_adapter \
  --rescore-predictions outputs/leverage-sft-smoke/post-training-predictions.jsonl
```

If generated responses are post-processed to strip `<think>...</think>`, orphan
`</think>` tags, and leading `assistant` / `user` role labels, the score changes
to:

```csv
model,passed_count_before,passed_count_after_strip,recovered_tasks
qwen3-0.6b-base,3,4,qa_capital_france
qwen3-0.6b-lora-smoke,2,8,instruction_json; reasoning_order; coding_sql_count; qa_author; instruction_lowercase; pj_repo_001
```

## Classification

### 1. Prompt / Decoding / Output Cleanup

This is the top priority.

Symptoms:

- Many responses include `<think>`, `</think>`, or leading role labels.
- Several semantically correct answers fail exact or regex scoring only because
  the response includes wrapper text.
- The adapter especially tends to emit `assistant` before the final answer.

Examples:

- `qa_author`: adapter produced `assistant ... William Shakespeare`; expected
  exact `William Shakespeare`.
- `instruction_lowercase`: adapter produced `assistant ... ready`; expected
  exact `ready`.
- `coding_sql_count`: adapter produced `assistant ... SELECT COUNT(*) FROM users;`;
  regex would pass after stripping wrapper text.

Decision:

Add a deterministic response-cleaning step in the local SFT adapter eval path
before scoring. Also inspect whether `tokenizer.apply_chat_template` is causing
role labels to be learned or continued in generation.

### 2. Eval Strictness

Some failures are legitimate eval-contract friction rather than model-quality
signals.

Symptoms:

- The model gives a mostly correct answer but exact scoring rejects extra
  punctuation or duplicated final answers.
- Regex tasks reject otherwise acceptable answers wrapped in code fences or
  duplicated with thinking text.

Examples:

- `qa_capital_france`: base produced `Paris` after an orphan `</think>`.
- `instruction_json`: both models include valid JSON plus wrapper text.
- `instruction_bullets`: content is close, but regex is strict about exact
  bullet formatting and no wrapper text.

Decision:

Do not loosen every eval yet. First normalize generation output. After that,
review only tasks that still fail despite a clean final answer.

### 3. Data / Capability Gap

Project-judgment tasks are still mostly real failures after cleanup.

Symptoms:

- The model does not know the desired project policy labels or operational
  judgment patterns.
- The adapter was trained on only 10 rows, so it should not be expected to
  learn the 18-task project-judgment surface.

Examples:

- `pj_exp_003`: adapter answered `DROP`; expected `REPEAT`.
- `pj_cost_002`: adapter recommends keeping an idle A100 running.
- `pj_track_001`: adapter answered `operations`; expected `research`.
- `pj_eval_002`: both models answered `exact`; expected `contains_all`.

Decision:

After output cleanup is fixed, build the next reviewed dataset slice around
project-judgment examples. Keep it small, but cover experiment controls,
RunPod cost policy, track classification, repo hygiene, and eval design.

## Next Action

Output cleanup is now implemented inside `llm.leverage.evaluate_sft_adapter`.
Before spending on another RunPod run, use `--rescore-predictions` to verify
whether a saved prediction file is failing because of formatting or because the
model answer is actually wrong.

The next data iteration should focus on project-judgment examples. After cleanup,
the smoke adapter still passes only `1/18` project-judgment tasks.
