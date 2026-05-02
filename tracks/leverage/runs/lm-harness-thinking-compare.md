# LM Harness Thinking Compare

## Goal

Compare the two valid IFEval modes for Qwen:

- thinking off
- thinking on with `think_end_token` stripping

Do not use thinking on without stripping for IFEval because IFEval scores the
visible answer against formatting constraints.

## Command

RunPod:

- GPU: `NVIDIA A40`
- Cloud: Secure
- Image: `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`
- CUDA filter: `12.8`
- Cost rate: `$0.44/hr`

Remote command:

```bash
uv pip install lm-eval peft transformers accelerate datasets sentencepiece protobuf langdetect immutabledict &&
uv run python -m llm.leverage.evaluate_lm_harness \
  --config tracks/leverage/configs/leverage-sft-qwen35-9b.toml \
  --task ifeval \
  --variant base \
  --batch-size 1 \
  --limit 2 \
  --log-samples \
  --no-enable-thinking \
  --output-root outputs/leverage-lm-harness-thinking/off &&
uv run python -m llm.leverage.evaluate_lm_harness \
  --config tracks/leverage/configs/leverage-sft-qwen35-9b.toml \
  --task ifeval \
  --variant base \
  --batch-size 1 \
  --limit 2 \
  --log-samples \
  --enable-thinking \
  --think-end-token "</think>" \
  --output-root outputs/leverage-lm-harness-thinking/on-strip
```

## Result

Both modes completed.

| mode | prompt strict | instruction strict | response tokens | generation time |
| --- | ---: | ---: | --- | --- |
| thinking off | 1.0 | 1.0 | 458, 140 | 2 examples in 29.9s |
| thinking on + strip | 0.0 | 0.5 | 1280, 1280 | 2 examples in 112.9s |

This is a `--limit 2` smoke, not a benchmark score.

## Interpretation

`enable_thinking=False` works for Qwen through lm-evaluation-harness. The
rendered prompt still contains an empty think block:

```text
<|im_start|>assistant
<think>

</think>
```

The model then emits the final answer directly. This is much faster and matches
IFEval's visible-output contract.

`enable_thinking=True,think_end_token="</think>"` did not work well for this
test. The model generated thinking-style text and did not reach `</think>`
within the 1280-token cap, so there was no final answer for IFEval to score.
The stripping option is correct in principle, but it does not reduce generation
cost and only helps if the model reaches the thinking terminator.

## Timing

From `outputs/leverage-lm-harness-thinking/runpod-timings.json`:

- Total wall time: 644.496s
- Remote command: 524.896s
- SSH readiness: 45.417s
- Setup: 41.496s
- CUDA smoke: 24.845s

Approximate RunPod cost:

```text
644.496s / 3600 * $0.44/hr = $0.08
```

## Cleanup

- `runpodctl pod delete qsrviq4jaayv50` returned deleted.
- `runpodctl pod list -o json` returned `[]`.

## Next Decision

Use `--no-enable-thinking` as the default IFEval mode for Qwen in this project.
Keep `--enable-thinking --think-end-token "</think>"` as an optional diagnostic
mode, not the default.
