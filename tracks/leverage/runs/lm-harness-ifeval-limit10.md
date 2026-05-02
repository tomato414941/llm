# LM Harness IFEval Limit-10

## Goal

Check whether EleutherAI `lm-evaluation-harness` can evaluate both the
`Qwen/Qwen3.5-9B` base model and the trained LoRA adapter through the same
external benchmark path.

This is a usability smoke, not a real benchmark score. `--limit 10` was used
because full `ifeval` has 541 generation requests and was too slow for a first
RunPod check on one A40.

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
  --run both \
  --limit 10
```

## Result

Both base and adapter completed.

| model | prompt strict | prompt loose | instruction strict | instruction loose |
| --- | ---: | ---: | ---: | ---: |
| `Qwen/Qwen3.5-9B` | 0.10 | 0.10 | 0.4444 | 0.4444 |
| `Qwen/Qwen3.5-9B` + LoRA adapter | 0.20 | 0.20 | 0.5000 | 0.5000 |

Treat the apparent adapter improvement as directional only. Ten examples is
too small for a real benchmark claim.

## Timing

From `outputs/leverage-lm-harness/runpod-timings.json`:

- Total wall time: 1704.765s
- Remote benchmark command: 1600.646s
- SSH readiness: 34.136s
- Repo and adapter sync: 4.780s
- Setup: 42.994s

Approximate RunPod cost:

```text
1704.765s / 3600 * $0.44/hr = $0.21
```

## Findings

- PEFT adapter loading works with `lm-evaluation-harness`:
  `pretrained=Qwen/Qwen3.5-9B,peft=outputs/leverage-sft-qwen35-9b/lora-adapter`.
- `ifeval` required extra packages not installed by `lm-eval==0.4.11` in this
  environment: `langdetect` and `immutabledict`.
- Full `ifeval` is not a good first usability benchmark on this setup. One
  base example took about 55-60s, and one adapter example took about 60-95s.
- `--limit 10` is useful as a smoke, but not as a publishable score.

## Cleanup

- `runpodctl pod delete xtinswlz1ju19w` returned deleted.
- `runpodctl pod list -o json` returned `[]`.

## Next Decision

Keep the adapter as the source artifact and continue using
`pretrained=<base>,peft=<adapter>` for compatible benchmark runners.

For a more usable next external benchmark, prefer a shorter benchmark or a task
with lower generation cost before attempting full `ifeval`.
