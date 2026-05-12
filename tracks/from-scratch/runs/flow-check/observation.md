# Checkpoint Observation

## Inputs

- checkpoint: `tracks/from-scratch/runs/flow-check/checkpoint.pt`
- tokens: `tracks/from-scratch/data/processed/tinyshakespeare_bpe_500.pt`
- output: `tracks/from-scratch/runs/flow-check/observation.md`

## Checkpoint

- step: 19
- metadata: `{'run_id': 'flow_check', 'config_path': 'tracks/from-scratch/runs/flow-check/config.toml', 'manifest_path': 'tracks/from-scratch/data/manifests/tinyshakespeare-bpe-500.toml', 'max_iters': 20, 'batch_size': 8, 'learning_rate': 0.001, 'weight_decay': 0.1, 'warmup_iters': 0, 'lr_decay_iters': 3000, 'min_learning_rate': '', 'seed': 1337, 'device': 'cpu', 'parameter_count': 42804, 'tokenizer': 'bpe', 'tokenizer_path': 'data/processed/tinyshakespeare_bpe_500.json', 'input': 'data/raw/tinyshakespeare.txt', 'byte_count': 1115394, 'token_count': 574057, 'vocab_size': 500, 'tokens': 'tracks/from-scratch/data/processed/tinyshakespeare_bpe_500.pt'}`
- run_id: `flow_check`

## Token Data

- metadata: `{'input': 'data/raw/tinyshakespeare.txt', 'tokenizer_path': 'data/processed/tinyshakespeare_bpe_500.json', 'byte_count': 1115394, 'token_count': 574057, 'vocab_size': 500}`

## Validation

- loss: 5.8404
- perplexity: 343.92

## Generation Settings

- prompt_id: `king_open`
- prompt: `KING:`
- seed: 1337
- samples: 1
- max_new_tokens: 80
- temperature: 1.0
- top_k: 20
- sampling: True

## Generated Samples

### Sample 1

```text
KING:werp ut t peyft ytwperuuIpdy, Ifertp   smmfpeu, uerts t, eut :
dp sdefeud sv:
 sertou, yererdImIvIaewu
```


# Checkpoint Observation

## Inputs

- checkpoint: `tracks/from-scratch/runs/flow-check/checkpoint.pt`
- tokens: `tracks/from-scratch/data/processed/tinyshakespeare_bpe_500.pt`
- output: `tracks/from-scratch/runs/flow-check/observation.md`

## Checkpoint

- step: 19
- metadata: `{'run_id': 'flow_check', 'config_path': 'tracks/from-scratch/runs/flow-check/config.toml', 'manifest_path': 'tracks/from-scratch/data/manifests/tinyshakespeare-bpe-500.toml', 'max_iters': 20, 'batch_size': 8, 'learning_rate': 0.001, 'weight_decay': 0.1, 'warmup_iters': 0, 'lr_decay_iters': 3000, 'min_learning_rate': '', 'seed': 1337, 'device': 'cpu', 'parameter_count': 42804, 'tokenizer': 'bpe', 'tokenizer_path': 'data/processed/tinyshakespeare_bpe_500.json', 'input': 'data/raw/tinyshakespeare.txt', 'byte_count': 1115394, 'token_count': 574057, 'vocab_size': 500, 'tokens': 'tracks/from-scratch/data/processed/tinyshakespeare_bpe_500.pt'}`
- run_id: `flow_check`

## Token Data

- metadata: `{'input': 'data/raw/tinyshakespeare.txt', 'tokenizer_path': 'data/processed/tinyshakespeare_bpe_500.json', 'byte_count': 1115394, 'token_count': 574057, 'vocab_size': 500}`

## Validation

- loss: 5.8404
- perplexity: 343.92

## Generation Settings

- prompt_id: `scene_open`
- prompt: `SCENE`
- seed: 1337
- samples: 1
- max_new_tokens: 80
- temperature: 1.0
- top_k: 20
- sampling: True

## Generated Samples

### Sample 1

```text
SCENEwerp ut t peyft ytwperuuIpdy, Ifertp   smmfpeu, uerts t, eut :
dp sdefeud sv:
 sertou, yererdImIvIaewu
```
