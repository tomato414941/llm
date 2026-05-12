# Checkpoint Observation

## Inputs

- checkpoint: `tracks/from-scratch/runs/pico-gpt2-tinyshakespeare-short/checkpoint.pt`
- tokens: `tracks/from-scratch/data/processed/tinyshakespeare_bpe_500.pt`
- output: `tracks/from-scratch/runs/pico-gpt2-tinyshakespeare-short/observation.md`

## Checkpoint

- step: 19
- metadata: `{'run_id': 'pico_gpt2_tinyshakespeare', 'config_path': 'tracks/from-scratch/runs/pico-gpt2-tinyshakespeare/config.toml', 'manifest_path': 'tracks/from-scratch/data/manifests/tinyshakespeare-bpe-500.toml', 'max_iters': 20, 'batch_size': 8, 'learning_rate': 0.001, 'warmup_iters': 100, 'lr_decay_iters': 3000, 'min_learning_rate': 0.0001, 'seed': 1337, 'parameter_count': 235892, 'tokenizer': 'bpe', 'tokenizer_path': 'tracks/from-scratch/data/processed/tinyshakespeare_bpe_500.json', 'input': 'tracks/from-scratch/data/raw/tinyshakespeare.txt', 'byte_count': 1115394, 'token_count': 574057, 'vocab_size': 500, 'tokens': 'tracks/from-scratch/data/processed/tinyshakespeare_bpe_500.pt'}`
- run_id: `pico_gpt2_tinyshakespeare`

## Token Data

- metadata: `{'input': 'tracks/from-scratch/data/raw/tinyshakespeare.txt', 'tokenizer_path': 'tracks/from-scratch/data/processed/tinyshakespeare_bpe_500.json', 'byte_count': 1115394, 'token_count': 574057, 'vocab_size': 500}`

## Validation

- loss: 6.0899
- perplexity: 441.36

## Generation Settings

- prompt_id: `king_open`
- prompt: `KING:`
- seed: 1337
- samples: 1
- max_new_tokens: 80
- temperature: 1.0
- top_k: 20

## Generated Samples

### Sample 1

```text
KING:ds sehis:
oofif this dtseonem m�oosedseonIfotoo this outon this �de hhoue is t, HI-hs liide ffsdoY-Iseto , Ho-HIe II-e
i, e

```


# Checkpoint Observation

## Inputs

- checkpoint: `tracks/from-scratch/runs/pico-gpt2-tinyshakespeare-short/checkpoint.pt`
- tokens: `tracks/from-scratch/data/processed/tinyshakespeare_bpe_500.pt`
- output: `tracks/from-scratch/runs/pico-gpt2-tinyshakespeare-short/observation.md`

## Checkpoint

- step: 19
- metadata: `{'run_id': 'pico_gpt2_tinyshakespeare', 'config_path': 'tracks/from-scratch/runs/pico-gpt2-tinyshakespeare/config.toml', 'manifest_path': 'tracks/from-scratch/data/manifests/tinyshakespeare-bpe-500.toml', 'max_iters': 20, 'batch_size': 8, 'learning_rate': 0.001, 'warmup_iters': 100, 'lr_decay_iters': 3000, 'min_learning_rate': 0.0001, 'seed': 1337, 'parameter_count': 235892, 'tokenizer': 'bpe', 'tokenizer_path': 'tracks/from-scratch/data/processed/tinyshakespeare_bpe_500.json', 'input': 'tracks/from-scratch/data/raw/tinyshakespeare.txt', 'byte_count': 1115394, 'token_count': 574057, 'vocab_size': 500, 'tokens': 'tracks/from-scratch/data/processed/tinyshakespeare_bpe_500.pt'}`
- run_id: `pico_gpt2_tinyshakespeare`

## Token Data

- metadata: `{'input': 'tracks/from-scratch/data/raw/tinyshakespeare.txt', 'tokenizer_path': 'tracks/from-scratch/data/processed/tinyshakespeare_bpe_500.json', 'byte_count': 1115394, 'token_count': 574057, 'vocab_size': 500}`

## Validation

- loss: 6.0899
- perplexity: 441.36

## Generation Settings

- prompt_id: `scene_open`
- prompt: `SCENE`
- seed: 1337
- samples: 1
- max_new_tokens: 80
- temperature: 1.0
- top_k: 20

## Generated Samples

### Sample 1

```text
SCENEws sehis:
oofif this dtseonmy IHstsede onIfotoo this e ton this iTe hhou:is t, HI-lis liide ffsdos Iseto , Ho-HIe II-e
i, e

```
