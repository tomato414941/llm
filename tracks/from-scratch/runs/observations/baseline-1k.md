# Checkpoint Observation

## Inputs

- checkpoint: `tracks/from-scratch/checkpoints/baseline-1k.pt`
- tokens: `tracks/from-scratch/data/processed/tinyshakespeare_bpe_500.pt`
- output: `tracks/from-scratch/runs/observations/baseline-1k.md`

## Checkpoint

- step: 999
- metadata: `{'max_iters': 1000, 'batch_size': 32, 'learning_rate': 0.001, 'parameter_count': 58804, 'tokenizer': 'bpe', 'tokenizer_path': 'tracks/from-scratch/data/processed/tinyshakespeare_bpe_500.json', 'input': 'tracks/from-scratch/data/raw/tinyshakespeare.txt', 'byte_count': 1115394, 'token_count': 574057, 'vocab_size': 500, 'tokens': 'tracks/from-scratch/data/processed/tinyshakespeare_bpe_500.pt'}`

## Token Data

- metadata: `{'input': 'tracks/from-scratch/data/raw/tinyshakespeare.txt', 'tokenizer_path': 'tracks/from-scratch/data/processed/tinyshakespeare_bpe_500.json', 'byte_count': 1115394, 'token_count': 574057, 'vocab_size': 500}`

## Validation

- loss: 3.7631
- perplexity: 43.08

## Generation Settings

- prompt: `KING:`
- seed: 1337
- samples: 3
- max_new_tokens: 200
- temperature: 1.0
- top_k: 20

## Generated Samples

### Sample 1

```text
KING: bereforth.

VRUTIO:
No twould not what wearts berfold me you lif
Sprishoner'd antaint, and croubsion.

LAUTIO:
You scred your lation
Ifff not in tear
If I I king hou very will my so-y;
Well ho', and still the head meed to plas,
Tafffer, and brove, in thought of Citien Romant,
Eell can heptry, gard; 't not sta gue:
I but me than he in sus with look I put of thees
W
```

### Sample 2

```text
KING: Con amost not pactend blord:
Go harren your swords mue act
Afister fort thee, and say you all:
On, whil, so?

GLOUCEY:
Y no; 'll bers boy's dishes, give with lool ambly.

BOR:
YORW Is so, dexd to the ewell prown,
Comes the cry's murse and furn.

Noy bouth.
Gakewer'll not thant
Atwerell truuch speen, for a love
Enger sid and fran halln ha
```

### Sample 3

```text
KING: wort with their anis,
I sprongus, host the provizy
Well bother's day to trant, that tovert a know
As word, for arrie; and mers deal list
To were to put up't: you looks peath,
This narevess in man arter trie:
Sir our by sirs to me in the prongone
Thou lificks in more abite sin
As bue you leart! by loot.

CORIUS:
Thrt to tid offrout cens,
Better wrink forth of, I s
```
