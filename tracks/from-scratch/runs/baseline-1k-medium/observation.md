# Checkpoint Observation

## Inputs

- checkpoint: `tracks/from-scratch/runs/baseline-1k-medium/checkpoint.pt`
- tokens: `tracks/from-scratch/data/processed/tinyshakespeare_bpe_500.pt`
- output: `tracks/from-scratch/runs/baseline-1k-medium/observation.md`

## Checkpoint

- step: 999
- metadata: `{'max_iters': 1000, 'batch_size': 32, 'learning_rate': 0.001, 'parameter_count': 267892, 'tokenizer': 'bpe', 'tokenizer_path': 'tracks/from-scratch/data/processed/tinyshakespeare_bpe_500.json', 'input': 'tracks/from-scratch/data/raw/tinyshakespeare.txt', 'byte_count': 1115394, 'token_count': 574057, 'vocab_size': 500, 'tokens': 'tracks/from-scratch/data/processed/tinyshakespeare_bpe_500.pt'}`

## Token Data

- metadata: `{'input': 'tracks/from-scratch/data/raw/tinyshakespeare.txt', 'tokenizer_path': 'tracks/from-scratch/data/processed/tinyshakespeare_bpe_500.json', 'byte_count': 1115394, 'token_count': 574057, 'vocab_size': 500}`

## Validation

- loss: 3.4161
- perplexity: 30.45

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
KING: but to your man Richard;
But swould not not king breather, and world me you life
Specen'd to tow of, to him likes thee him me?

LUCEN:
But well'st the cover
I am bestious and best to taking homans,
But and thou, when he plow me prece: I crank it we sin this auch.

QUEEN ELIUS:
What wade grant him strament;
Or hep no near the good pacty.

Clown:
Your is another, so so hopes a pareceness to 
```

### Sample 2

```text
KING: Even as that were all your high speak:
Gentle heavester's word.

DUCHESS OF KING RICHARD II:
Theshock, while still at be revoipecty
Bake me, in the wrege! They had whalt to know,
What dishs hoporgned to they he's worse,
Do som, the life on neatn than head him shath.

GLOUCESTER:
There is my fair conse, will news pertory in his lost,
But that I throng neel to him it mysel
```

### Sample 3

```text
KING: when, I know
The take the gentle to hope provizy
The cost:
It is to her offer too lord? I comes him.

ANNELIA:
Here shall not we we to mosceralt, as my doth.

VOLUMNIA:
Well, I't me were a brinch is the temn to him:
Sirly some; couch as much high casss.

ARUTOLYCA:
It me sir, if you am morel, men! be doivelens that knost not talown up,
And sicize from prowns and huch.

CAP
```
