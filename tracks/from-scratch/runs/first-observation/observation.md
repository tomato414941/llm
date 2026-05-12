# Checkpoint Observation

## Inputs

- checkpoint: `tracks/from-scratch/runs/first-observation/checkpoint.pt`
- tokens: `tracks/from-scratch/data/processed/tinyshakespeare_bpe_500.pt`
- output: `tracks/from-scratch/runs/first-observation/observation.md`

## Checkpoint

- step: 99
- metadata: `{'max_iters': 100, 'batch_size': 32, 'learning_rate': 0.001, 'parameter_count': 58804, 'tokenizer': 'bpe', 'tokenizer_path': 'tracks/from-scratch/data/processed/tinyshakespeare_bpe_500.json', 'input': 'tracks/from-scratch/data/raw/tinyshakespeare.txt', 'byte_count': 1115394, 'token_count': 574057, 'vocab_size': 500, 'tokens': 'tracks/from-scratch/data/processed/tinyshakespeare_bpe_500.pt'}`

## Token Data

- metadata: `{'input': 'tracks/from-scratch/data/raw/tinyshakespeare.txt', 'tokenizer_path': 'tracks/from-scratch/data/processed/tinyshakespeare_bpe_500.json', 'byte_count': 1115394, 'token_count': 574057, 'vocab_size': 500}`

## Validation

- loss: 5.0762
- perplexity: 160.17

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
KING:werpanid to pee ft dt.

Ewik
Cde berfote k ty mipeahonerts t, to and cts bide fes,
Ss ILto the cerer
Ie Im aewor
ILetse filorcanweroiot ce pt me p.

M:
DLab, to  gen, s,
Safy there b the tipady w sde gr le Itoy pecs peptevetdtabte I:
Cepdene pumte dsoe
O:
Abakeseekess e
```

### Sample 2

```text
KING:e
Iadow se aahing beenive harsen bty the d c.

the ac, cy fides at te s at ts,
Mdany ss the sotts g, pe p.

s
Ser bos wocet e wee  the e to kut and ds thes hts sendddeny hewedoen todt,
sumts  ctt,
Ialoc
Hert mps s.

UUIs the c
Ah cektut
mpet, mt Mabtent hrid and and gters t
Lg
```

### Sample 3

```text
KING:ty pert
B
PI aape atit the s sosy p,
As
OSod dan
to te I tteee ecabiset  ts  M Id
Wers dy lorce sener, to gy d a dotdest p
C
sdest ed and w
Ed
the t,
Imdd d ILCsers ces w:
H, hiepe meriwy ceduaths the ss bens bue cil:
Ht,
LOdisonss kiet ce  b thy sic ss fy gras ktubs perer
```
