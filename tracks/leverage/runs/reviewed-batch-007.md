# Reviewed Instruction Batch 007

Batch 007 tested whether the shared source instruction contract lowered
`needs_edit` by giving the generator the same `output_format` and `constraints`
that the judge uses.

## Seeds

Added 200 seeds:

- `reasoning`: 55
- `instruction_following`: 45
- `coding`: 40
- `tool_use`: 30
- `knowledge_qa`: 20
- `summarization_transformation`: 10

## Generation

Generation used the same fixed teacher model as batch 006 for a clean
comparison:

- generator: `qwen3-6-plus-openrouter` / `qwen/qwen3.6-plus`
- temperature: 0.1
- max tokens: 512
- reasoning effort: `none`
- exclude reasoning: true

Recorded generation cost from provider usage metadata: `$0.016096275`.

## Filter Result

- generated rows: 200
- `needs_judge`: 200
- deterministic rejects: 0

## Judge Result

Judging used the same fixed stable judge as batch 006:

- judge: `gpt-5-4-openrouter` / `openai/gpt-5.4`
- parse_error: 0
- accept: 193
- needs_edit: 7
- reject: 0

Accepted rows by capability:

- `reasoning`: 55
- `instruction_following`: 43
- `coding`: 36
- `tool_use`: 29
- `knowledge_qa`: 20
- `summarization_transformation`: 10

## Comparison To Batch 006

Batch 006:

- accept: 64 / 100 = 64.0%
- needs_edit: 35 / 100 = 35.0%

Batch 007:

- accept: 193 / 200 = 96.5%
- needs_edit: 7 / 200 = 3.5%

The shared instruction contract materially improved judge acceptance.

## Promotion

Only exact-prompt-unique accepted rows were promoted:

- accepted rows: 193
- promoted rows: 79
- skipped exact duplicate user prompts: 114

Promoted rows by capability:

- `instruction_following`: 43
- `reasoning`: 11
- `coding`: 8
- `tool_use`: 6
- `knowledge_qa`: 6
- `summarization_transformation`: 5

Reviewed dataset size after promotion: 284 rows.

The high duplicate skip count is a seed-design problem, not a judge problem.
The duplicate gate correctly prevented repeated prompts from entering the
reviewed dataset.

## Follow-up

The next batch should prioritize unique prompt construction over more prompt
contract changes. A simple exact-duplicate prompt report before API generation
would have caught the wasted rows earlier.

## Artifacts

- `tracks/leverage/runs/instruction-outputs/readiness-batch-007-raw.jsonl`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-007-filter.csv`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-007-filter-summary.csv`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-007-candidates.jsonl`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-007-judgments.jsonl`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-007-judgments.csv`
- `tracks/leverage/runs/instruction-outputs/readiness-batch-007-judgments-summary.csv`
