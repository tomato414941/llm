# Reviewed Instruction Review Rules

Use these rules before promoting generated answers into a committed reviewed
instruction file.

Raw generated answers remain experiment artifacts. The expected raw location is
`experiments/leverage/instruction_outputs/`; the reviewed location is
`datasets/reviewed_instructions/`.

## Required Checks

- The row is not copied from a held-out eval prompt.
- The assistant answer is correct for the prompt.
- The answer follows the requested format and constraints.
- The answer contains no API keys, tokens, private paths, or local environment
  details.
- The answer is useful for the intended student model behavior.
- The answer is not just a generic explanation that ignores the project context.
- The row has clear provenance in `source_prompt_id` and `review`.

## Review Status

Use `accepted_instruction` only when the row passed review and can be considered
for a small SFT/LoRA experiment.

Do not use this status for raw generated outputs. Raw outputs should remain
under `experiments/leverage/instruction_outputs/` until reviewed.

## Promotion Boundary

An accepted reviewed instruction is still not proof of value. It becomes useful only after:

- the training split is defined
- a held-out eval is selected
- a baseline model result exists
- a small training run shows measurable improvement
