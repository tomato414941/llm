import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Observation:
    output_path: Path
    checkpoint_path: Path
    tokens_path: Path
    checkpoint_step: int
    checkpoint_metadata: object
    tokens_metadata: object
    validation_loss: float
    validation_perplexity: float
    run_id: str
    prompt: str
    prompt_id: str
    seed: int
    samples: int
    max_new_tokens: int
    temperature: float
    top_k: int | None
    generated_samples: list[str]


SUMMARY_FIELDS = (
    "output_path",
    "checkpoint_path",
    "tokens_path",
    "checkpoint_step",
    "validation_loss",
    "validation_perplexity",
    "run_id",
    "prompt_id",
    "prompt",
    "seed",
    "samples",
    "max_new_tokens",
    "temperature",
    "top_k",
)


def render_markdown(observation: Observation | list[Observation]) -> str:
    if isinstance(observation, list):
        return "\n\n".join(render_markdown(item) for item in observation)
    sample_sections = "\n\n".join(
        f"### Sample {index}\n\n```text\n{sample}\n```"
        for index, sample in enumerate(observation.generated_samples, start=1)
    )
    return f"""# Checkpoint Observation

## Inputs

- checkpoint: `{observation.checkpoint_path}`
- tokens: `{observation.tokens_path}`
- output: `{observation.output_path}`

## Checkpoint

- step: {observation.checkpoint_step}
- metadata: `{observation.checkpoint_metadata}`
- run_id: `{observation.run_id}`

## Token Data

- metadata: `{observation.tokens_metadata}`

## Validation

- loss: {observation.validation_loss:.4f}
- perplexity: {observation.validation_perplexity:.2f}

## Generation Settings

- prompt_id: `{observation.prompt_id}`
- prompt: `{observation.prompt}`
- seed: {observation.seed}
- samples: {observation.samples}
- max_new_tokens: {observation.max_new_tokens}
- temperature: {observation.temperature}
- top_k: {observation.top_k}

## Generated Samples

{sample_sections}
"""


def summary_row(observation: Observation) -> dict[str, object]:
    return {
        "output_path": str(observation.output_path),
        "checkpoint_path": str(observation.checkpoint_path),
        "tokens_path": str(observation.tokens_path),
        "checkpoint_step": observation.checkpoint_step,
        "validation_loss": observation.validation_loss,
        "validation_perplexity": observation.validation_perplexity,
        "run_id": observation.run_id,
        "prompt_id": observation.prompt_id,
        "prompt": observation.prompt,
        "seed": observation.seed,
        "samples": observation.samples,
        "max_new_tokens": observation.max_new_tokens,
        "temperature": observation.temperature,
        "top_k": "" if observation.top_k is None else observation.top_k,
    }


def write_observation(path: Path, observation: Observation | list[Observation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(observation), encoding="utf-8")


def append_summary_row(path: Path, observation: Observation) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=SUMMARY_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(summary_row(observation))
