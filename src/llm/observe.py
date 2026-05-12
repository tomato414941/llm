import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import torch

from llm.checkpoint import load_checkpoint
from llm.config import compact_defaults, load_toml, observe_config_defaults
from llm.device import resolve_device
from llm.evaluate import (
    estimate_validation_loss,
    load_prepared_tokens,
    perplexity,
    prepared_tokens,
    validate_vocab_size,
)
from llm.generate import generate_samples
from llm.observation import Observation, append_summary_row, write_observation
from llm.training import split_train_val


@dataclass(frozen=True)
class ObservationContext:
    model: torch.nn.Module
    tokenizer: object
    checkpoint: dict[str, object]
    prepared: dict[str, object]
    validation_loss: float
    validation_perplexity: float


def load_prompt_file(path: Path) -> list[dict[str, str]]:
    prompts: list[dict[str, str]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"prompt file line {line_number} must be a JSON object")
        prompt_id = payload.get("id")
        prompt = payload.get("prompt")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise ValueError(f"prompt file line {line_number} must contain a non-empty id")
        if not isinstance(prompt, str):
            raise ValueError(f"prompt file line {line_number} must contain a prompt")
        prompts.append({"id": prompt_id, "prompt": prompt})
    if not prompts:
        raise ValueError("prompt file must contain at least one prompt")
    return prompts


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.eval_iters <= 0:
        raise ValueError("--eval-iters must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if args.top_k is not None and args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    if args.prompt and args.prompt_file is not None:
        raise ValueError("exactly one of --prompt or --prompt-file may be used")


def checkpoint_run_id(checkpoint: dict[str, object]) -> str:
    metadata = checkpoint.get("metadata", {})
    if isinstance(metadata, dict):
        run_id = metadata.get("run_id", "")
        if isinstance(run_id, str):
            return run_id
    return ""


def load_observation_context(args: argparse.Namespace) -> ObservationContext:
    device = resolve_device(args.device)
    print(f"device: {device}")
    model, tokenizer, checkpoint = load_checkpoint(args.checkpoint)
    model = model.to(device)
    prepared = load_prepared_tokens(args.tokens)
    validate_vocab_size(model.config.vocab_size, prepared)
    data = prepared_tokens(prepared).to(device)
    _train_data, val_data = split_train_val(data)
    block_size = model.config.block_size
    if len(val_data) <= block_size:
        raise ValueError("validation data is too small to observe this checkpoint block size")

    validation_loss = estimate_validation_loss(
        model=model,
        val_data=val_data,
        block_size=block_size,
        batch_size=args.batch_size,
        eval_iters=args.eval_iters,
    )

    return ObservationContext(
        model=model,
        tokenizer=tokenizer,
        checkpoint=checkpoint,
        prepared=prepared,
        validation_loss=validation_loss,
        validation_perplexity=perplexity(validation_loss),
    )


def build_observation(
    context: ObservationContext,
    args: argparse.Namespace,
    prompt: str | None = None,
    prompt_id: str = "",
) -> Observation:
    torch.manual_seed(args.seed)
    generation_prompt = args.prompt if prompt is None else prompt
    generated_samples = generate_samples(
        model=context.model,
        tokenizer=context.tokenizer,
        prompt=generation_prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        sampling=args.sampling,
        samples=args.samples,
    )

    return Observation(
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        tokens_path=args.tokens,
        checkpoint_step=int(context.checkpoint["step"]),
        checkpoint_metadata=context.checkpoint.get("metadata", {}),
        tokens_metadata=context.prepared.get("metadata", {}),
        validation_loss=context.validation_loss,
        validation_perplexity=context.validation_perplexity,
        run_id=checkpoint_run_id(context.checkpoint),
        prompt=generation_prompt,
        prompt_id=prompt_id,
        seed=args.seed,
        samples=args.samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        sampling=args.sampling,
        generated_samples=generated_samples,
    )


def build_observations(args: argparse.Namespace) -> list[Observation]:
    context = load_observation_context(args)
    if args.prompt_file is None:
        return [build_observation(context, args)]
    return [
        build_observation(context, args, prompt=prompt["prompt"], prompt_id=prompt["id"])
        for prompt in load_prompt_file(args.prompt_file)
    ]


def build_parser(defaults: dict[str, object]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.set_defaults(**defaults)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--tokens", type=Path)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--eval-iters", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--sampling", action=argparse.BooleanOptionalAction)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    return parser


def parse_args() -> argparse.Namespace:
    base_defaults = {
        "eval_iters": 20,
        "batch_size": 32,
        "max_new_tokens": 300,
        "temperature": 1.0,
        "sampling": True,
        "seed": 1337,
        "samples": 1,
        "device": "auto",
    }
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    config_args, _unknown = config_parser.parse_known_args()
    defaults = dict(base_defaults)
    if config_args.config is not None:
        loaded_config = load_toml(config_args.config)
        defaults.update(compact_defaults(observe_config_defaults(loaded_config)))
        defaults["config"] = config_args.config
    parser = build_parser(defaults)
    args = parser.parse_args()
    if isinstance(args.checkpoint, str):
        args.checkpoint = Path(args.checkpoint)
    if isinstance(args.tokens, str):
        args.tokens = Path(args.tokens)
    if isinstance(args.prompt_file, str):
        args.prompt_file = Path(args.prompt_file)
    if isinstance(args.output, str):
        args.output = Path(args.output)
    if isinstance(args.summary_output, str):
        args.summary_output = Path(args.summary_output)
    return args


def main() -> None:
    args = parse_args()
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required")
    if args.tokens is None:
        raise ValueError("--tokens is required")
    if args.output is None:
        raise ValueError("--output is required")
    validate_args(args)

    observations = build_observations(args)

    write_observation(args.output, observations[0] if len(observations) == 1 else observations)
    if args.summary_output is not None:
        for observation in observations:
            append_summary_row(args.summary_output, observation)

    print(f"observation written to {args.output}")
    if args.summary_output is not None:
        print(f"summary appended to {args.summary_output}")


if __name__ == "__main__":
    main()
