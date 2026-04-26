import argparse
from pathlib import Path

import torch

from llm.checkpoint import load_checkpoint
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


def build_observation(args: argparse.Namespace) -> Observation:
    model, tokenizer, checkpoint = load_checkpoint(args.checkpoint)
    prepared = load_prepared_tokens(args.tokens)
    validate_vocab_size(model.config.vocab_size, prepared)
    data = prepared_tokens(prepared)
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
    torch.manual_seed(args.seed)
    generated_samples = generate_samples(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        samples=args.samples,
    )

    return Observation(
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        tokens_path=args.tokens,
        checkpoint_step=int(checkpoint["step"]),
        checkpoint_metadata=checkpoint.get("metadata", {}),
        tokens_metadata=prepared.get("metadata", {}),
        validation_loss=validation_loss,
        validation_perplexity=perplexity(validation_loss),
        prompt=args.prompt,
        seed=args.seed,
        samples=args.samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        generated_samples=generated_samples,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--samples", type=int, default=1)
    args = parser.parse_args()
    validate_args(args)

    observation = build_observation(args)
    write_observation(args.output, observation)
    if args.summary_output is not None:
        append_summary_row(args.summary_output, observation)

    print(f"observation written to {args.output}")
    if args.summary_output is not None:
        print(f"summary appended to {args.summary_output}")


if __name__ == "__main__":
    main()
