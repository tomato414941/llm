import argparse
from pathlib import Path

import torch

from llm.checkpoint import load_checkpoint
from llm.device import resolve_device
from llm.tokenizer import BPETokenizer, CharTokenizer


def default_context_ids(tokenizer: CharTokenizer | BPETokenizer) -> list[int]:
    encoded = tokenizer.encode("\n")
    return encoded if encoded else [0]


def validate_args(args: argparse.Namespace) -> None:
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if args.top_k is not None and args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.samples <= 0:
        raise ValueError("--samples must be positive")


def prompt_context_ids(tokenizer: CharTokenizer | BPETokenizer, prompt: str) -> list[int]:
    if not prompt:
        return default_context_ids(tokenizer)
    try:
        return tokenizer.encode(prompt)
    except ValueError as error:
        raise ValueError(f"prompt cannot be encoded by checkpoint tokenizer: {error}") from error


def generate_sample(
    model: torch.nn.Module,
    tokenizer: CharTokenizer | BPETokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
) -> str:
    device = next(model.parameters()).device
    context = torch.tensor([prompt_context_ids(tokenizer, prompt)], dtype=torch.long, device=device)
    generated = model.generate(
        context,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )[0].tolist()
    return tokenizer.decode(generated)


def generate_samples(
    model: torch.nn.Module,
    tokenizer: CharTokenizer | BPETokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    samples: int,
) -> list[str]:
    return [
        generate_sample(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        for _ in range(samples)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/mini-gpt.pt"))
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    validate_args(args)
    device = resolve_device(args.device)
    print(f"device: {device}")

    model, tokenizer, checkpoint = load_checkpoint(args.checkpoint)
    model = model.to(device)
    torch.manual_seed(args.seed)

    print(f"checkpoint step: {checkpoint['step']}")
    print(f"losses: {checkpoint['losses']}")
    if "metadata" in checkpoint:
        print(f"metadata: {checkpoint['metadata']}")
    samples = generate_samples(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        samples=args.samples,
    )
    for sample_index, sample in enumerate(samples):
        print(f"\n--- sample {sample_index + 1} ---")
        print(sample)


if __name__ == "__main__":
    main()
