import argparse
from pathlib import Path
from typing import Any

import torch

from llm.models import TransformerConfig, TransformerLanguageModel
from llm.tokenizer import CharTokenizer


def load_checkpoint(path: Path) -> tuple[TransformerLanguageModel, CharTokenizer, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    chars = tuple(checkpoint["tokenizer_chars"])
    tokenizer = CharTokenizer(
        chars=chars,
        stoi={char: index for index, char in enumerate(chars)},
        itos={index: char for index, char in enumerate(chars)},
    )
    config = TransformerConfig.from_dict(checkpoint["config"])
    model = TransformerLanguageModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, tokenizer, checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/mini_gpt.pt"))
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int)
    args = parser.parse_args()

    model, tokenizer, checkpoint = load_checkpoint(args.checkpoint)
    if args.prompt:
        context_ids = tokenizer.encode(args.prompt)
    else:
        context_ids = [0]
    context = torch.tensor([context_ids], dtype=torch.long)

    generated = model.generate(
        context,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )[0].tolist()

    print(f"checkpoint step: {checkpoint['step']}")
    print(f"losses: {checkpoint['losses']}")
    print("\n--- sample ---")
    print(tokenizer.decode(generated))


if __name__ == "__main__":
    main()
