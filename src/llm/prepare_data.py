import argparse
from pathlib import Path
from time import perf_counter

import torch

from llm.tokenizer import BPETokenizer


def load_non_empty_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if not text:
        raise ValueError("input text must not be empty")
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("tracks/from-scratch/data/processed/tokens.pt"))
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    text = load_non_empty_text(args.input)
    byte_count = len(text.encode("utf-8"))
    tokenizer = BPETokenizer.load(args.tokenizer)

    start = perf_counter()
    token_ids = tokenizer.encode(text)
    encode_seconds = perf_counter() - start

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "tokens": torch.tensor(token_ids, dtype=torch.long),
            "tokenizer": tokenizer.to_payload(),
            "metadata": {
                "input": str(args.input),
                "tokenizer_path": str(args.tokenizer),
                "manifest_path": str(args.manifest) if args.manifest is not None else "",
                "byte_count": byte_count,
                "token_count": len(token_ids),
                "vocab_size": tokenizer.vocab_size,
            },
        },
        args.output,
    )

    print(f"tokens saved to {args.output}")
    print(f"vocab size: {tokenizer.vocab_size}")
    print(f"bytes: {byte_count}")
    print(f"tokens: {len(token_ids)}")
    print(f"ratio: {len(token_ids) / byte_count:.3f}")
    print(f"encode seconds: {encode_seconds:.2f}")


if __name__ == "__main__":
    main()
