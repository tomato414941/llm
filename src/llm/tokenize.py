import argparse
from pathlib import Path
from time import perf_counter

from llm.tokenizer import BPETokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/from-scratch/processed/bpe_tokenizer.json"))
    parser.add_argument("--vocab-size", type=int, default=1000)
    args = parser.parse_args()

    text = args.input.read_text(encoding="utf-8")
    byte_count = len(text.encode("utf-8"))

    start = perf_counter()
    tokenizer = BPETokenizer.train(text, vocab_size=args.vocab_size)
    train_seconds = perf_counter() - start

    start = perf_counter()
    token_count = len(tokenizer.encode(text))
    encode_seconds = perf_counter() - start

    tokenizer.save(args.output)

    print(f"tokenizer saved to {args.output}")
    print(f"vocab size: {tokenizer.vocab_size}")
    print(f"bytes: {byte_count}")
    print(f"tokens: {token_count}")
    print(f"ratio: {token_count / byte_count:.3f}")
    print(f"train seconds: {train_seconds:.2f}")
    print(f"encode seconds: {encode_seconds:.2f}")


if __name__ == "__main__":
    main()

