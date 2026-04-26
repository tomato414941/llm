import argparse
from pathlib import Path

import torch

from llm.models import TransformerConfig, TransformerLanguageModel
from llm.tokenizer import CharTokenizer
from llm.training import estimate_loss, get_batch

DEFAULT_TEXT = """
In the beginning there was a small language model.
It learned with attention, residual paths, normalization, and feed-forward layers.
Each step asked one quiet question: what comes next?
"""


def load_text(path: Path | None) -> str:
    if path is None:
        return DEFAULT_TEXT
    return path.read_text(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--max-iters", type=int, default=3000)
    parser.add_argument("--eval-interval", type=int, default=300)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--generate-tokens", type=int, default=200)
    args = parser.parse_args()

    torch.manual_seed(1337)

    text = load_text(args.input)
    tokenizer = CharTokenizer.from_text(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split_index = int(0.9 * len(data))
    train_data = data[:split_index]
    val_data = data[split_index:]

    if len(train_data) <= args.block_size or len(val_data) <= args.block_size:
        raise ValueError("input text is too small for the requested block size")

    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
    )
    model = TransformerLanguageModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    for step in range(args.max_iters):
        if step % args.eval_interval == 0 or step == args.max_iters - 1:
            losses = estimate_loss(
                model=model,
                train_data=train_data,
                val_data=val_data,
                block_size=args.block_size,
                batch_size=args.batch_size,
                eval_iters=args.eval_iters,
            )
            print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch(train_data, args.block_size, args.batch_size)
        _, loss = model(xb, yb)
        if loss is None:
            raise RuntimeError("loss was not computed")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    context = torch.zeros((1, 1), dtype=torch.long)
    generated = model.generate(context, max_new_tokens=args.generate_tokens)[0].tolist()
    print("\n--- sample ---")
    print(tokenizer.decode(generated))


if __name__ == "__main__":
    main()
