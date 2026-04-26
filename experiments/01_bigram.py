import argparse
from pathlib import Path

import torch

from llm.models import BigramLanguageModel
from llm.tokenizer import CharTokenizer

DEFAULT_TEXT = """
In the beginning there was a small language model.
It knew only pairs of characters, but it learned by prediction.
Each step asked one quiet question: what comes next?
"""


def get_batch(data: torch.Tensor, block_size: int, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss(
    model: BigramLanguageModel,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    block_size: int,
    batch_size: int,
    eval_iters: int,
) -> dict[str, float]:
    model.eval()
    out = {}
    for split, data in (("train", train_data), ("val", val_data)):
        losses = torch.zeros(eval_iters)
        for index in range(eval_iters):
            xb, yb = get_batch(data, block_size, batch_size)
            _, loss = model(xb, yb)
            if loss is None:
                raise RuntimeError("loss was not computed")
            losses[index] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def load_text(path: Path | None) -> str:
    if path is None:
        return DEFAULT_TEXT
    return path.read_text(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--max-iters", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
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

    model = BigramLanguageModel(tokenizer.vocab_size)
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
