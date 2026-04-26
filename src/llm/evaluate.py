import argparse
import math
from pathlib import Path

import torch

from llm.generate import load_checkpoint
from llm.training import get_batch


def perplexity(loss: float) -> float:
    return math.exp(loss)


def load_tokens(path: Path) -> torch.Tensor:
    prepared = torch.load(path, map_location="cpu", weights_only=True)
    return prepared["tokens"].to(dtype=torch.long)


def split_train_val(data: torch.Tensor, train_ratio: float = 0.9) -> tuple[torch.Tensor, torch.Tensor]:
    split_index = int(train_ratio * len(data))
    return data[:split_index], data[split_index:]


@torch.no_grad()
def estimate_validation_loss(
    model: torch.nn.Module,
    val_data: torch.Tensor,
    block_size: int,
    batch_size: int,
    eval_iters: int,
) -> float:
    model.eval()
    losses = torch.zeros(eval_iters)
    for index in range(eval_iters):
        xb, yb = get_batch(val_data, block_size, batch_size)
        _, loss = model(xb, yb)
        if loss is None:
            raise RuntimeError("loss was not computed")
        losses[index] = loss.item()
    return losses.mean().item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-iters", type=int, default=20)
    args = parser.parse_args()

    model, _tokenizer, checkpoint = load_checkpoint(args.checkpoint)
    data = load_tokens(args.tokens)
    _train_data, val_data = split_train_val(data)
    block_size = model.config.block_size

    if len(val_data) <= block_size:
        raise ValueError("validation data is too small for the checkpoint block size")

    val_loss = estimate_validation_loss(
        model=model,
        val_data=val_data,
        block_size=block_size,
        batch_size=args.batch_size,
        eval_iters=args.eval_iters,
    )

    print(f"checkpoint step: {checkpoint['step']}")
    print(f"validation loss: {val_loss:.4f}")
    print(f"validation perplexity: {perplexity(val_loss):.2f}")


if __name__ == "__main__":
    main()
