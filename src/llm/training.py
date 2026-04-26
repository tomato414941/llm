from typing import Protocol

import torch


class LanguageModel(Protocol):
    training: bool

    def __call__(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

    def eval(self) -> None: ...

    def train(self) -> None: ...


def get_batch(data: torch.Tensor, block_size: int, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss(
    model: LanguageModel,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    block_size: int,
    batch_size: int,
    eval_iters: int,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    try:
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
        return out
    finally:
        if was_training:
            model.train()
        else:
            model.eval()
