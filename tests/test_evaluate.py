import torch

from llm.evaluate import estimate_validation_loss, split_train_val


def test_split_train_val_uses_default_ninety_ten_split() -> None:
    data = torch.arange(10)

    train_data, val_data = split_train_val(data)

    assert train_data.tolist() == list(range(9))
    assert val_data.tolist() == [9]


def test_estimate_validation_loss_returns_mean_loss() -> None:
    class ConstantLossModel(torch.nn.Module):
        def forward(self, idx, targets=None):
            return torch.zeros((*idx.shape, 2)), torch.tensor(2.0)

    loss = estimate_validation_loss(
        model=ConstantLossModel(),
        val_data=torch.arange(20),
        block_size=4,
        batch_size=2,
        eval_iters=3,
    )

    assert loss == 2.0
