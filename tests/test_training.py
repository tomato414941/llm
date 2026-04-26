import torch

from llm.training import estimate_loss, split_train_val


class ConstantLossModel(torch.nn.Module):
    def forward(self, idx, targets=None):
        return torch.zeros((*idx.shape, 2)), torch.tensor(1.0)


def test_split_train_val_uses_default_ninety_ten_split() -> None:
    data = torch.arange(10)

    train_data, val_data = split_train_val(data)

    assert train_data.tolist() == list(range(9))
    assert val_data.tolist() == [9]


def test_estimate_loss_restores_train_mode() -> None:
    model = ConstantLossModel()

    estimate_loss(
        model=model,
        train_data=torch.arange(20),
        val_data=torch.arange(20),
        block_size=4,
        batch_size=2,
        eval_iters=1,
    )

    assert model.training


def test_estimate_loss_restores_eval_mode() -> None:
    model = ConstantLossModel()
    model.eval()

    estimate_loss(
        model=model,
        train_data=torch.arange(20),
        val_data=torch.arange(20),
        block_size=4,
        batch_size=2,
        eval_iters=1,
    )

    assert not model.training
