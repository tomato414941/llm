from argparse import Namespace

import torch
import pytest

from llm.evaluate import (
    estimate_validation_loss,
    metadata_line,
    prepared_tokens,
    prepared_vocab_size,
    validate_args,
    validate_vocab_size,
)
from llm.tokenizer import CharTokenizer


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


def test_estimate_validation_loss_restores_train_mode() -> None:
    class ConstantLossModel(torch.nn.Module):
        def forward(self, idx, targets=None):
            return torch.zeros((*idx.shape, 2)), torch.tensor(2.0)

    model = ConstantLossModel()

    estimate_validation_loss(
        model=model,
        val_data=torch.arange(20),
        block_size=4,
        batch_size=2,
        eval_iters=1,
    )

    assert model.training


def test_estimate_validation_loss_restores_eval_mode() -> None:
    class ConstantLossModel(torch.nn.Module):
        def forward(self, idx, targets=None):
            return torch.zeros((*idx.shape, 2)), torch.tensor(2.0)

    model = ConstantLossModel()
    model.eval()

    estimate_validation_loss(
        model=model,
        val_data=torch.arange(20),
        block_size=4,
        batch_size=2,
        eval_iters=1,
    )

    assert not model.training


def test_validate_vocab_size_accepts_matching_prepared_tokens() -> None:
    tokenizer = CharTokenizer.from_text("abc")
    prepared = {
        "tokens": torch.tensor([0, 1, 2]),
        "tokenizer": tokenizer.to_payload(),
    }

    validate_vocab_size(3, prepared)


def test_validate_vocab_size_rejects_mismatched_prepared_tokens() -> None:
    tokenizer = CharTokenizer.from_text("abc")
    prepared = {
        "tokens": torch.tensor([0, 1, 2]),
        "tokenizer": tokenizer.to_payload(),
    }

    with pytest.raises(ValueError, match="vocab size"):
        validate_vocab_size(4, prepared)


def test_prepared_tokens_rejects_non_tensor_payload() -> None:
    with pytest.raises(ValueError, match="tensor"):
        prepared_tokens({"tokens": [1, 2, 3]})


def test_prepared_tokens_rejects_missing_tokens_payload() -> None:
    with pytest.raises(ValueError, match="tokens"):
        prepared_tokens({})


def test_prepared_vocab_size_rejects_missing_tokenizer_payload() -> None:
    with pytest.raises(ValueError, match="tokenizer"):
        prepared_vocab_size({})


def test_metadata_line_returns_none_without_metadata() -> None:
    assert metadata_line("tokens", {}) is None


def test_validate_args_rejects_non_positive_values() -> None:
    with pytest.raises(ValueError, match="--batch-size"):
        validate_args(Namespace(batch_size=0, eval_iters=1))

    with pytest.raises(ValueError, match="--eval-iters"):
        validate_args(Namespace(batch_size=1, eval_iters=0))
