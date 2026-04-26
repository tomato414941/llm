import torch
import pytest

from llm.models import BigramLanguageModel, MLPLanguageModel, SingleHeadAttentionLanguageModel


def test_bigram_language_model_returns_sequence_logits_and_loss() -> None:
    model = BigramLanguageModel(vocab_size=5)
    idx = torch.tensor([[0, 1, 2], [2, 3, 4]])
    targets = torch.tensor([[1, 2, 3], [3, 4, 0]])

    logits, loss = model(idx, targets)

    assert logits.shape == (6, 5)
    assert loss is not None


def test_mlp_language_model_returns_next_token_logits_and_loss() -> None:
    model = MLPLanguageModel(vocab_size=5, block_size=3, embedding_dim=4, hidden_dim=8)
    idx = torch.tensor([[0, 1, 2], [2, 3, 4]])
    targets = torch.tensor([[1, 2, 3], [3, 4, 0]])

    logits, loss = model(idx, targets)

    assert logits.shape == (2, 5)
    assert loss is not None


def test_mlp_language_model_rejects_wrong_block_size() -> None:
    model = MLPLanguageModel(vocab_size=5, block_size=3, embedding_dim=4, hidden_dim=8)

    with pytest.raises(ValueError, match="expected time dimension"):
        model(torch.tensor([[0, 1]]))


def test_single_head_attention_language_model_returns_sequence_logits_and_loss() -> None:
    model = SingleHeadAttentionLanguageModel(vocab_size=5, block_size=3, embedding_dim=4)
    idx = torch.tensor([[0, 1, 2], [2, 3, 4]])
    targets = torch.tensor([[1, 2, 3], [3, 4, 0]])

    logits, loss = model(idx, targets)

    assert logits.shape == (2, 3, 5)
    assert loss is not None


def test_single_head_attention_language_model_rejects_too_long_context() -> None:
    model = SingleHeadAttentionLanguageModel(vocab_size=5, block_size=3, embedding_dim=4)

    with pytest.raises(ValueError, match="expected time dimension"):
        model(torch.tensor([[0, 1, 2, 3]]))
