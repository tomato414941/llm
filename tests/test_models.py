import torch
import pytest

from llm.models import (
    MultiHeadAttention,
    SelfAttentionHead,
    TransformerConfig,
    TransformerLanguageModel,
)


def test_self_attention_head_returns_sequence_features() -> None:
    head = SelfAttentionHead(embedding_dim=4, head_size=2, block_size=3)
    x = torch.randn(2, 3, 4)

    out = head(x)

    assert out.shape == (2, 3, 2)


def test_self_attention_head_scales_by_head_dimension() -> None:
    head = SelfAttentionHead(embedding_dim=4, head_size=2, block_size=3)
    x = torch.ones(1, 3, 4)

    with torch.no_grad():
        k = head.key(x)
        q = head.query(x)
        expected = q @ k.transpose(-2, -1) * q.shape[-1] ** -0.5
        expected = expected.masked_fill(head.tril[:3, :3] == 0, float("-inf"))
        expected = torch.softmax(expected, dim=-1)
        expected = expected @ head.value(x)

    assert torch.allclose(head(x), expected)


def test_multi_head_attention_returns_projected_features() -> None:
    attention = MultiHeadAttention(
        embedding_dim=4,
        num_heads=2,
        head_size=2,
        block_size=3,
    )
    x = torch.randn(2, 3, 4)

    out = attention(x)

    assert out.shape == (2, 3, 4)


def test_transformer_language_model_returns_sequence_logits_and_loss() -> None:
    config = TransformerConfig(
        vocab_size=5,
        block_size=3,
        embedding_dim=4,
        num_heads=2,
        num_layers=2,
        dropout=0.1,
    )
    model = TransformerLanguageModel(config)
    idx = torch.tensor([[0, 1, 2], [2, 3, 4]])
    targets = torch.tensor([[1, 2, 3], [3, 4, 0]])

    logits, loss = model(idx, targets)

    assert logits.shape == (2, 3, 5)
    assert loss is not None


def test_transformer_language_model_rejects_too_long_context() -> None:
    config = TransformerConfig(
        vocab_size=5,
        block_size=3,
        embedding_dim=4,
        num_heads=2,
        num_layers=2,
        dropout=0.1,
    )
    model = TransformerLanguageModel(config)

    with pytest.raises(ValueError, match="expected time dimension"):
        model(torch.tensor([[0, 1, 2, 3]]))


def test_transformer_language_model_rejects_non_positive_temperature() -> None:
    config = TransformerConfig(
        vocab_size=5,
        block_size=3,
        embedding_dim=4,
        num_heads=2,
        num_layers=2,
        dropout=0.1,
    )
    model = TransformerLanguageModel(config)

    with pytest.raises(ValueError, match="temperature"):
        model.generate(torch.tensor([[0]]), max_new_tokens=1, temperature=0)


def test_transformer_language_model_rejects_non_positive_top_k() -> None:
    config = TransformerConfig(
        vocab_size=5,
        block_size=3,
        embedding_dim=4,
        num_heads=2,
        num_layers=2,
        dropout=0.1,
    )
    model = TransformerLanguageModel(config)

    with pytest.raises(ValueError, match="top_k"):
        model.generate(torch.tensor([[0]]), max_new_tokens=1, top_k=0)


def test_transformer_config_round_trips_dict() -> None:
    config = TransformerConfig(
        vocab_size=5,
        block_size=3,
        embedding_dim=4,
        num_heads=2,
        num_layers=2,
        dropout=0.1,
    )

    assert TransformerConfig.from_dict(config.to_dict()) == config
