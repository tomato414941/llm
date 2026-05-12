import torch
import pytest

from llm.models import (
    MultiHeadAttention,
    SelfAttentionHead,
    TransformerConfig,
    TransformerLanguageModel,
)
from llm.models.transformer import FeedForward


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


def test_self_attention_head_does_not_attend_to_future_tokens() -> None:
    head = SelfAttentionHead(embedding_dim=4, head_size=2, block_size=4, dropout=0.0)
    x = torch.randn(1, 4, 4)
    changed_future = x.clone()
    changed_future[:, 2:, :] = torch.randn(1, 2, 4)

    original = head(x)
    changed = head(changed_future)

    assert torch.allclose(original[:, :2, :], changed[:, :2, :])


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


def test_feed_forward_uses_gelu_activation() -> None:
    feed_forward = FeedForward(embedding_dim=4)

    assert isinstance(feed_forward.activation, torch.nn.GELU)


def test_feed_forward_returns_sequence_features() -> None:
    feed_forward = FeedForward(embedding_dim=4)
    x = torch.randn(2, 3, 4)

    out = feed_forward(x)

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


def test_transformer_language_model_ties_token_embedding_and_lm_head_weights() -> None:
    config = TransformerConfig(
        vocab_size=5,
        block_size=3,
        embedding_dim=4,
        num_heads=2,
        num_layers=2,
    )
    model = TransformerLanguageModel(config)

    assert model.lm_head.weight is model.token_embedding_table.weight


def test_transformer_language_model_scales_residual_projection_init() -> None:
    config = TransformerConfig(
        vocab_size=64,
        block_size=8,
        embedding_dim=64,
        num_heads=4,
        num_layers=4,
        dropout=0.0,
    )
    torch.manual_seed(1337)
    model = TransformerLanguageModel(config)
    block = model.blocks[0]
    expected_std = model._residual_projection_std()

    assert abs(block.attention.projection.weight.std().item() - expected_std) < 0.002
    assert abs(block.feed_forward.output_projection.weight.std().item() - expected_std) < 0.002
    assert abs(block.feed_forward.input_projection.weight.std().item() - 0.02) < 0.002


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


def test_transformer_language_model_can_generate_greedily() -> None:
    config = TransformerConfig(
        vocab_size=5,
        block_size=3,
        embedding_dim=4,
        num_heads=2,
        num_layers=2,
        dropout=0.0,
    )
    model = TransformerLanguageModel(config)

    torch.manual_seed(1)
    generated = model.generate(torch.tensor([[0]]), max_new_tokens=2, do_sample=False)
    torch.manual_seed(2)
    regenerated = model.generate(torch.tensor([[0]]), max_new_tokens=2, do_sample=False)

    assert generated.shape == (1, 3)
    assert torch.equal(generated, regenerated)


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
