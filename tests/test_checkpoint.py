import pytest
import torch

from llm.checkpoint import load_checkpoint, load_tokenizer
from llm.models import TransformerConfig, TransformerLanguageModel
from llm.tokenizer import CharTokenizer


def checkpoint_payload() -> dict[str, object]:
    config = TransformerConfig(
        vocab_size=3,
        block_size=4,
        embedding_dim=4,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
    )
    model = TransformerLanguageModel(config)
    tokenizer = CharTokenizer.from_text("abc")
    return {
        "model_state_dict": model.state_dict(),
        "tokenizer": tokenizer.to_payload(),
        "config": config.to_dict(),
        "step": 1,
        "losses": {"train": 1.0, "val": 1.1},
    }


def test_load_checkpoint_reads_current_payload(tmp_path) -> None:
    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint_payload(), path)

    model, tokenizer, checkpoint = load_checkpoint(path)

    assert isinstance(model, TransformerLanguageModel)
    assert tokenizer.vocab_size == 3
    assert checkpoint["step"] == 1


@pytest.mark.parametrize("key", ["model_state_dict", "config", "step", "losses"])
def test_load_checkpoint_rejects_missing_required_keys(tmp_path, key: str) -> None:
    payload = checkpoint_payload()
    del payload[key]
    path = tmp_path / "checkpoint.pt"
    torch.save(payload, path)

    with pytest.raises(ValueError, match=key):
        load_checkpoint(path)


def test_load_tokenizer_supports_legacy_tokenizer_chars() -> None:
    tokenizer = load_tokenizer({"tokenizer_chars": ["a", "b", "c"]})

    assert tokenizer.decode([0, 1, 2]) == "abc"


def test_load_tokenizer_rejects_missing_tokenizer_payload() -> None:
    with pytest.raises(ValueError, match="tokenizer"):
        load_tokenizer({})
