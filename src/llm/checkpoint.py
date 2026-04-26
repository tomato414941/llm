from pathlib import Path
from typing import Any

import torch

from llm.models import TransformerConfig, TransformerLanguageModel
from llm.tokenizer import BPETokenizer, CharTokenizer, tokenizer_from_payload


def require_checkpoint_key(checkpoint: dict[str, Any], key: str) -> Any:
    if key not in checkpoint:
        raise ValueError(f"checkpoint must contain {key!r}")
    return checkpoint[key]


def load_tokenizer(checkpoint: dict[str, Any]) -> CharTokenizer | BPETokenizer:
    if "tokenizer" in checkpoint:
        tokenizer_payload = checkpoint["tokenizer"]
        if not isinstance(tokenizer_payload, dict):
            raise ValueError("checkpoint tokenizer must be a payload dictionary")
        return tokenizer_from_payload(tokenizer_payload)

    if "tokenizer_chars" not in checkpoint:
        raise ValueError("checkpoint must contain 'tokenizer' or 'tokenizer_chars'")

    chars = tuple(checkpoint["tokenizer_chars"])
    return CharTokenizer(
        chars=chars,
        stoi={char: index for index, char in enumerate(chars)},
        itos={index: char for index, char in enumerate(chars)},
    )


def load_checkpoint(
    path: Path,
) -> tuple[TransformerLanguageModel, CharTokenizer | BPETokenizer, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    tokenizer = load_tokenizer(checkpoint)
    config_payload = require_checkpoint_key(checkpoint, "config")
    if not isinstance(config_payload, dict):
        raise ValueError("checkpoint config must be a dictionary")
    model_state_dict = require_checkpoint_key(checkpoint, "model_state_dict")
    require_checkpoint_key(checkpoint, "step")
    require_checkpoint_key(checkpoint, "losses")

    config = TransformerConfig.from_dict(config_payload)
    model = TransformerLanguageModel(config)
    model.load_state_dict(model_state_dict)
    model.eval()
    return model, tokenizer, checkpoint
