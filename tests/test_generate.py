from argparse import Namespace

import pytest
import torch

from llm.generate import generate_sample, generate_samples, prompt_context_ids, validate_args
from llm.models import TransformerConfig, TransformerLanguageModel
from llm.tokenizer import CharTokenizer


def valid_args(**overrides) -> Namespace:
    values = {
        "max_new_tokens": 1,
        "temperature": 1.0,
        "top_k": None,
        "samples": 1,
    }
    values.update(overrides)
    return Namespace(**values)


def test_validate_args_rejects_non_positive_values() -> None:
    with pytest.raises(ValueError, match="--max-new-tokens"):
        validate_args(valid_args(max_new_tokens=0))

    with pytest.raises(ValueError, match="--temperature"):
        validate_args(valid_args(temperature=0))

    with pytest.raises(ValueError, match="--top-k"):
        validate_args(valid_args(top_k=0))

    with pytest.raises(ValueError, match="--samples"):
        validate_args(valid_args(samples=0))


def test_prompt_context_ids_wraps_tokenizer_errors() -> None:
    tokenizer = CharTokenizer.from_text("abc")

    with pytest.raises(ValueError, match="prompt cannot be encoded"):
        prompt_context_ids(tokenizer, "z")


def test_generate_sample_is_reproducible_with_seed() -> None:
    tokenizer = CharTokenizer.from_text("abc")
    model = TransformerLanguageModel(
        TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=4,
            embedding_dim=4,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
        )
    )
    model.eval()

    torch.manual_seed(123)
    first = generate_sample(
        model=model,
        tokenizer=tokenizer,
        prompt="a",
        max_new_tokens=3,
        temperature=1.0,
        top_k=None,
    )
    torch.manual_seed(123)
    second = generate_sample(
        model=model,
        tokenizer=tokenizer,
        prompt="a",
        max_new_tokens=3,
        temperature=1.0,
        top_k=None,
    )

    assert first == second


def test_generate_samples_returns_requested_count() -> None:
    tokenizer = CharTokenizer.from_text("abc")
    model = TransformerLanguageModel(
        TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=4,
            embedding_dim=4,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
        )
    )

    samples = generate_samples(
        model=model,
        tokenizer=tokenizer,
        prompt="a",
        max_new_tokens=1,
        temperature=1.0,
        top_k=None,
        samples=2,
    )

    assert len(samples) == 2
