from argparse import Namespace

import pytest

from llm.observe import validate_args


def valid_args(**overrides) -> Namespace:
    values = {
        "batch_size": 1,
        "eval_iters": 1,
        "max_new_tokens": 1,
        "temperature": 1.0,
        "top_k": None,
        "samples": 1,
    }
    values.update(overrides)
    return Namespace(**values)


def test_validate_args_rejects_non_positive_values() -> None:
    with pytest.raises(ValueError, match="--batch-size"):
        validate_args(valid_args(batch_size=0))

    with pytest.raises(ValueError, match="--eval-iters"):
        validate_args(valid_args(eval_iters=0))

    with pytest.raises(ValueError, match="--max-new-tokens"):
        validate_args(valid_args(max_new_tokens=0))

    with pytest.raises(ValueError, match="--temperature"):
        validate_args(valid_args(temperature=0))

    with pytest.raises(ValueError, match="--top-k"):
        validate_args(valid_args(top_k=0))

    with pytest.raises(ValueError, match="--samples"):
        validate_args(valid_args(samples=0))
