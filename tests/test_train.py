import csv
from argparse import Namespace

import pytest
import torch

from llm.config import compact_defaults, config_defaults, load_toml
from llm.models import TransformerConfig, TransformerLanguageModel
from llm.tokenizer import CharTokenizer
from llm.train import (
    append_metrics_row,
    count_parameters,
    learning_rate_for_step,
    parse_args,
    save_checkpoint,
    validate_args,
)


def test_append_metrics_row_writes_header_and_values(tmp_path) -> None:
    path = tmp_path / "metrics.csv"

    append_metrics_row(path, 0, {"train": 1.0, "val": 2.0})

    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    assert rows == [
        {
            "step": "0",
            "train_loss": "1.0",
            "val_loss": "2.0",
            "train_ppl": "2.718281828459045",
            "val_ppl": "7.38905609893065",
        }
    ]


def test_append_metrics_row_appends_without_rewriting_header(tmp_path) -> None:
    path = tmp_path / "metrics.csv"

    append_metrics_row(path, 0, {"train": 1.0, "val": 2.0})
    append_metrics_row(path, 1, {"train": 0.5, "val": 1.5})

    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "step,train_loss,val_loss,train_ppl,val_ppl"
    assert len(lines) == 3


def valid_args(**overrides) -> Namespace:
    values = {
        "max_iters": 1,
        "eval_interval": 1,
        "eval_iters": 1,
        "block_size": 1,
        "batch_size": 1,
        "embedding_dim": 1,
        "num_heads": 1,
        "num_layers": 1,
        "generate_tokens": 1,
        "dropout": 0.0,
        "learning_rate": 1e-3,
        "warmup_iters": 0,
        "lr_decay_iters": 1,
        "min_learning_rate": None,
        "temperature": 1.0,
        "top_k": None,
        "seed": 1337,
    }
    values.update(overrides)
    return Namespace(**values)


def test_validate_args_rejects_non_positive_integer_fields() -> None:
    with pytest.raises(ValueError, match="--eval-interval"):
        validate_args(valid_args(eval_interval=0))


def test_validate_args_rejects_invalid_ranges() -> None:
    with pytest.raises(ValueError, match="--dropout"):
        validate_args(valid_args(dropout=1.0))

    with pytest.raises(ValueError, match="--learning-rate"):
        validate_args(valid_args(learning_rate=0))

    with pytest.raises(ValueError, match="--warmup-iters"):
        validate_args(valid_args(warmup_iters=-1))

    with pytest.raises(ValueError, match="--lr-decay-iters"):
        validate_args(valid_args(lr_decay_iters=0))

    with pytest.raises(ValueError, match="--min-learning-rate"):
        validate_args(valid_args(min_learning_rate=-1e-3))

    with pytest.raises(ValueError, match="--min-learning-rate"):
        validate_args(valid_args(min_learning_rate=2e-3))

    with pytest.raises(ValueError, match="--temperature"):
        validate_args(valid_args(temperature=0))

    with pytest.raises(ValueError, match="--top-k"):
        validate_args(valid_args(top_k=0))


def test_config_defaults_reads_train_config(tmp_path) -> None:
    path = tmp_path / "run.toml"
    path.write_text(
        """
[run]
run_id = "smoke"

[data]
tokens = "data/processed/tokens.pt"

[train]
max_iters = 2
warmup_iters = 1
lr_decay_iters = 2
min_learning_rate = 0.0001
seed = 7

[model]
block_size = 8

[outputs]
checkpoint = "checkpoints/smoke.pt"
""",
        encoding="utf-8",
    )

    defaults = compact_defaults(config_defaults(load_toml(path)))

    assert defaults["run_id"] == "smoke"
    assert defaults["tokens"] == "data/processed/tokens.pt"
    assert defaults["max_iters"] == 2
    assert defaults["warmup_iters"] == 1
    assert defaults["lr_decay_iters"] == 2
    assert defaults["min_learning_rate"] == 0.0001
    assert defaults["seed"] == 7
    assert defaults["block_size"] == 8
    assert defaults["checkpoint"] == "checkpoints/smoke.pt"


def test_parse_args_reads_lr_schedule_from_config(tmp_path, monkeypatch) -> None:
    path = tmp_path / "run.toml"
    path.write_text(
        """
[train]
warmup_iters = 2
lr_decay_iters = 4
min_learning_rate = 0.0001
""",
        encoding="utf-8",
    )
    monkeypatch.setattr("sys.argv", ["train", "--config", str(path)])

    args = parse_args()

    assert args.warmup_iters == 2
    assert args.lr_decay_iters == 4
    assert args.min_learning_rate == 0.0001


def test_save_checkpoint_includes_resume_state(tmp_path) -> None:
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "checkpoint.pt"

    save_checkpoint(
        path=path,
        model=model,
        tokenizer=tokenizer,
        config=config,
        optimizer=optimizer,
        step=2,
        losses={"train": 1.0, "val": 1.1},
        metadata={"run_id": "smoke"},
        tokens_seen=128,
    )

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    assert "optimizer_state_dict" in checkpoint
    assert "rng_state" in checkpoint
    assert checkpoint["tokens_seen"] == 128


def test_learning_rate_for_step_keeps_default_constant_schedule() -> None:
    assert learning_rate_for_step(
        step=5,
        decay_iters=10,
        learning_rate=1e-3,
    ) == 1e-3


def test_learning_rate_for_step_warms_up_then_cosine_decays() -> None:
    assert learning_rate_for_step(
        step=0,
        decay_iters=6,
        learning_rate=1e-3,
        warmup_iters=2,
        min_learning_rate=1e-4,
    ) == pytest.approx(5e-4)

    assert learning_rate_for_step(
        step=2,
        decay_iters=6,
        learning_rate=1e-3,
        warmup_iters=2,
        min_learning_rate=1e-4,
    ) == pytest.approx(1e-3)

    assert learning_rate_for_step(
        step=6,
        decay_iters=6,
        learning_rate=1e-3,
        warmup_iters=2,
        min_learning_rate=1e-4,
    ) == pytest.approx(1e-4)


def test_count_parameters_deduplicates_tied_weights() -> None:
    config = TransformerConfig(
        vocab_size=5,
        block_size=3,
        embedding_dim=4,
        num_heads=2,
        num_layers=1,
    )
    model = TransformerLanguageModel(config)
    unique_parameters = {id(parameter): parameter for parameter in model.parameters()}

    assert count_parameters(model) == sum(
        parameter.numel() for parameter in unique_parameters.values()
    )
