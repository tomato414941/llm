import csv
from pathlib import Path

import torch

from llm.models import TransformerConfig, TransformerLanguageModel
from llm.scaling import (
    latest_summary_row,
    scaling_row,
    upsert_scaling_row,
)
from llm.tokenizer import CharTokenizer


def checkpoint_payload(**metadata_overrides) -> dict[str, object]:
    config = TransformerConfig(
        vocab_size=3,
        block_size=4,
        embedding_dim=8,
        num_heads=2,
        num_layers=1,
        dropout=0.1,
    )
    model = TransformerLanguageModel(config)
    tokenizer = CharTokenizer.from_text("abc")
    metadata = {
        "run_id": "smoke",
        "tokens": "tracks/from-scratch/data/processed/tokens.pt",
        "parameter_count": 1234,
        "max_iters": 10,
        "batch_size": 2,
        "learning_rate": 0.001,
    }
    metadata.update(metadata_overrides)
    return {
        "model_state_dict": model.state_dict(),
        "tokenizer": tokenizer.to_payload(),
        "config": config.to_dict(),
        "step": 9,
        "losses": {"train": 1.0, "val": 1.1},
        "metadata": metadata,
        "tokens_seen": 80,
    }


def test_scaling_row_combines_checkpoint_and_summary() -> None:
    checkpoint = checkpoint_payload()
    summary = {
        "validation_loss": "1.25",
        "validation_perplexity": "3.49",
        "prompt_id": "king",
        "prompt": "KING:",
        "seed": "1337",
        "samples": "2",
        "max_new_tokens": "20",
        "temperature": "1.0",
        "top_k": "10",
    }

    row = scaling_row(
        checkpoint_path=Path("tracks/from-scratch/checkpoints/smoke.pt"),
        checkpoint=checkpoint,
        summary_row=summary,
        note="baseline",
    )

    assert row["run_id"] == "smoke"
    assert row["tokens_seen"] == 80
    assert row["parameter_count"] == 1234
    assert row["embedding_dim"] == 8
    assert row["validation_loss"] == "1.25"
    assert row["prompt_id"] == "king"
    assert row["note"] == "baseline"


def test_scaling_row_falls_back_to_checkpoint_stem_without_run_id(tmp_path) -> None:
    checkpoint = checkpoint_payload(run_id="")

    row = scaling_row(tmp_path / "legacy.pt", checkpoint, None, "")

    assert row["run_id"] == "legacy"
    assert row["validation_loss"] == ""


def test_latest_summary_row_matches_checkpoint_name() -> None:
    rows = [
        {"checkpoint_path": "tracks/from-scratch/checkpoints/model.pt", "validation_loss": "2.0"},
        {"checkpoint_path": "/tmp/model.pt", "validation_loss": "1.5"},
    ]

    row = latest_summary_row(rows, Path("tracks/from-scratch/checkpoints/model.pt"))

    assert row is not None
    assert row["validation_loss"] == "1.5"


def test_upsert_scaling_row_replaces_existing_run_id(tmp_path) -> None:
    path = tmp_path / "scaling.csv"
    first = scaling_row(tmp_path / "first.pt", checkpoint_payload(), None, "first")
    second = scaling_row(tmp_path / "second.pt", checkpoint_payload(), None, "second")

    upsert_scaling_row(path, first)
    upsert_scaling_row(path, second)

    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    assert len(rows) == 1
    assert rows[0]["run_id"] == "smoke"
    assert rows[0]["note"] == "second"


def test_scaling_cli_inputs_can_load_checkpoint_payload(tmp_path) -> None:
    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint_payload(), path)

    payload = torch.load(path, map_location="cpu", weights_only=True)

    row = scaling_row(path, payload, None, "loaded")
    assert row["run_id"] == "smoke"
