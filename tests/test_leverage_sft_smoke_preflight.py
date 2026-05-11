from pathlib import Path

import pytest

from llm.leverage.sft_smoke_preflight import preflight


CONFIG_PATH = Path("tracks/leverage/configs/leverage-sft-smoke.toml")
REVIEWED_PATH = Path("tracks/leverage/datasets/reviewed-instructions/bootstrap.jsonl")


def test_preflight_real_config_exports_bootstrap_training_rows() -> None:
    lines = preflight(CONFIG_PATH, overwrite=True)
    expected_rows = len(REVIEWED_PATH.read_text(encoding="utf-8").splitlines())

    assert any(f"exported training rows: {expected_rows}" in line for line in lines)
    assert any("early_stopping.enabled=False" in line for line in lines)
    assert any("checked max runtime minutes: 60" in line for line in lines)
    assert Path("tracks/leverage/sft/bootstrap.train.jsonl").exists()


def test_preflight_rejects_too_many_training_rows(tmp_path: Path) -> None:
    config_path = tmp_path / "leverage-sft-smoke.toml"
    text = CONFIG_PATH.read_text(encoding="utf-8")
    text = text.replace("max_train_examples = 1500", "max_train_examples = 1")
    config_path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="exceeding max_train_examples"):
        preflight(config_path, overwrite=True)


def test_preflight_rejects_runpod_required(tmp_path: Path) -> None:
    config_path = tmp_path / "leverage-sft-smoke.toml"
    text = CONFIG_PATH.read_text(encoding="utf-8")
    text = text.replace("required = false", "required = true")
    config_path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="runpod.required"):
        preflight(config_path, overwrite=True)


def test_preflight_rejects_missing_eval_task(tmp_path: Path) -> None:
    config_path = tmp_path / "leverage-sft-smoke.toml"
    text = CONFIG_PATH.read_text(encoding="utf-8")
    text = text.replace(
        "tracks/leverage/evals/leverage-smoke.jsonl",
        "tracks/leverage/evals/missing-smoke.jsonl",
    )
    config_path.write_text(text, encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="eval task file not found"):
        preflight(config_path, overwrite=True)


def test_preflight_rejects_output_root_outside_outputs(tmp_path: Path) -> None:
    config_path = tmp_path / "leverage-sft-smoke.toml"
    text = CONFIG_PATH.read_text(encoding="utf-8")
    text = text.replace('root = "outputs/leverage-sft-smoke"', 'root = "tracks/bad-output"')
    config_path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="outputs.root"):
        preflight(config_path, overwrite=True)


def test_preflight_refuses_existing_export_without_overwrite(tmp_path: Path) -> None:
    source_config = CONFIG_PATH.read_text(encoding="utf-8")
    export_path = tmp_path / "bootstrap.train.jsonl"
    export_path.write_text("already exists\n", encoding="utf-8")
    config_path = tmp_path / "leverage-sft-smoke.toml"
    text = source_config.replace(
        "tracks/leverage/sft/bootstrap.train.jsonl",
        str(export_path),
    )
    config_path.write_text(text, encoding="utf-8")

    with pytest.raises(FileExistsError, match="output already exists"):
        preflight(config_path, overwrite=False)


def test_preflight_accepts_temp_copy_with_relative_output(tmp_path: Path) -> None:
    config_path = tmp_path / "leverage-sft-smoke.toml"
    export_path = tmp_path / "bootstrap.train.jsonl"
    text = CONFIG_PATH.read_text(encoding="utf-8").replace(
        "tracks/leverage/sft/bootstrap.train.jsonl",
        str(export_path),
    )
    config_path.write_text(text, encoding="utf-8")

    lines = preflight(config_path, overwrite=True)

    assert export_path.exists()
    assert any(str(export_path) in line for line in lines)


def test_preflight_rejects_invalid_early_stopping_config(tmp_path: Path) -> None:
    config_path = tmp_path / "leverage-sft-smoke.toml"
    text = CONFIG_PATH.read_text(encoding="utf-8").replace(
        "enabled = false",
        "enabled = true",
    )
    config_path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="validation_examples"):
        preflight(config_path, overwrite=True)


def test_preflight_allows_longer_runtime_when_early_stopping_enabled(tmp_path: Path) -> None:
    config_path = tmp_path / "leverage-sft-smoke.toml"
    text = CONFIG_PATH.read_text(encoding="utf-8")
    text = text.replace("max_epochs = 1", "max_epochs = 3")
    text = text.replace("max_runtime_minutes = 60", "max_runtime_minutes = 180")
    text = text.replace(
        "[early_stopping]\nenabled = false\nvalidation_examples = 0\neval_every_steps = 0\npatience = 0\nmin_delta = 0.0",
        "[early_stopping]\nenabled = true\nvalidation_examples = 16\neval_every_steps = 10\npatience = 2\nmin_delta = 0.001",
    )
    config_path.write_text(text, encoding="utf-8")

    lines = preflight(config_path, overwrite=True)

    assert any("early_stopping.enabled=True" in line for line in lines)
    assert any("checked max runtime minutes: 180" in line for line in lines)
