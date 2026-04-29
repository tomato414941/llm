from pathlib import Path

import pytest

from llm.leverage.sft_smoke_preflight import preflight


CONFIG_PATH = Path("tracks/leverage/configs/leverage-sft-smoke.toml")


def test_preflight_real_config_exports_bootstrap_training_rows() -> None:
    lines = preflight(CONFIG_PATH, overwrite=True)

    assert any("exported training rows: 15" in line for line in lines)
    assert Path("tracks/leverage/sft/bootstrap.train.jsonl").exists()


def test_preflight_rejects_too_many_training_rows(tmp_path: Path) -> None:
    config_path = tmp_path / "leverage-sft-smoke.toml"
    text = CONFIG_PATH.read_text(encoding="utf-8")
    text = text.replace("max_train_examples = 15", "max_train_examples = 1")
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
