from pathlib import Path

import pytest

from llm.leverage import train_sft_smoke
from llm.leverage.train_sft_smoke import (
    gpu_sample_summary,
    load_training_rows,
    nvidia_smi_sample,
    require_training_packages,
    run_smoke,
)


def write_train_export(path: Path, rows: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (
        '{"id":"instr_0001","messages":['
        '{"role":"system","content":"sys"},'
        '{"role":"user","content":"user"},'
        '{"role":"assistant","content":"assistant"}'
        "]}\n"
    )
    path.write_text(line * rows, encoding="utf-8")


def write_config(tmp_path: Path, *, rows: int = 1, max_examples: int = 10) -> Path:
    train_export = tmp_path / "tracks" / "leverage" / "sft" / "bootstrap.train.jsonl"
    write_train_export(train_export, rows=rows)
    config = tmp_path / "tracks" / "leverage" / "configs" / "leverage-sft-smoke.toml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        f"""
[model]
student = "Qwen/Qwen3.5-0.8B"

[data]
train_export = "{train_export}"

[method]
max_train_examples = {max_examples}
max_epochs = 3
batch_size = 2

[outputs]
adapter_dir = "{tmp_path / "outputs" / "lora-adapter"}"
logs = "{tmp_path / "outputs" / "logs"}"
metrics = "{tmp_path / "outputs" / "metrics.csv"}"
notes = "{tmp_path / "outputs" / "notes.md"}"
""",
        encoding="utf-8",
    )
    return config


def test_run_smoke_dry_run_reports_training_plan(tmp_path: Path) -> None:
    config = write_config(tmp_path, rows=2)

    lines = run_smoke(config, dry_run=True)

    assert any("would train 2 rows" in line for line in lines)
    assert any("Qwen/Qwen3.5-0.8B" in line for line in lines)
    assert any("batch size: 2" in line for line in lines)
    assert any("max length: 1024" in line for line in lines)
    assert any("gradient checkpointing: False" in line for line in lines)
    assert any("gradient accumulation steps: 1" in line for line in lines)
    assert any("log every steps: 50" in line for line in lines)
    assert any("adapter output" in line for line in lines)


def test_qwen35_9b_config_dry_run_reports_training_plan() -> None:
    config_path = Path("tracks/leverage/configs/leverage-sft-qwen35-9b.toml")
    train_export_path = Path("tracks/leverage/sft/bootstrap.train.jsonl")
    expected_rows = len(train_export_path.read_text(encoding="utf-8").splitlines())

    lines = run_smoke(config_path, dry_run=True)

    assert any(f"would train {expected_rows} rows" in line for line in lines)
    assert any("Qwen/Qwen3.5-9B" in line for line in lines)
    assert any("batch size: 2" in line for line in lines)
    assert any("max length: 512" in line for line in lines)
    assert any("gradient checkpointing: True" in line for line in lines)
    assert any("gradient accumulation steps: 4" in line for line in lines)
    assert any("log every steps: 10" in line for line in lines)


def test_load_training_rows_rejects_too_many_rows(tmp_path: Path) -> None:
    train_export = tmp_path / "bootstrap.train.jsonl"
    write_train_export(train_export, rows=2)

    with pytest.raises(ValueError, match="exceeding max_examples"):
        load_training_rows(train_export, 1)


def test_load_training_rows_requires_messages(tmp_path: Path) -> None:
    train_export = tmp_path / "bootstrap.train.jsonl"
    train_export.write_text('{"id":"bad"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="must contain messages"):
        load_training_rows(train_export, 1)


def test_require_training_packages_reports_missing_optional_dependencies(monkeypatch) -> None:
    monkeypatch.setattr(train_sft_smoke, "REQUIRED_PACKAGES", ("definitely_missing_sft_package",))

    with pytest.raises(RuntimeError, match="missing optional SFT training packages"):
        require_training_packages()


def test_nvidia_smi_sample_parses_gpu_utilization(monkeypatch) -> None:
    def fake_run(*_args, **_kwargs):
        return train_sft_smoke.subprocess.CompletedProcess(
            args=["nvidia-smi"],
            returncode=0,
            stdout="87, 22411, 24564\n",
        )

    monkeypatch.setattr(train_sft_smoke.subprocess, "run", fake_run)

    assert nvidia_smi_sample() == {
        "gpu_utilization_percent": "87",
        "gpu_memory_used_mb": "22411",
        "gpu_memory_total_mb": "24564",
    }


def test_nvidia_smi_sample_falls_back_to_empty_strings(monkeypatch) -> None:
    def fake_run(*_args, **_kwargs):
        raise FileNotFoundError("nvidia-smi unavailable")

    monkeypatch.setattr(train_sft_smoke.subprocess, "run", fake_run)

    assert nvidia_smi_sample() == {
        "gpu_utilization_percent": "",
        "gpu_memory_used_mb": "",
        "gpu_memory_total_mb": "",
    }


def test_gpu_sample_summary_reports_average_and_maximum() -> None:
    summary = gpu_sample_summary(
        [
            {
                "gpu_utilization_percent": "20",
                "gpu_memory_used_mb": "1000",
                "gpu_memory_total_mb": "24000",
            },
            {
                "gpu_utilization_percent": "40",
                "gpu_memory_used_mb": "2000",
                "gpu_memory_total_mb": "24000",
            },
        ]
    )

    assert summary == {
        "gpu_sample_count": 2.0,
        "gpu_utilization_avg_percent": 30.0,
        "gpu_utilization_max_percent": 40.0,
        "gpu_memory_used_max_mb": 2000.0,
        "gpu_memory_total_mb": 24000.0,
    }


def test_gpu_sample_summary_ignores_blank_values() -> None:
    summary = gpu_sample_summary(
        [
            {
                "gpu_utilization_percent": "",
                "gpu_memory_used_mb": "",
                "gpu_memory_total_mb": "",
            }
        ]
    )

    assert summary == {
        "gpu_sample_count": 0.0,
        "gpu_utilization_avg_percent": 0.0,
        "gpu_utilization_max_percent": 0.0,
        "gpu_memory_used_max_mb": 0.0,
        "gpu_memory_total_mb": 0.0,
    }
