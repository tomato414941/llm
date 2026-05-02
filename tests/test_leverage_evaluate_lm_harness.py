from pathlib import Path

import pytest

from llm.leverage import evaluate_lm_harness
from llm.leverage.evaluate_lm_harness import (
    build_lm_eval_command,
    model_args,
    run_lm_harness,
    selected_runs,
)


def test_model_args_uses_base_model_without_adapter() -> None:
    assert model_args(base_model="Qwen/Qwen3.5-9B", adapter_dir=None) == (
        "pretrained=Qwen/Qwen3.5-9B,trust_remote_code=True"
    )


def test_model_args_adds_peft_adapter_path() -> None:
    assert model_args(
        base_model="Qwen/Qwen3.5-9B",
        adapter_dir=Path("outputs/leverage-sft-qwen35-9b/lora-adapter"),
    ) == (
        "pretrained=Qwen/Qwen3.5-9B,"
        "trust_remote_code=True,"
        "peft=outputs/leverage-sft-qwen35-9b/lora-adapter"
    )


def test_build_lm_eval_command_defaults_to_hf_backend() -> None:
    command = build_lm_eval_command(
        base_model="Qwen/Qwen3.5-9B",
        adapter_dir=None,
        tasks=["ifeval"],
        output_path=Path("outputs/leverage-lm-harness/base"),
        device="cuda:0",
        batch_size="auto",
        apply_chat_template=True,
        limit=None,
        log_samples=False,
    )

    assert command == [
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        "pretrained=Qwen/Qwen3.5-9B,trust_remote_code=True",
        "--tasks",
        "ifeval",
        "--device",
        "cuda:0",
        "--batch_size",
        "auto",
        "--output_path",
        "outputs/leverage-lm-harness/base",
        "--apply_chat_template",
    ]


def test_selected_runs_expands_both() -> None:
    assert selected_runs("both") == ["base", "adapter"]
    assert selected_runs("base") == ["base"]
    assert selected_runs("adapter") == ["adapter"]


def test_run_lm_harness_dry_run_prints_base_and_adapter_commands() -> None:
    lines = run_lm_harness(
        base_model="Qwen/Qwen3.5-9B",
        adapter_dir=Path("outputs/leverage-sft-qwen35-9b/lora-adapter"),
        output_root=Path("outputs/leverage-lm-harness"),
        tasks=["ifeval"],
        device="cuda:0",
        batch_size="auto",
        apply_chat_template=True,
        limit=10,
        log_samples=True,
        run="both",
        dry_run=True,
    )

    assert len(lines) == 2
    assert lines[0].startswith("base: lm_eval")
    assert "pretrained=Qwen/Qwen3.5-9B" in lines[0]
    assert "peft=" not in lines[0]
    assert lines[1].startswith("adapter: lm_eval")
    assert "peft=outputs/leverage-sft-qwen35-9b/lora-adapter" in lines[1]
    assert "--limit 10" in lines[0]
    assert "--limit 10" in lines[1]
    assert "--log_samples" in lines[0]
    assert "--log_samples" in lines[1]


def test_run_lm_harness_requires_adapter_for_real_adapter_run(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="adapter directory"):
        run_lm_harness(
            base_model="Qwen/Qwen3.5-9B",
            adapter_dir=tmp_path / "missing-adapter",
            output_root=tmp_path / "outputs",
            tasks=["ifeval"],
            device="cuda:0",
            batch_size="auto",
            apply_chat_template=True,
            limit=None,
            log_samples=False,
            run="adapter",
            dry_run=False,
        )


def test_run_lm_harness_requires_lm_eval_for_real_base_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(evaluate_lm_harness.shutil, "which", lambda name: None)

    with pytest.raises(RuntimeError, match="lm_eval is not installed"):
        run_lm_harness(
            base_model="Qwen/Qwen3.5-9B",
            adapter_dir=tmp_path / "unused-adapter",
            output_root=tmp_path / "outputs",
            tasks=["ifeval"],
            device="cuda:0",
            batch_size="auto",
            apply_chat_template=True,
            limit=None,
            log_samples=False,
            run="base",
            dry_run=False,
        )
