from pathlib import Path

import pytest

from llm.leverage import evaluate_lm_harness
from llm.leverage.evaluate_lm_harness import (
    build_lm_eval_command,
    model_args,
    run_lm_harness,
)


def test_model_args_uses_base_model_without_adapter() -> None:
    assert model_args(
        base_model="Qwen/Qwen3.5-9B",
        adapter_dir=None,
        enable_thinking=None,
        think_end_token=None,
    ) == (
        "pretrained=Qwen/Qwen3.5-9B,trust_remote_code=True"
    )


def test_model_args_adds_peft_adapter_path() -> None:
    assert model_args(
        base_model="Qwen/Qwen3.5-9B",
        adapter_dir=Path("outputs/leverage-sft-qwen35-9b/lora-adapter"),
        enable_thinking=None,
        think_end_token=None,
    ) == (
        "pretrained=Qwen/Qwen3.5-9B,"
        "trust_remote_code=True,"
        "peft=outputs/leverage-sft-qwen35-9b/lora-adapter"
    )


def test_model_args_adds_thinking_arguments() -> None:
    assert model_args(
        base_model="Qwen/Qwen3.5-9B",
        adapter_dir=None,
        enable_thinking=True,
        think_end_token="</think>",
    ) == (
        "pretrained=Qwen/Qwen3.5-9B,"
        "trust_remote_code=True,"
        "enable_thinking=True,"
        "think_end_token=</think>"
    )


def test_model_args_can_disable_thinking() -> None:
    assert model_args(
        base_model="Qwen/Qwen3.5-9B",
        adapter_dir=None,
        enable_thinking=False,
        think_end_token=None,
    ) == "pretrained=Qwen/Qwen3.5-9B,trust_remote_code=True,enable_thinking=False"


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
        enable_thinking=None,
        think_end_token=None,
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


def test_run_lm_harness_dry_run_prints_base_command() -> None:
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
        enable_thinking=True,
        think_end_token="</think>",
        run="base",
        dry_run=True,
    )

    assert len(lines) == 1
    assert lines[0].startswith("base: lm_eval")
    assert "pretrained=Qwen/Qwen3.5-9B" in lines[0]
    assert "peft=" not in lines[0]
    assert "--limit 10" in lines[0]
    assert "--log_samples" in lines[0]
    assert "enable_thinking=True" in lines[0]
    assert "think_end_token=</think>" in lines[0]


def test_run_lm_harness_dry_run_prints_adapter_command() -> None:
    lines = run_lm_harness(
        base_model="Qwen/Qwen3.5-9B",
        adapter_dir=Path("outputs/leverage-sft-qwen35-9b/lora-adapter"),
        output_root=Path("outputs/leverage-lm-harness"),
        tasks=["ifeval"],
        device="cuda:0",
        batch_size="auto",
        apply_chat_template=True,
        limit=10,
        log_samples=False,
        enable_thinking=False,
        think_end_token=None,
        run="adapter",
        dry_run=True,
    )

    assert len(lines) == 1
    assert lines[0].startswith("adapter: lm_eval")
    assert "pretrained=Qwen/Qwen3.5-9B" in lines[0]
    assert "peft=outputs/leverage-sft-qwen35-9b/lora-adapter" in lines[0]
    assert "--limit 10" in lines[0]
    assert "enable_thinking=False" in lines[0]


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
            enable_thinking=None,
            think_end_token=None,
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
            enable_thinking=None,
            think_end_token=None,
            run="base",
            dry_run=False,
        )
