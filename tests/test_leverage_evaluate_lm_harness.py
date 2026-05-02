import json
from pathlib import Path

import pytest

from llm.leverage import evaluate_lm_harness
from llm.leverage.evaluate_lm_harness import (
    build_lm_eval_command,
    model_args,
    run_command_with_timing,
    run_lm_harness,
    update_generation_timing,
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
        variant="base",
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
        variant="adapter",
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
            variant="adapter",
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
            variant="base",
            dry_run=False,
        )


def test_update_generation_timing_records_first_and_last_marker(monkeypatch: pytest.MonkeyPatch) -> None:
    values = iter([12.0, 15.5])
    monkeypatch.setattr(evaluate_lm_harness.time, "monotonic", lambda: next(values))
    timing: dict[str, object] = {
        "generation_started_after_seconds": None,
        "generation_last_seen_after_seconds": None,
    }

    update_generation_timing("Running generate_until requests: 10%", timing, started=10.0)
    update_generation_timing("Running generate_until requests: 100%", timing, started=10.0)

    assert timing["generation_started_after_seconds"] == 2.0
    assert timing["generation_last_seen_after_seconds"] == 5.5


def test_update_generation_timing_ignores_non_generation_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evaluate_lm_harness.time, "monotonic", lambda: 12.0)
    timing: dict[str, object] = {
        "generation_started_after_seconds": None,
        "generation_last_seen_after_seconds": None,
    }

    update_generation_timing("Loading weights", timing, started=10.0)

    assert timing["generation_started_after_seconds"] is None
    assert timing["generation_last_seen_after_seconds"] is None


def test_run_command_with_timing_writes_timing_json(tmp_path: Path) -> None:
    timing_path = tmp_path / "timing.json"

    run_command_with_timing(
        [
            "sh",
            "-c",
            "printf '%s\\n' 'Loading weights' 'Running generate_until requests: 100%'",
        ],
        timing_path,
    )

    timing = json.loads(timing_path.read_text(encoding="utf-8"))
    assert timing["returncode"] == 0
    assert timing["elapsed_seconds"] >= 0
    assert timing["generation_started_after_seconds"] is not None
    assert timing["generation_last_seen_after_seconds"] is not None
    assert timing["generation_seconds"] >= 0
