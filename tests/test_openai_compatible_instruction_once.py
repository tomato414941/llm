from argparse import Namespace
import importlib.util
from pathlib import Path
import sys

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "leverage" / "openai_compatible_instruction_once.py"
SPEC = importlib.util.spec_from_file_location("openai_compatible_instruction_once", MODULE_PATH)
assert SPEC is not None
openai_compatible_instruction_once = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(openai_compatible_instruction_once)

api_key_for_run = openai_compatible_instruction_once.api_key_for_run
collect_command = openai_compatible_instruction_once.collect_command
collect_env = openai_compatible_instruction_once.collect_env
normalize_args = openai_compatible_instruction_once.normalize_args
parse_args = openai_compatible_instruction_once.parse_args
preflight = openai_compatible_instruction_once.preflight
redact_command = openai_compatible_instruction_once.redact_command


def args(tmp_path: Path) -> Namespace:
    return Namespace(
        base_url="https://openrouter.ai/api/v1",
        api_key=None,
        secret_path=tmp_path / "openrouter",
        seeds=Path("prompts/leverage_training_seed_v0.jsonl"),
        model="qwen/qwen3.5-flash-02-23",
        model_label="qwen3_5_flash_openrouter",
        output=Path("experiments/leverage/instruction_outputs/qwen.jsonl"),
        max_tokens=512,
        temperature=0.2,
        thinking_mode="none",
        timeout_seconds=60.0,
        repo_root=tmp_path,
        dry_run=True,
    )


def write_repo_shape(tmp_path: Path) -> None:
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "leverage_training_seed_v0.jsonl").write_text("{}", encoding="utf-8")


def test_parse_args_defaults_to_openrouter_instruction_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["openai_compatible_instruction_once.py"])

    parsed = normalize_args(parse_args())

    assert parsed.base_url == "https://openrouter.ai/api/v1"
    assert parsed.model == "qwen/qwen3.5-flash-02-23"
    assert parsed.model_label == "qwen3_5_flash_openrouter"
    assert parsed.secret_path == Path.home() / ".secrets" / "openrouter"
    assert parsed.seeds == Path("prompts/leverage_training_seed_v0.jsonl")
    assert parsed.output == Path("experiments/leverage/instruction_outputs/qwen3_5_flash_openrouter.jsonl")


def test_collect_command_targets_instruction_collector(tmp_path: Path) -> None:
    run_args = args(tmp_path)

    command = collect_command(run_args)

    assert "llm.leverage.collect_instructions" in command
    assert command[command.index("--seeds") + 1] == "prompts/leverage_training_seed_v0.jsonl"
    assert command[command.index("--model") + 1] == "qwen/qwen3.5-flash-02-23"
    assert command[command.index("--output") + 1] == "experiments/leverage/instruction_outputs/qwen.jsonl"


def test_collect_env_keeps_api_key_out_of_process_arguments(tmp_path: Path) -> None:
    run_args = args(tmp_path)

    command = collect_command(run_args)
    env = collect_env(run_args, "secret-key")

    assert "secret-key" not in command
    assert env["OPENAI_API_KEY"] == "secret-key"


def test_dry_run_does_not_require_secret_file(tmp_path: Path) -> None:
    run_args = args(tmp_path)
    run_args.dry_run = True

    assert api_key_for_run(run_args) == "dry-run-api-key"


def test_redact_command_hides_api_key() -> None:
    command = ["env", "OPENAI_API_KEY=secret-key", "uv"]

    assert redact_command(command, ["secret-key"]) == "env OPENAI_API_KEY=[REDACTED] uv"


def test_preflight_requires_seed_file(tmp_path: Path) -> None:
    run_args = args(tmp_path)

    with pytest.raises(FileNotFoundError, match="seed file"):
        preflight(run_args)

    write_repo_shape(tmp_path)

    preflight(run_args)
