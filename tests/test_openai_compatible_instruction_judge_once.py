from argparse import Namespace
import importlib.util
from pathlib import Path
import sys

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "leverage"
    / "openai_compatible_instruction_judge_once.py"
)
SPEC = importlib.util.spec_from_file_location("openai_compatible_instruction_judge_once", MODULE_PATH)
assert SPEC is not None
openai_compatible_instruction_judge_once = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(openai_compatible_instruction_judge_once)

api_key_for_run = openai_compatible_instruction_judge_once.api_key_for_run
judge_command = openai_compatible_instruction_judge_once.judge_command
judge_env = openai_compatible_instruction_judge_once.judge_env
normalize_args = openai_compatible_instruction_judge_once.normalize_args
parse_args = openai_compatible_instruction_judge_once.parse_args
preflight = openai_compatible_instruction_judge_once.preflight


def args(tmp_path: Path) -> Namespace:
    return Namespace(
        base_url="https://openrouter.ai/api/v1",
        api_key=None,
        secret_path=tmp_path / "openrouter",
        input=Path("tracks/leverage/runs/instruction-outputs/candidates.jsonl"),
        output=Path("tracks/leverage/runs/instruction-outputs/judgments.jsonl"),
        csv_output=Path("tracks/leverage/runs/instruction-outputs/judgments.csv"),
        summary_output=Path("tracks/leverage/runs/instruction-outputs/judgments-summary.csv"),
        judge_model="judge/model",
        judge_label="judge_label",
        max_tokens=512,
        temperature=0.0,
        reasoning_effort="none",
        exclude_reasoning=True,
        timeout_seconds=60.0,
        limit=2,
        resume=False,
        repo_root=tmp_path,
        dry_run=True,
    )


def test_parse_args_defaults_to_openrouter_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["openai_compatible_instruction_judge_once.py"])

    parsed = normalize_args(parse_args())

    assert parsed.base_url == "https://openrouter.ai/api/v1"
    assert parsed.secret_path == Path.home() / ".secrets" / "openrouter"
    assert parsed.input == Path("tracks/leverage/runs/instruction-outputs/qwen3-5-flash-openrouter-candidates.jsonl")
    assert parsed.output == Path("tracks/leverage/runs/instruction-outputs/qwen3-5-flash-openrouter-judgments.jsonl")
    assert parsed.summary_output == Path(
        "tracks/leverage/runs/instruction-outputs/qwen3-5-flash-openrouter-judgments-summary.csv"
    )
    assert parsed.resume is False


def test_judge_command_targets_judge_module(tmp_path: Path) -> None:
    command = judge_command(args(tmp_path))

    assert "llm.leverage.judge_instruction_outputs" in command
    assert command[command.index("--judge-model") + 1] == "judge/model"
    assert command[command.index("--summary-output") + 1] == (
        "tracks/leverage/runs/instruction-outputs/judgments-summary.csv"
    )
    assert command[command.index("--reasoning-effort") + 1] == "none"
    assert "--exclude-reasoning" in command
    assert command[command.index("--limit") + 1] == "2"
    assert "--overwrite" in command
    assert "--resume" not in command


def test_judge_command_can_resume_without_overwrite(tmp_path: Path) -> None:
    run_args = args(tmp_path)
    run_args.resume = True

    command = judge_command(run_args)

    assert "--resume" in command
    assert "--overwrite" not in command


def test_judge_env_keeps_api_key_out_of_process_arguments(tmp_path: Path) -> None:
    run_args = args(tmp_path)
    command = judge_command(run_args)
    env = judge_env(run_args, "secret-key")

    assert "secret-key" not in command
    assert env["OPENAI_API_KEY"] == "secret-key"


def test_dry_run_does_not_require_input_or_secret(tmp_path: Path) -> None:
    run_args = args(tmp_path)
    run_args.dry_run = True

    preflight(run_args)
    assert api_key_for_run(run_args) == "dry-run-api-key"


def test_non_dry_run_requires_input(tmp_path: Path) -> None:
    run_args = args(tmp_path)
    run_args.dry_run = False

    with pytest.raises(FileNotFoundError, match="input file"):
        preflight(run_args)
