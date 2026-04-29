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
        seeds=Path("tracks/leverage/prompts/instruction-seeds.jsonl"),
        seed_id=[],
        model="qwen/qwen3.6-plus",
        model_label="qwen3-6-plus-openrouter",
        generator_candidate=[],
        random_seed=0,
        output=Path("tracks/leverage/runs/instruction-outputs/qwen.jsonl"),
        max_tokens=16384,
        temperature=0.2,
        thinking_mode="none",
        reasoning_effort="none",
        exclude_reasoning=True,
        timeout_seconds=60.0,
        resume=False,
        repo_root=tmp_path,
        dry_run=True,
    )


def write_repo_shape(tmp_path: Path) -> None:
    prompt_dir = tmp_path / "tracks" / "leverage" / "prompts"
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "instruction-seeds.jsonl").write_text("{}", encoding="utf-8")


def test_parse_args_defaults_to_openrouter_instruction_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["openai_compatible_instruction_once.py"])

    parsed = normalize_args(parse_args())

    assert parsed.base_url == "https://openrouter.ai/api/v1"
    assert parsed.model == "qwen/qwen3.6-plus"
    assert parsed.model_label == "qwen3-6-plus-openrouter"
    assert parsed.secret_path == Path.home() / ".secrets" / "openrouter"
    assert parsed.seeds == Path("tracks/leverage/prompts/instruction-seeds.jsonl")
    assert parsed.output == Path("tracks/leverage/runs/instruction-outputs/qwen3-6-plus-openrouter.jsonl")
    assert parsed.max_tokens == 16384
    assert parsed.reasoning_effort == "none"
    assert parsed.exclude_reasoning is True
    assert parsed.resume is False


def test_collect_command_targets_instruction_collector(tmp_path: Path) -> None:
    run_args = args(tmp_path)

    command = collect_command(run_args)

    assert "llm.leverage.collect_instructions" in command
    assert command[command.index("--seeds") + 1] == "tracks/leverage/prompts/instruction-seeds.jsonl"
    assert command[command.index("--model") + 1] == "qwen/qwen3.6-plus"
    assert command[command.index("--random-seed") + 1] == "0"
    assert command[command.index("--output") + 1] == "tracks/leverage/runs/instruction-outputs/qwen.jsonl"
    assert command[command.index("--reasoning-effort") + 1] == "none"
    assert "--exclude-reasoning" in command
    assert "--overwrite" in command
    assert "--resume" not in command


def test_collect_command_can_pass_random_generator_candidates(tmp_path: Path) -> None:
    run_args = args(tmp_path)
    run_args.generator_candidate = [
        "qwen3-6-plus-openrouter=qwen/qwen3.6-plus:0.6",
        "gpt-5-4-openrouter=openai/gpt-5.4:0.4",
    ]
    run_args.random_seed = 123

    command = collect_command(run_args)

    assert command.count("--generator-candidate") == 2
    assert command[command.index("--generator-candidate") + 1] == (
        "qwen3-6-plus-openrouter=qwen/qwen3.6-plus:0.6"
    )
    assert command[command.index("--random-seed") + 1] == "123"


def test_collect_command_can_resume_without_overwrite(tmp_path: Path) -> None:
    run_args = args(tmp_path)
    run_args.resume = True

    command = collect_command(run_args)

    assert "--resume" in command
    assert "--overwrite" not in command


def test_collect_command_can_target_seed_ids(tmp_path: Path) -> None:
    run_args = args(tmp_path)
    run_args.seed_id = ["lt_seed_051", "lt_seed_052"]

    command = collect_command(run_args)

    assert command.count("--seed-id") == 2
    assert command[command.index("--seed-id") + 1] == "lt_seed_051"
    assert command[command.index("--seed-id", command.index("--seed-id") + 1) + 1] == "lt_seed_052"


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
