from argparse import Namespace
import importlib.util
from pathlib import Path
import sys

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "leverage" / "openai_compatible_eval_once.py"
SPEC = importlib.util.spec_from_file_location("openai_compatible_eval_once", MODULE_PATH)
assert SPEC is not None
openai_compatible_eval_once = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(openai_compatible_eval_once)

api_key_for_run = openai_compatible_eval_once.api_key_for_run
collect_command = openai_compatible_eval_once.collect_command
collect_env = openai_compatible_eval_once.collect_env
display_command = openai_compatible_eval_once.display_command
evaluate_command = openai_compatible_eval_once.evaluate_command
normalize_args = openai_compatible_eval_once.normalize_args
parse_args = openai_compatible_eval_once.parse_args
preflight = openai_compatible_eval_once.preflight
redact_command = openai_compatible_eval_once.redact_command
run_command = openai_compatible_eval_once.run_command


def args(tmp_path: Path) -> Namespace:
    return Namespace(
        base_url="https://openrouter.ai/api/v1",
        api_key=None,
        secret_path=tmp_path / "llm-openrouter",
        tasks=[Path("tracks/leverage/evals/leverage-smoke.jsonl"), Path("tracks/leverage/evals/project-judgment.jsonl")],
        model="qwen/qwen3.5-flash-02-23",
        model_label="qwen3-5-flash-openrouter",
        output=Path("tracks/leverage/runs/qwen.jsonl"),
        scores_output=Path("tracks/leverage/runs/qwen-scores.csv"),
        summary_output=Path("tracks/leverage/runs/qwen-summary.csv"),
        max_tokens=512,
        temperature=0.0,
        thinking_mode="none",
        reasoning_effort="none",
        exclude_reasoning=True,
        timeout_seconds=60.0,
        repo_root=tmp_path,
        dry_run=True,
    )


def write_repo_shape(tmp_path: Path) -> None:
    eval_dir = tmp_path / "tracks" / "leverage" / "evals"
    eval_dir.mkdir(parents=True)
    (eval_dir / "leverage-smoke.jsonl").write_text("{}", encoding="utf-8")
    (eval_dir / "project-judgment.jsonl").write_text("{}", encoding="utf-8")


def test_parse_args_defaults_to_openrouter_qwen(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["openai_compatible_eval_once.py"])

    parsed = normalize_args(parse_args())

    assert parsed.base_url == "https://openrouter.ai/api/v1"
    assert parsed.model == "qwen/qwen3.5-flash-02-23"
    assert parsed.model_label == "qwen3-5-flash-openrouter"
    assert parsed.secret_path == Path.home() / ".secrets" / "openrouter"
    assert parsed.tasks == [
        Path("tracks/leverage/evals/leverage-smoke.jsonl"),
        Path("tracks/leverage/evals/project-judgment.jsonl"),
    ]
    assert parsed.thinking_mode == "none"
    assert parsed.reasoning_effort == "none"
    assert parsed.exclude_reasoning is True


def test_collect_command_targets_openai_compatible_api(tmp_path: Path) -> None:
    run_args = args(tmp_path)

    command = collect_command(run_args, "secret-key")

    assert command[:4] == [
        "uv",
        "run",
        "python",
        "-u",
    ]
    assert "llm.leverage.collect_openai" in command
    assert command[command.index("--model") + 1] == "qwen/qwen3.5-flash-02-23"
    assert command[command.index("--model-label") + 1] == "qwen3-5-flash-openrouter"
    assert command[command.index("--thinking-mode") + 1] == "none"
    assert command[command.index("--reasoning-effort") + 1] == "none"
    assert "--exclude-reasoning" in command


def test_collect_env_keeps_api_key_out_of_process_arguments(tmp_path: Path) -> None:
    run_args = args(tmp_path)

    command = collect_command(run_args, "secret-key")
    env = collect_env(run_args, "secret-key")

    assert "secret-key" not in command
    assert env == {
        "OPENAI_BASE_URL": "https://openrouter.ai/api/v1",
        "OPENAI_API_KEY": "secret-key",
    }


def test_display_command_includes_redactable_env_for_dry_run(tmp_path: Path) -> None:
    run_args = args(tmp_path)
    command = collect_command(run_args, "secret-key")
    env = collect_env(run_args, "secret-key")

    displayed = display_command(command, env_overrides=env)

    assert displayed[:3] == [
        "env",
        "OPENAI_API_KEY=secret-key",
        "OPENAI_BASE_URL=https://openrouter.ai/api/v1",
    ]


def test_evaluate_command_scores_saved_predictions(tmp_path: Path) -> None:
    command = evaluate_command(args(tmp_path))

    assert "llm.leverage.evaluate" in command
    assert command[command.index("--predictions") + 1] == "tracks/leverage/runs/qwen.jsonl"
    assert command[command.index("--summary-output") + 1] == "tracks/leverage/runs/qwen-summary.csv"


def test_dry_run_does_not_require_secret_file(tmp_path: Path) -> None:
    run_args = args(tmp_path)
    run_args.dry_run = True

    assert api_key_for_run(run_args) == "dry-run-api-key"


def test_non_dry_run_reads_secret_file(tmp_path: Path) -> None:
    run_args = args(tmp_path)
    run_args.dry_run = False
    run_args.secret_path.write_text("secret-key\n", encoding="utf-8")

    assert api_key_for_run(run_args) == "secret-key"


def test_redact_command_hides_api_key() -> None:
    command = ["env", "OPENAI_API_KEY=secret-key", "uv"]

    assert redact_command(command, ["secret-key"]) == "env OPENAI_API_KEY=[REDACTED] uv"


def test_run_command_redacts_failed_command(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_run(command, cwd, check, env):
        assert check is False
        assert cwd == tmp_path
        assert env["OPENAI_API_KEY"] == "secret-key"
        return openai_compatible_eval_once.subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(openai_compatible_eval_once.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError) as exc_info:
        run_command(
            ["env", "OPENAI_API_KEY=secret-key", "uv"],
            cwd=tmp_path,
            secrets=["secret-key"],
            dry_run=False,
            env_overrides={"OPENAI_API_KEY": "secret-key"},
        )

    message = str(exc_info.value)
    assert "secret-key" not in message
    assert "OPENAI_API_KEY=[REDACTED]" in message


def test_preflight_requires_committed_task_files(tmp_path: Path) -> None:
    run_args = args(tmp_path)

    with pytest.raises(FileNotFoundError, match="leverage-smoke"):
        preflight(run_args)

    write_repo_shape(tmp_path)

    preflight(run_args)
