#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import shlex
import subprocess


DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "qwen/qwen3.5-flash-02-23"
DEFAULT_MODEL_LABEL = "qwen3-5-flash-openrouter"
DEFAULT_SECRET_PATH = Path.home() / ".secrets" / "openrouter"
DEFAULT_TASKS = [Path("tracks/leverage/evals/leverage-smoke.jsonl"), Path("tracks/leverage/evals/project-judgment-v0.jsonl")]
DEFAULT_OUTPUT = Path("tracks/leverage/runs/qwen3-5-flash-openrouter.jsonl")
DEFAULT_SCORES = Path("tracks/leverage/runs/qwen3-5-flash-openrouter-scores.csv")
DEFAULT_SUMMARY = Path("tracks/leverage/runs/qwen3-5-flash-openrouter-summary.csv")


def load_api_key(path: Path) -> str:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"API key file does not exist: {path}") from exc
    if not value:
        raise ValueError(f"API key file is empty: {path}")
    return value


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def redact_command(command: list[str], secrets: list[str]) -> str:
    text = shell_join(command)
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[REDACTED]")
    return text


def collect_command(args: argparse.Namespace, api_key: str) -> list[str]:
    command = [
        "uv",
        "run",
        "python",
        "-u",
        "-m",
        "llm.leverage.collect_openai",
    ]
    for task in args.tasks:
        command.extend(["--tasks", str(task)])
    command.extend(
        [
            "--model",
            args.model,
            "--model-label",
            args.model_label,
            "--output",
            str(args.output),
            "--max-tokens",
            str(args.max_tokens),
            "--temperature",
            str(args.temperature),
            "--thinking-mode",
            args.thinking_mode,
            "--reasoning-effort",
            args.reasoning_effort,
            "--timeout-seconds",
            str(args.timeout_seconds),
            "--overwrite",
        ]
    )
    if args.exclude_reasoning:
        command.append("--exclude-reasoning")
    else:
        command.append("--no-exclude-reasoning")
    return command


def collect_env(args: argparse.Namespace, api_key: str) -> dict[str, str]:
    return {
        "OPENAI_BASE_URL": args.base_url,
        "OPENAI_API_KEY": api_key,
    }


def evaluate_command(args: argparse.Namespace) -> list[str]:
    command = ["uv", "run", "python", "-u", "-m", "llm.leverage.evaluate"]
    for task in args.tasks:
        command.extend(["--tasks", str(task)])
    command.extend(
        [
            "--predictions",
            str(args.output),
            "--output",
            str(args.scores_output),
            "--summary-output",
            str(args.summary_output),
        ]
    )
    return command


def display_command(command: list[str], *, env_overrides: dict[str, str] | None) -> list[str]:
    if not env_overrides:
        return command
    env_parts = [f"{name}={value}" for name, value in sorted(env_overrides.items())]
    return ["env", *env_parts, *command]


def run_command(
    command: list[str],
    *,
    cwd: Path,
    secrets: list[str],
    dry_run: bool,
    env_overrides: dict[str, str] | None = None,
) -> None:
    printable = display_command(command, env_overrides=env_overrides)
    print(f"$ {redact_command(printable, secrets)}")
    if dry_run:
        return
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    result = subprocess.run(command, cwd=cwd, check=False, env=env)
    if result.returncode != 0:
        redacted = redact_command(printable, secrets)
        raise RuntimeError(f"command failed with exit code {result.returncode}: {redacted}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key")
    parser.add_argument("--secret-path", type=Path, default=DEFAULT_SECRET_PATH)
    parser.add_argument("--tasks", type=Path, action="append")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-label", default=DEFAULT_MODEL_LABEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scores-output", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--thinking-mode", choices=("disabled", "none"), default="none")
    parser.add_argument(
        "--reasoning-effort",
        choices=("provider_default", "none", "minimal", "low", "medium", "high", "xhigh"),
        default="none",
    )
    parser.add_argument("--exclude-reasoning", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.repo_root = args.repo_root.resolve()
    if args.tasks is None:
        args.tasks = DEFAULT_TASKS.copy()
    return args


def preflight(args: argparse.Namespace) -> None:
    if not args.base_url:
        raise ValueError("--base-url must not be empty")
    if not args.model:
        raise ValueError("--model must not be empty")
    if not args.model_label:
        raise ValueError("--model-label must not be empty")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    if args.temperature < 0:
        raise ValueError("--temperature must be non-negative")
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive")
    for task in args.tasks:
        if not (args.repo_root / task).exists():
            raise FileNotFoundError(f"task file does not exist: {task}")


def api_key_for_run(args: argparse.Namespace) -> str:
    if args.api_key:
        return args.api_key
    if args.dry_run:
        return "dry-run-api-key"
    return load_api_key(args.secret_path)


def main() -> int:
    args = normalize_args(parse_args())
    preflight(args)
    api_key = api_key_for_run(args)
    secrets = [api_key]
    run_command(
        collect_command(args, api_key),
        cwd=args.repo_root,
        secrets=secrets,
        dry_run=args.dry_run,
        env_overrides=collect_env(args, api_key),
    )
    run_command(evaluate_command(args), cwd=args.repo_root, secrets=secrets, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
