#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import shlex
import subprocess


DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_JUDGE_MODEL = "qwen/qwen3.5-flash-02-23"
DEFAULT_JUDGE_LABEL = "qwen3-5-flash-openrouter"
DEFAULT_SECRET_PATH = Path.home() / ".secrets" / "openrouter"
DEFAULT_INPUT = Path("tracks/leverage/runs/instruction-outputs/qwen3-5-flash-openrouter-candidates.jsonl")
DEFAULT_OUTPUT = Path("tracks/leverage/runs/instruction-outputs/qwen3-5-flash-openrouter-judgments.jsonl")
DEFAULT_CSV_OUTPUT = Path("tracks/leverage/runs/instruction-outputs/qwen3-5-flash-openrouter-judgments.csv")
DEFAULT_SUMMARY_OUTPUT = Path("tracks/leverage/runs/instruction-outputs/qwen3-5-flash-openrouter-judgments-summary.csv")


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


def judge_command(args: argparse.Namespace) -> list[str]:
    command = [
        "uv",
        "run",
        "python",
        "-u",
        "-m",
        "llm.leverage.judge_instruction_outputs",
        "--input",
        str(args.input),
        "--output",
        str(args.output),
        "--csv-output",
        str(args.csv_output),
        "--summary-output",
        str(args.summary_output),
        "--judge-model",
        args.judge_model,
        "--judge-label",
        args.judge_label,
        "--max-tokens",
        str(args.max_tokens),
        "--temperature",
        str(args.temperature),
        "--reasoning-effort",
        args.reasoning_effort,
        "--timeout-seconds",
        str(args.timeout_seconds),
    ]
    if args.resume:
        command.append("--resume")
    else:
        command.append("--overwrite")
    if args.exclude_reasoning:
        command.append("--exclude-reasoning")
    else:
        command.append("--no-exclude-reasoning")
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    return command


def judge_env(args: argparse.Namespace, api_key: str) -> dict[str, str]:
    return {
        "OPENAI_BASE_URL": args.base_url,
        "OPENAI_API_KEY": api_key,
    }


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
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--judge-label", default=DEFAULT_JUDGE_LABEL)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--reasoning-effort",
        choices=("provider_default", "none", "minimal", "low", "medium", "high", "xhigh"),
        default="none",
    )
    parser.add_argument("--exclude-reasoning", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.repo_root = args.repo_root.resolve()
    return args


def preflight(args: argparse.Namespace) -> None:
    if not args.base_url:
        raise ValueError("--base-url must not be empty")
    if not args.judge_model:
        raise ValueError("--judge-model must not be empty")
    if not args.judge_label:
        raise ValueError("--judge-label must not be empty")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    if args.temperature < 0:
        raise ValueError("--temperature must be non-negative")
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if not args.dry_run and not (args.repo_root / args.input).exists():
        raise FileNotFoundError(f"input file does not exist: {args.input}")


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
    run_command(
        judge_command(args),
        cwd=args.repo_root,
        secrets=[api_key],
        dry_run=args.dry_run,
        env_overrides=judge_env(args, api_key),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
