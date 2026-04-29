#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
from pathlib import Path
import shlex
import sys
import tomllib
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import runpod_common
from runpod_common import (
    DEFAULT_REMOTE_DIR,
    DEFAULT_RUNPODCTL,
    DEFAULT_SECRET_PATH,
    DEFAULT_SSH_KEY,
    DEFAULT_SSH_PUBLIC_KEY,
    PodConnection,
    Runner,
    assert_no_existing_pods,
    cleanup_pod,
    deadline_from_minutes,
    ensure_before_deadline,
    find_created_pod,
    list_pods,
    load_api_key,
    q,
    remote_transport_setup_command,
    run_with_deadline,
    runpodctl_create_command,
    rsync_ssh,
    ssh_command,
    timestamped_name,
    wait_for_connection,
    wait_for_ssh,
    with_remote_dir,
)

time = runpod_common.time


DEFAULT_IMAGE = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"
SYNC_DIRS = ("src", "tests", "tracks/from-scratch/configs", "tracks/from-scratch/evals")
SYNC_FILES = ("pyproject.toml", "uv.lock", "README.md", "AGENTS.md", "LICENSE")


@dataclass(frozen=True)
class OutputPaths:
    checkpoint: str
    metrics: str
    observation: str
    summary: str
    tokens: str


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as config_file:
        data = tomllib.load(config_file)
    if not isinstance(data, dict):
        raise ValueError("config must be a TOML table")
    return data


def nested_str(config: dict[str, Any], section: str, key: str) -> str:
    section_data = config.get(section, {})
    if not isinstance(section_data, dict):
        raise ValueError(f"config section [{section}] must be a table")
    value = section_data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"config must contain [{section}] {key}")
    return value


def output_paths(config_path: Path) -> OutputPaths:
    config = load_toml(config_path)
    return OutputPaths(
        checkpoint=nested_str(config, "outputs", "checkpoint"),
        metrics=nested_str(config, "outputs", "metrics"),
        observation=nested_str(config, "outputs", "observation"),
        summary=nested_str(config, "outputs", "summary"),
        tokens=nested_str(config, "data", "tokens"),
    )


def rsync_to_remote_command(args: argparse.Namespace, connection: PodConnection) -> list[str]:
    sources = [
        *SYNC_DIRS,
        *SYNC_FILES,
        output_paths(args.repo_root / args.config).tokens,
        *resume_paths(args.train_extra_args),
    ]
    command = ["rsync", "-az", "--relative", "-e", rsync_ssh(args, connection)]
    command.extend(str(source) for source in sources if (args.repo_root / source).exists())
    command.append(f"{connection.user}@{connection.host}:{args.remote_dir}/")
    return command


def rsync_from_remote_command(args: argparse.Namespace, connection: PodConnection) -> list[str]:
    return [
        "rsync",
        "-az",
        "-e",
        rsync_ssh(args, connection),
        f"{connection.user}@{connection.host}:{args.remote_dir}/tracks/from-scratch/checkpoints",
        f"{connection.user}@{connection.host}:{args.remote_dir}/tracks/from-scratch/runs",
        f"{args.repo_root}/",
    ]


def remote_setup_command() -> str:
    return (
        "set -euo pipefail; "
        "cd \"$REMOTE_DIR\"; "
        "if ! command -v uv >/dev/null 2>&1; then "
        "curl -LsSf https://astral.sh/uv/install.sh | sh; "
        "fi; "
        "export PATH=\"$HOME/.local/bin:$PATH\"; "
        "UV_LINK_MODE=copy uv sync --extra dev"
    )


def remote_cuda_preflight_command() -> str:
    return (
        "set -euo pipefail; "
        "cd \"$REMOTE_DIR\"; "
        "export PATH=\"$HOME/.local/bin:$PATH\"; "
        "uv run python -u -c "
        + q(
            "import torch; "
            "print(f'python torch: {torch.__version__}'); "
            "print(f'cuda available: {torch.cuda.is_available()}'); "
            "print(f'cuda device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}'); "
            "raise SystemExit(0 if torch.cuda.is_available() else 1)"
        )
    )


def quote_extra_args(extra_args: str) -> str:
    return " ".join(q(part) for part in shlex.split(extra_args))


def resume_paths(extra_args: str) -> list[str]:
    parts = shlex.split(extra_args)
    paths: list[str] = []
    for index, part in enumerate(parts):
        if part == "--resume" and index + 1 < len(parts):
            paths.append(parts[index + 1])
        elif part.startswith("--resume="):
            paths.append(part.split("=", 1)[1])
    return paths


def ensure_device_cuda(extra_args: str) -> str:
    parts = shlex.split(extra_args)
    has_device = any(part == "--device" or part.startswith("--device=") for part in parts)
    if has_device:
        return quote_extra_args(extra_args)
    return " ".join(part for part in (quote_extra_args(extra_args), "--device cuda") if part)


def remote_train_command(config: Path, extra_args: str) -> str:
    command = (
        "set -euo pipefail; cd \"$REMOTE_DIR\"; "
        "export PATH=\"$HOME/.local/bin:$PATH\"; "
        f"uv run python -u -m llm.train --config {q(config)}"
    )
    command = f"{command} {ensure_device_cuda(extra_args)}"
    return command


def remote_observe_command(config: Path, extra_args: str = "") -> str:
    command = (
        "set -euo pipefail; "
        "cd \"$REMOTE_DIR\"; "
        "export PATH=\"$HOME/.local/bin:$PATH\"; "
        f"uv run python -u -m llm.observe --config {q(config)}"
    )
    command = f"{command} {ensure_device_cuda(extra_args)}"
    return command


def remote_scaling_command(config: Path, outputs: OutputPaths) -> str:
    return (
        "set -euo pipefail; "
        "cd \"$REMOTE_DIR\"; "
        "export PATH=\"$HOME/.local/bin:$PATH\"; "
        "uv run python -u -m llm.scaling "
        f"--checkpoint {q(outputs.checkpoint)} "
        f"--summary {q(outputs.summary)} "
        "--output tracks/from-scratch/runs/summaries/scaling.csv"
    )


def require_existing_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"required {label} does not exist: {path}")


def preflight(args: argparse.Namespace, outputs: OutputPaths) -> None:
    require_existing_path(args.repo_root / args.config, "config")
    require_existing_path(args.repo_root / outputs.tokens, "token file")
    require_existing_path(args.repo_root / "src", "source directory")
    require_existing_path(args.repo_root / "pyproject.toml", "pyproject.toml")
    require_existing_path(args.repo_root / "uv.lock", "uv.lock")
    if not args.dry_run:
        require_existing_path(Path(args.runpodctl), "runpodctl")
        require_existing_path(args.ssh_key, "SSH private key")
    if args.bootstrap_sshd:
        require_existing_path(args.ssh_public_key, "SSH public key")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-extra-args", default="")
    parser.add_argument("--observe-extra-args", default="")
    parser.add_argument("--gpu-type", default="NVIDIA GeForce RTX 4090")
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--max-cost", type=float, default=0.40)
    parser.add_argument("--pod-name-prefix", default="llm-train-once")
    parser.add_argument("--pod-name")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--container-disk-size", type=int, default=40)
    parser.add_argument("--volume-size", type=int, default=20)
    parser.add_argument("--remote-volume", default="/workspace")
    parser.add_argument("--remote-dir", default=DEFAULT_REMOTE_DIR)
    parser.add_argument("--vcpu", type=int, default=8)
    parser.add_argument("--mem", type=int, default=20)
    parser.add_argument("--secure-cloud", action="store_true")
    parser.add_argument("--runpodctl", default=DEFAULT_RUNPODCTL)
    parser.add_argument("--secret-path", type=Path, default=DEFAULT_SECRET_PATH)
    parser.add_argument("--ssh-key", type=Path, default=DEFAULT_SSH_KEY)
    parser.add_argument("--ssh-public-key", type=Path, default=DEFAULT_SSH_PUBLIC_KEY)
    parser.add_argument("--bootstrap-sshd", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--wait-seconds", type=int, default=900)
    parser.add_argument("--ssh-wait-seconds", type=int, default=180)
    parser.add_argument("--max-runtime-minutes", type=int, default=60)
    parser.add_argument("--allow-existing-pods", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-pod", action="store_true")
    parser.add_argument("--keep-pod-on-failure", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.repo_root = args.repo_root.resolve()
    if args.pod_name is None:
        args.pod_name = timestamped_name(args.pod_name_prefix)
    outputs = output_paths(args.repo_root / args.config)
    preflight(args, outputs)
    api_key = "" if args.dry_run else load_api_key(args.secret_path)
    public_key = ""
    if args.bootstrap_sshd and args.ssh_public_key.exists():
        public_key = args.ssh_public_key.read_text(encoding="utf-8").strip()
    runner = Runner(
        dry_run=args.dry_run,
        secrets=[api_key, public_key],
        env={"RUNPOD_API_KEY": api_key} if api_key else None,
    )
    deadline = deadline_from_minutes(args.max_runtime_minutes)
    pod_id: str | None = None
    success = False
    pods_before_create: list[dict[str, str]] = []
    try:
        if not args.dry_run and not args.allow_existing_pods:
            assert_no_existing_pods(runner, args)
        ensure_before_deadline(deadline)
        if not args.dry_run:
            pods_before_create = list_pods(runner, args)
        run_with_deadline(runner, runpodctl_create_command(args), cwd=args.repo_root, deadline=deadline)
        if args.dry_run:
            connection = PodConnection(host="dry-run.runpod.local", port=22)
            pod_id = "dry-run-pod"
        else:
            pods_after_create = list_pods(runner, args)
            pod = find_created_pod(pods_before_create, pods_after_create, args.pod_name)
            pod_id = str(pod["ID"])
            print(f"created pod: {pod_id}")
            connection = wait_for_connection(runner, args, pod_id, args.wait_seconds)
            print(f"ssh: {connection.user}@{connection.host}:{connection.port}")

        ensure_before_deadline(deadline)
        wait_for_ssh(runner, args, connection)
        ensure_before_deadline(deadline)
        run_with_deadline(runner, ssh_command(args, connection, remote_transport_setup_command()), deadline=deadline)
        ensure_before_deadline(deadline)
        run_with_deadline(runner, ssh_command(args, connection, f"mkdir -p {q(args.remote_dir)}"), deadline=deadline)
        ensure_before_deadline(deadline)
        run_with_deadline(runner, rsync_to_remote_command(args, connection), cwd=args.repo_root, deadline=deadline)
        ensure_before_deadline(deadline)
        run_with_deadline(runner, ssh_command(args, connection, with_remote_dir(args, remote_setup_command())), deadline=deadline)
        ensure_before_deadline(deadline)
        run_with_deadline(
            runner,
            ssh_command(args, connection, with_remote_dir(args, remote_cuda_preflight_command())),
            deadline=deadline,
        )
        ensure_before_deadline(deadline)
        run_with_deadline(
            runner,
            ssh_command(
                args,
                connection,
                with_remote_dir(args, remote_train_command(args.config, args.train_extra_args)),
            ),
            deadline=deadline,
        )
        ensure_before_deadline(deadline)
        run_with_deadline(
            runner,
            ssh_command(
                args,
                connection,
                with_remote_dir(args, remote_observe_command(args.config, args.observe_extra_args)),
            ),
            deadline=deadline,
        )
        ensure_before_deadline(deadline)
        run_with_deadline(
            runner,
            ssh_command(args, connection, with_remote_dir(args, remote_scaling_command(args.config, outputs))),
            deadline=deadline,
        )
        ensure_before_deadline(deadline)
        run_with_deadline(runner, rsync_from_remote_command(args, connection), cwd=args.repo_root, deadline=deadline)
        success = True
        return 0
    finally:
        cleanup_pod(runner, args, pod_id, success=success)


if __name__ == "__main__":
    raise SystemExit(main())
