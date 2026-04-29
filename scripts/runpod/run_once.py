#!/usr/bin/env python3
import argparse
from pathlib import Path
import shlex
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

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
    normalize_pods,
    parse_ssh_info,
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


DEFAULT_IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
DEFAULT_SYNC = ("src", "tests", "pyproject.toml", "uv.lock", "README.md", "AGENTS.md", "LICENSE")
DEFAULT_SETUP_COMMAND = (
    "set -euo pipefail; "
    "cd \"$REMOTE_DIR\"; "
    "if ! command -v uv >/dev/null 2>&1; then "
    "curl -LsSf https://astral.sh/uv/install.sh | sh; "
    "fi; "
    "export PATH=\"$HOME/.local/bin:$PATH\"; "
    "UV_LINK_MODE=copy uv sync --extra dev"
)


def split_shell_command(command: str) -> list[str]:
    parts = shlex.split(command)
    if not parts:
        raise ValueError("command must not be empty")
    return parts


def rsync_to_remote_command(args: argparse.Namespace, connection: PodConnection) -> list[str]:
    sources = [*DEFAULT_SYNC, *args.sync]
    seen: set[str] = set()
    command = ["rsync", "-az", "--relative", "-e", rsync_ssh(args, connection)]
    for source in sources:
        if source in seen:
            continue
        seen.add(source)
        if (args.repo_root / source).exists():
            command.append(source)
    command.append(f"{connection.user}@{connection.host}:{args.remote_dir}/")
    return command


def rsync_from_remote_command(args: argparse.Namespace, connection: PodConnection) -> list[str]:
    command = ["rsync", "-az", "-e", rsync_ssh(args, connection)]
    for output in args.output:
        command.append(f"{connection.user}@{connection.host}:{args.remote_dir}/{output}")
    command.append(f"{args.repo_root}/")
    return command


def remote_cuda_smoke_command() -> str:
    return (
        "set -euo pipefail; "
        "cd \"$REMOTE_DIR\"; "
        "export PATH=\"$HOME/.local/bin:$PATH\"; "
        "uv run python -u -c "
        + q(
            "import torch; "
            "print(f'torch={torch.__version__}'); "
            "print(f'cuda_available={torch.cuda.is_available()}'); "
            "print(f'cuda_device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}'); "
            "raise SystemExit(0 if torch.cuda.is_available() else 1)"
        )
    )


def remote_user_command(command: str) -> str:
    return (
        "set -euo pipefail; "
        "cd \"$REMOTE_DIR\"; "
        "export PATH=\"$HOME/.local/bin:$PATH\"; "
        f"{command}"
    )


def preflight(args: argparse.Namespace) -> None:
    for source in ("src", "pyproject.toml", "uv.lock"):
        if not (args.repo_root / source).exists():
            raise FileNotFoundError(f"required source does not exist: {source}")
    for source in args.sync:
        if not (args.repo_root / source).exists():
            raise FileNotFoundError(f"sync source does not exist: {source}")
    if not args.remote:
        raise ValueError("at least one --remote command is required")
    if not args.output:
        raise ValueError("at least one --output path is required")
    if not args.dry_run:
        if not Path(args.runpodctl).exists():
            raise FileNotFoundError(f"runpodctl does not exist: {args.runpodctl}")
        if not args.ssh_key.exists():
            raise FileNotFoundError(f"SSH private key does not exist: {args.ssh_key}")
    if args.bootstrap_sshd and not args.ssh_public_key.exists():
        raise FileNotFoundError(f"SSH public key does not exist: {args.ssh_public_key}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="runpod-once")
    parser.add_argument("--pod-name")
    parser.add_argument("--sync", action="append", default=[])
    parser.add_argument("--output", action="append", default=[])
    parser.add_argument("--local", action="append", default=[])
    parser.add_argument("--remote", action="append", default=[])
    parser.add_argument("--setup-command", default=DEFAULT_SETUP_COMMAND)
    parser.add_argument("--no-setup", action="store_true")
    parser.add_argument("--cuda-smoke", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu-type", default="NVIDIA GeForce RTX 4090")
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--max-cost", type=float, default=1.0)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--container-disk-size", type=int, default=80)
    parser.add_argument("--volume-size", type=int, default=80)
    parser.add_argument("--remote-volume", default="/workspace")
    parser.add_argument("--remote-dir", default=DEFAULT_REMOTE_DIR)
    parser.add_argument("--vcpu", type=int, default=8)
    parser.add_argument("--mem", type=int, default=32)
    parser.add_argument("--secure-cloud", action="store_true")
    parser.add_argument("--runpodctl", default=DEFAULT_RUNPODCTL)
    parser.add_argument("--secret-path", type=Path, default=DEFAULT_SECRET_PATH)
    parser.add_argument("--ssh-key", type=Path, default=DEFAULT_SSH_KEY)
    parser.add_argument("--ssh-public-key", type=Path, default=DEFAULT_SSH_PUBLIC_KEY)
    parser.add_argument("--bootstrap-sshd", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--wait-seconds", type=int, default=900)
    parser.add_argument("--ssh-wait-seconds", type=int, default=180)
    parser.add_argument("--max-runtime-minutes", type=int, default=30)
    parser.add_argument("--allow-existing-pods", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-pod", action="store_true")
    parser.add_argument("--keep-pod-on-failure", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.repo_root = args.repo_root.resolve()
    args.pod_name_prefix = args.name
    if args.pod_name is None:
        args.pod_name = timestamped_name(args.name)
    preflight(args)

    for command in args.local:
        Runner(dry_run=args.dry_run, secrets=[]).run(split_shell_command(command), cwd=args.repo_root)

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

        wait_for_ssh(runner, args, connection)
        steps = [
            ssh_command(args, connection, remote_transport_setup_command()),
            ssh_command(args, connection, f"mkdir -p {q(args.remote_dir)}"),
            rsync_to_remote_command(args, connection),
        ]
        if not args.no_setup:
            steps.append(ssh_command(args, connection, with_remote_dir(args, args.setup_command)))
        if args.cuda_smoke:
            steps.append(ssh_command(args, connection, with_remote_dir(args, remote_cuda_smoke_command())))
        steps.extend(
            ssh_command(args, connection, with_remote_dir(args, remote_user_command(command)))
            for command in args.remote
        )
        steps.append(rsync_from_remote_command(args, connection))

        for step in steps:
            ensure_before_deadline(deadline)
            run_with_deadline(runner, step, cwd=args.repo_root, deadline=deadline)
        success = True
        return 0
    finally:
        cleanup_pod(runner, args, pod_id, success=success)


if __name__ == "__main__":
    raise SystemExit(main())
