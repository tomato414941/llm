#!/usr/bin/env python3
import argparse
from pathlib import Path
import shlex
import subprocess
import sys


DEFAULT_TEMPLATE_ID = "runpod-torch-v280"
DEFAULT_IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
DEFAULT_SYNC = ("src", "tests", "pyproject.toml", "uv.lock", "README.md", "AGENTS.md", "LICENSE")
DEFAULT_REMOTE_DIR = "/workspace/llm"
DEFAULT_RUNPODCTL = "/home/dev/bin/runpodctl"
DEFAULT_SECRET_PATH = Path.home() / ".secrets" / "runpod"
DEFAULT_SSH_KEY = Path.home() / ".runpod" / "ssh" / "RunPod-Key-Go"
DEFAULT_SSH_PUBLIC_KEY = Path.home() / ".runpod" / "ssh" / "RunPod-Key-Go.pub"
DEFAULT_SHARED_RUNNER = Path.home() / "projects" / "runpod-job-runner" / "scripts" / "run_job.py"
DEFAULT_SETUP_COMMAND = (
    "set -euo pipefail; "
    "cd \"$REMOTE_DIR\"; "
    "if ! command -v uv >/dev/null 2>&1; then "
    "curl -LsSf https://astral.sh/uv/install.sh | sh; "
    "fi; "
    "export PATH=\"$HOME/.local/bin:$PATH\"; "
    "rm -rf .venv; "
    "UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/tmp/llm-runpod-venv uv sync --extra dev; "
    "ln -sfn /tmp/llm-runpod-venv .venv"
)


def shell_command(command: str) -> list[str]:
    if not command.strip():
        raise ValueError("command must not be empty")
    return ["bash", "-lc", command]


def remote_cuda_smoke_command() -> str:
    return (
        "set -euo pipefail; "
        "export PATH=\"$HOME/.local/bin:$PATH\"; "
        "uv run python -u -c "
        + shlex.quote(
            "import torch; "
            "print(f'torch={torch.__version__}'); "
            "print(f'cuda_available={torch.cuda.is_available()}'); "
            "print(f'cuda_device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}'); "
            "raise SystemExit(0 if torch.cuda.is_available() else 1)"
        )
    )


def remote_user_command(command: str) -> str:
    return f"set -euo pipefail; export PATH=\"$HOME/.local/bin:$PATH\"; {command}"


def preflight(args: argparse.Namespace) -> None:
    if not args.shared_runner.exists():
        raise FileNotFoundError(f"shared RunPod runner does not exist: {args.shared_runner}")
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


def append_repeated(command: list[str], flag: str, values: list[str | Path]) -> None:
    for value in values:
        command.extend([flag, str(value)])


def default_timings_output(args: argparse.Namespace) -> Path | None:
    if args.timings_output is not None:
        return args.timings_output
    if not args.output:
        return None
    return Path(args.output[0]) / "runpod-timings.json"


def build_shared_runner_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(args.shared_runner),
        "--repo-root",
        str(args.repo_root),
        "--name",
        args.name,
        "--runpodctl",
        args.runpodctl,
        "--secret-path",
        str(args.secret_path),
        "--ssh-key",
        str(args.ssh_key),
        "--ssh-public-key",
        str(args.ssh_public_key),
        "--gpu-type",
        args.gpu_type,
        "--gpu-count",
        str(args.gpu_count),
        "--container-disk-size",
        str(args.container_disk_size),
        "--volume-size",
        str(args.volume_size),
        "--remote-volume",
        args.remote_volume,
        "--remote-dir",
        args.remote_dir,
        "--wait-seconds",
        str(args.wait_seconds),
        "--ssh-wait-seconds",
        str(args.ssh_wait_seconds),
        "--max-runtime-minutes",
        str(args.max_runtime_minutes),
        "--setup-command",
        "true" if args.no_setup else args.setup_command,
    ]
    if args.pod_name:
        command.extend(["--pod-name", args.pod_name])
    if args.secure_cloud:
        command.append("--secure-cloud")
    if args.data_center_ids:
        command.extend(["--data-center-ids", args.data_center_ids])
    if args.allow_existing_pods:
        command.append("--allow-existing-pods")
    if args.dry_run:
        command.append("--dry-run")
    if args.keep_pod:
        command.append("--keep-pod")
    if args.keep_pod_on_failure:
        command.append("--keep-pod-on-failure")
    if args.template_id:
        command.extend(["--template-id", args.template_id])
    else:
        command.extend(["--image", args.image])
    append_repeated(command, "--allowed-cuda-version", args.allowed_cuda_version)
    append_repeated(command, "--local", args.local)
    for source in [*DEFAULT_SYNC, *args.sync]:
        command.extend(["--sync", source])
    if args.cuda_smoke:
        command.extend(["--remote", remote_cuda_smoke_command()])
    append_repeated(command, "--remote", [remote_user_command(remote) for remote in args.remote])
    append_repeated(command, "--output", args.output)
    timings_output = default_timings_output(args)
    if timings_output is not None:
        command.extend(["--timings-output", str(timings_output)])
    return command


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
    parser.add_argument("--template-id", default=DEFAULT_TEMPLATE_ID)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--allowed-cuda-version", action="append", default=[])
    parser.add_argument("--container-disk-size", type=int, default=80)
    parser.add_argument("--volume-size", type=int, default=80)
    parser.add_argument("--remote-volume", default="/workspace")
    parser.add_argument("--remote-dir", default=DEFAULT_REMOTE_DIR)
    parser.add_argument("--vcpu", type=int, default=8)
    parser.add_argument("--mem", type=int, default=32)
    parser.add_argument("--secure-cloud", action="store_true")
    parser.add_argument("--data-center-ids", default="")
    parser.add_argument("--runpodctl", default=DEFAULT_RUNPODCTL)
    parser.add_argument("--secret-path", type=Path, default=DEFAULT_SECRET_PATH)
    parser.add_argument("--ssh-key", type=Path, default=DEFAULT_SSH_KEY)
    parser.add_argument("--ssh-public-key", type=Path, default=DEFAULT_SSH_PUBLIC_KEY)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--wait-seconds", type=int, default=900)
    parser.add_argument("--ssh-wait-seconds", type=int, default=180)
    parser.add_argument("--max-runtime-minutes", type=int, default=30)
    parser.add_argument("--allow-existing-pods", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-pod", action="store_true")
    parser.add_argument("--keep-pod-on-failure", action="store_true")
    parser.add_argument("--timings-output", type=Path)
    parser.add_argument("--shared-runner", type=Path, default=DEFAULT_SHARED_RUNNER)
    cli_args = sys.argv[1:]
    image_was_set = any(arg == "--image" or arg.startswith("--image=") for arg in cli_args)
    template_was_set = any(arg == "--template-id" or arg.startswith("--template-id=") for arg in cli_args)
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()
    if image_was_set and not template_was_set:
        args.template_id = None
    return args


def main() -> int:
    args = parse_args()
    preflight(args)
    return subprocess.run(build_shared_runner_command(args), cwd=args.repo_root, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
