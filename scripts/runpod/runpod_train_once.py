#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import tomllib
from typing import Any


DEFAULT_IMAGE = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"
DEFAULT_RUNPODCTL = "/home/dev/bin/runpodctl"
DEFAULT_SECRET_PATH = Path.home() / ".secrets" / "runpod"
DEFAULT_SSH_KEY = Path.home() / ".runpod" / "ssh" / "RunPod-Key-Go"
DEFAULT_SSH_PUBLIC_KEY = Path.home() / ".runpod" / "ssh" / "RunPod-Key-Go.pub"
DEFAULT_REMOTE_DIR = "/workspace/llm"
SYNC_DIRS = ("src", "tests", "configs", "eval_prompts")
SYNC_FILES = ("pyproject.toml", "uv.lock", "README.md", "AGENTS.md", "LICENSE")


@dataclass(frozen=True)
class OutputPaths:
    checkpoint: str
    metrics: str
    observation: str
    summary: str
    tokens: str


@dataclass(frozen=True)
class PodConnection:
    host: str
    port: int
    user: str = "root"


class CommandError(RuntimeError):
    pass


def load_api_key(path: Path) -> str:
    key = path.read_text(encoding="utf-8").strip()
    if not key:
        raise ValueError(f"RunPod API key file is empty: {path}")
    return key


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


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def redact(text: str, secrets: list[str]) -> str:
    redacted = text
    for secret in secrets:
        if secret:
            redacted = redacted.replace(secret, "[REDACTED]")
    return redacted


def q(value: str | Path) -> str:
    return shlex.quote(str(value))


class Runner:
    def __init__(self, dry_run: bool, secrets: list[str], env: dict[str, str] | None = None) -> None:
        self.dry_run = dry_run
        self.secrets = secrets
        self.env = env or {}

    def run(
        self,
        command: list[str],
        *,
        cwd: Path | None = None,
        capture: bool | None = None,
        check: bool = True,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        print(f"$ {redact(shell_join(command), self.secrets)}")
        if self.dry_run:
            return subprocess.CompletedProcess(command, 0, "", "")
        capture_output = Path(command[0]).name == "runpodctl" if capture is None else capture
        env = os.environ | self.env
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            text=True,
            capture_output=capture_output,
            env=env,
            timeout=timeout,
        )
        if capture_output and completed.stdout:
            print(redact(completed.stdout, self.secrets), end="")
        if capture_output and completed.stderr:
            print(redact(completed.stderr, self.secrets), end="", file=sys.stderr)
        if check and completed.returncode != 0:
            raise CommandError(f"command failed with exit code {completed.returncode}")
        return completed


def runpodctl_create_command(args: argparse.Namespace) -> list[str]:
    command = [
        args.runpodctl,
        "create",
        "pod",
        "--name",
        args.pod_name,
        "--gpuType",
        args.gpu_type,
        "--gpuCount",
        str(args.gpu_count),
        "--imageName",
        args.image,
        "--containerDiskSize",
        str(args.container_disk_size),
        "--volumeSize",
        str(args.volume_size),
        "--volumePath",
        args.remote_volume,
        "--ports",
        "22/tcp",
        "--cost",
        str(args.max_cost),
    ]
    if args.vcpu:
        command.extend(["--vcpu", str(args.vcpu)])
    if args.mem:
        command.extend(["--mem", str(args.mem)])
    if args.secure_cloud:
        command.append("--secureCloud")
    else:
        command.append("--communityCloud")
    if args.bootstrap_sshd:
        public_key = args.ssh_public_key.read_text(encoding="utf-8").strip()
        command.extend(["--env", f"PUBLIC_KEY={public_key}"])
        command.extend(
            [
                "--args",
                (
                    "bash -c 'apt-get update && "
                    "DEBIAN_FRONTEND=noninteractive apt-get install -y openssh-server rsync && "
                    "mkdir -p ~/.ssh && chmod 700 ~/.ssh && "
                    "echo \"$PUBLIC_KEY\" >> ~/.ssh/authorized_keys && "
                    "chmod 600 ~/.ssh/authorized_keys && "
                    "service ssh start && sleep infinity'"
                ),
            ]
        )
    return command


def parse_pod_table(output: str) -> list[dict[str, str]]:
    lines = [line for line in output.splitlines() if line.strip()]
    if len(lines) <= 1:
        return []
    headers = [header.strip() for header in lines[0].split("\t")]
    pods: list[dict[str, str]] = []
    for line in lines[1:]:
        values = [value.strip() for value in line.split("\t")]
        row = {header: values[index] if index < len(values) else "" for index, header in enumerate(headers)}
        if row.get("ID"):
            pods.append(row)
    return pods


def list_pods(runner: Runner, args: argparse.Namespace) -> list[dict[str, str]]:
    completed = runner.run([args.runpodctl, "get", "pod", "--allfields"], capture=True)
    return parse_pod_table(completed.stdout)


def find_pod_by_name(runner: Runner, args: argparse.Namespace, name: str) -> dict[str, str]:
    matches = [pod for pod in list_pods(runner, args) if pod.get("NAME") == name]
    if not matches:
        raise RuntimeError(f"pod was not found by name: {name}")
    return matches[-1]


def active_pods(runner: Runner, args: argparse.Namespace) -> list[dict[str, str]]:
    inactive_statuses = {"EXITED", "TERMINATED", "STOPPED"}
    return [
        pod
        for pod in list_pods(runner, args)
        if str(pod.get("STATUS", "")).upper() not in inactive_statuses
    ]


def assert_no_existing_pods(runner: Runner, args: argparse.Namespace) -> None:
    pods = active_pods(runner, args)
    if pods:
        pod_descriptions = ", ".join(
            f"{pod.get('NAME', '<unnamed>')}:{pod.get('ID', '<unknown>')}" for pod in pods
        )
        raise RuntimeError(f"RunPod account already has active pods: {pod_descriptions}")


def pod_ids(pods: list[dict[str, str]]) -> set[str]:
    return {str(pod["ID"]) for pod in pods if pod.get("ID")}


def find_created_pod(before: list[dict[str, str]], after: list[dict[str, str]], name: str) -> dict[str, str]:
    before_ids = pod_ids(before)
    created = [pod for pod in after if pod.get("ID") not in before_ids and pod.get("NAME") == name]
    if len(created) == 1:
        return created[0]
    if not created:
        raise RuntimeError(f"created pod was not found by name: {name}")
    raise RuntimeError(f"multiple created pods matched name: {name}")


def timestamped_name(prefix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{stamp}"


def wait_for_connection(
    runner: Runner,
    args: argparse.Namespace,
    pod_id: str,
    timeout_seconds: int,
) -> PodConnection:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        pod = next((item for item in list_pods(runner, args) if item.get("ID") == pod_id), None)
        if pod is not None:
            connection = pod_connection(pod)
            if connection is not None:
                return connection
        time.sleep(10)
    raise TimeoutError(f"pod did not expose SSH within {timeout_seconds} seconds: {pod_id}")


def pod_connection(pod: dict[str, str]) -> PodConnection | None:
    ports = pod.get("PORTS", "")
    if not ports:
        return None
    pattern = re.compile(r"([A-Za-z0-9.-]+):(\d+)->22\s*\(([^)]*)\)")
    for host, public_port, labels in pattern.findall(ports):
        if "tcp" in labels.lower() and "prv" not in labels.lower():
            return PodConnection(host=host, port=int(public_port))
    return None


def ssh_base(args: argparse.Namespace, connection: PodConnection) -> list[str]:
    return [
        "ssh",
        "-i",
        str(args.ssh_key),
        "-p",
        str(connection.port),
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ConnectionAttempts=1",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=2",
        f"{connection.user}@{connection.host}",
    ]


def rsync_ssh(args: argparse.Namespace, connection: PodConnection) -> str:
    return shell_join(
        [
            "ssh",
            "-i",
            str(args.ssh_key),
            "-p",
            str(connection.port),
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "ConnectionAttempts=1",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=2",
        ]
    )


def ssh_command(args: argparse.Namespace, connection: PodConnection, remote_command: str) -> list[str]:
    return [*ssh_base(args, connection), remote_command]


def wait_for_ssh(runner: Runner, args: argparse.Namespace, connection: PodConnection) -> None:
    deadline = time.monotonic() + args.ssh_wait_seconds
    while time.monotonic() < deadline:
        completed = runner.run(
            ssh_command(args, connection, "true"),
            check=False,
            capture=True,
        )
        if completed.returncode == 0:
            return
        time.sleep(5)
    raise TimeoutError(f"SSH did not become ready within {args.ssh_wait_seconds} seconds")


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
        f"{connection.user}@{connection.host}:{args.remote_dir}/checkpoints",
        f"{connection.user}@{connection.host}:{args.remote_dir}/experiments",
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


def remote_transport_setup_command() -> str:
    return (
        "set -euo pipefail; "
        "if ! command -v rsync >/dev/null 2>&1; then "
        "apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y rsync; "
        "fi"
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
        "--output experiments/summaries/scaling.csv"
    )


def with_remote_dir(args: argparse.Namespace, command: str) -> str:
    return f"REMOTE_DIR={q(args.remote_dir)}; {command}"


def remaining_seconds(deadline: float | None) -> float | None:
    if deadline is None:
        return None
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("maximum runtime was reached")
    return remaining


def run_with_deadline(
    runner: Runner,
    command: list[str],
    *,
    deadline: float | None,
    cwd: Path | None = None,
    capture: bool | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return runner.run(
        command,
        cwd=cwd,
        capture=capture,
        check=check,
        timeout=remaining_seconds(deadline),
    )


def cleanup_pod(
    runner: Runner,
    args: argparse.Namespace,
    pod_id: str | None,
    *,
    success: bool,
) -> None:
    if pod_id is None:
        return
    keep_on_failure = getattr(args, "keep_pod_on_failure", False)
    if args.keep_pod or (not success and keep_on_failure):
        return
    runner.run([args.runpodctl, "remove", "pod", pod_id], check=False)
    if not runner.dry_run:
        remaining = [pod for pod in list_pods(runner, args) if pod.get("ID") == pod_id]
        if remaining:
            raise RuntimeError(f"RunPod pod still exists after cleanup: {pod_id}")


def deadline_from_minutes(minutes: int) -> float | None:
    if minutes <= 0:
        return None
    return time.monotonic() + minutes * 60


def ensure_before_deadline(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("maximum runtime was reached")


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
