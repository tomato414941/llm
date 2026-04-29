#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time


DEFAULT_RUNPODCTL = "/home/dev/bin/runpodctl"
DEFAULT_SECRET_PATH = Path.home() / ".secrets" / "runpod"
DEFAULT_SSH_KEY = Path.home() / ".runpod" / "ssh" / "RunPod-Key-Go"
DEFAULT_SSH_PUBLIC_KEY = Path.home() / ".runpod" / "ssh" / "RunPod-Key-Go.pub"
DEFAULT_REMOTE_DIR = "/workspace/llm"


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
        "pod",
        "create",
        "-o",
        "json",
        "--name",
        args.pod_name,
        "--gpu-id",
        args.gpu_type,
        "--gpu-count",
        str(args.gpu_count),
        "--image",
        args.image,
        "--container-disk-in-gb",
        str(args.container_disk_size),
        "--volume-in-gb",
        str(args.volume_size),
        "--volume-mount-path",
        args.remote_volume,
        "--ports",
        "22/tcp",
    ]
    if args.secure_cloud:
        command.extend(["--cloud-type", "SECURE"])
    else:
        command.extend(["--cloud-type", "COMMUNITY", "--public-ip"])
    if args.bootstrap_sshd:
        command.append("--ssh")
    return command


def parse_json_output(output: str) -> object:
    text = output.strip()
    if not text:
        return []
    return json.loads(text)


def normalize_pod(raw: dict[str, object]) -> dict[str, str]:
    ports = raw.get("ports") or raw.get("PORTS") or raw.get("portMappings") or ""
    if isinstance(ports, list):
        ports = ",".join(format_port_mapping(item) for item in ports)
    return {
        "ID": str(raw.get("id") or raw.get("ID") or ""),
        "NAME": str(raw.get("name") or raw.get("NAME") or ""),
        "STATUS": str(raw.get("desiredStatus") or raw.get("status") or raw.get("STATUS") or ""),
        "PORTS": str(ports),
    }


def format_port_mapping(raw: object) -> str:
    if not isinstance(raw, dict):
        return str(raw)
    host = raw.get("ip") or raw.get("host") or raw.get("hostname") or raw.get("publicIp") or raw.get("privateIp")
    public_port = raw.get("publicPort") or raw.get("externalPort") or raw.get("port")
    container_port = raw.get("privatePort") or raw.get("containerPort") or raw.get("internalPort")
    port_type = str(raw.get("type") or raw.get("protocol") or raw.get("scheme") or "tcp").lower()
    is_public = raw.get("isIpPublic")
    label = f"{'pub' if is_public is True else 'prv' if is_public is False else ''},{port_type}".strip(",")
    if host and public_port and container_port:
        return f"{host}:{public_port}->{container_port} ({label})"
    return str(raw)


def normalize_pods(raw: object) -> list[dict[str, str]]:
    if isinstance(raw, dict):
        for key in ("pods", "data", "items"):
            value = raw.get(key)
            if isinstance(value, list):
                return [normalize_pod(item) for item in value if isinstance(item, dict)]
        return [normalize_pod(raw)]
    if isinstance(raw, list):
        return [normalize_pod(item) for item in raw if isinstance(item, dict)]
    return []


def list_pods(runner: Runner, args: argparse.Namespace) -> list[dict[str, str]]:
    completed = runner.run([args.runpodctl, "pod", "list", "-o", "json"], capture=True)
    return normalize_pods(parse_json_output(completed.stdout))


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
        pod = pod_get(runner, args, pod_id)
        if pod:
            connection = pod_connection(pod)
            if connection is not None:
                return connection
        connection = ssh_info_connection(runner, args, pod_id)
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


def pod_get(runner: Runner, args: argparse.Namespace, pod_id: str) -> dict[str, str] | None:
    completed = runner.run([args.runpodctl, "pod", "get", pod_id, "-o", "json"], capture=True, check=False)
    if completed.returncode != 0:
        return None
    pods = normalize_pods(parse_json_output(completed.stdout))
    return pods[0] if pods else None


def ssh_info_connection(runner: Runner, args: argparse.Namespace, pod_id: str) -> PodConnection | None:
    completed = runner.run([args.runpodctl, "ssh", "info", pod_id, "-o", "json"], capture=True, check=False)
    if completed.returncode != 0:
        return None
    return parse_ssh_info(completed.stdout)


def parse_ssh_info(output: str) -> PodConnection | None:
    try:
        raw = parse_json_output(output)
    except json.JSONDecodeError:
        raw = None
    if isinstance(raw, dict):
        host = raw.get("host") or raw.get("hostname") or raw.get("ip") or raw.get("publicIp")
        port = raw.get("port") or raw.get("sshPort")
        user = raw.get("user") or raw.get("username") or "root"
        command = raw.get("command") or raw.get("sshCommand")
        if host and port:
            return PodConnection(host=str(host), port=int(port), user=str(user))
        if command:
            return parse_ssh_command(str(command))
    return parse_ssh_command(output)


def parse_ssh_command(command: str) -> PodConnection | None:
    match = re.search(r"ssh\s+(?:-i\s+\S+\s+)?(?P<target>[A-Za-z0-9_.-]+@[A-Za-z0-9_.-]+)(?:\s+-p\s+(?P<port1>\d+))?", command)
    if not match:
        match = re.search(r"ssh\s+(?:-p\s+(?P<port2>\d+)\s+)?(?P<target>[A-Za-z0-9_.-]+@[A-Za-z0-9_.-]+)", command)
    if not match:
        return None
    user, host = match.group("target").split("@", 1)
    port_text = match.groupdict().get("port1") or match.groupdict().get("port2") or "22"
    return PodConnection(host=host, port=int(port_text), user=user)


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


def remote_transport_setup_command() -> str:
    return (
        "set -euo pipefail; "
        "if ! command -v rsync >/dev/null 2>&1; then "
        "apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y rsync; "
        "fi"
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
    runner.run([args.runpodctl, "pod", "delete", pod_id], check=False)
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
