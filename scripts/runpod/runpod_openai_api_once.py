#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from urllib import error, request

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runpod_train_once import (
    DEFAULT_RUNPODCTL,
    DEFAULT_SECRET_PATH,
    CommandError,
    Runner,
    active_pods,
    assert_no_existing_pods,
    cleanup_pod,
    deadline_from_minutes,
    ensure_before_deadline,
    find_created_pod,
    list_pods,
    load_api_key,
    run_with_deadline,
    shell_join,
    timestamped_name,
)


DEFAULT_IMAGE = "vllm/vllm-openai:latest"
DEFAULT_MODEL = "Qwen/Qwen3-14B-FP8"
DEFAULT_MODEL_LABEL = "qwen3_14b_fp8"
DEFAULT_POD_NAME_PREFIX = "llm-openai-api-once"
DEFAULT_TASKS = [Path("evals/leverage_smoke.jsonl"), Path("evals/project_judgment_v0.jsonl")]
DEFAULT_OUTPUT = Path("experiments/leverage/qwen3_14b_fp8_runpod.jsonl")
DEFAULT_SCORES = Path("experiments/leverage/qwen3_14b_fp8_scores.csv")
DEFAULT_SUMMARY = Path("experiments/leverage/qwen3_14b_fp8_summary.csv")
DEFAULT_API_KEY = "runpod-local"
DEFAULT_USER_AGENT = "llm-runpod-eval/1.0"


@dataclass(frozen=True)
class ApiEndpoint:
    base_url: str
    host: str
    port: int


def server_args(args: argparse.Namespace) -> str:
    parts = [
        "--model",
        args.model,
        "--served-model-name",
        args.served_model_name,
        "--host",
        "0.0.0.0",
        "--port",
        str(args.server_port),
        "--max-model-len",
        str(args.max_model_len),
    ]
    if args.language_model_only:
        parts.append("--language-model-only")
    if args.reasoning_parser:
        parts.extend(["--reasoning-parser", args.reasoning_parser])
    if args.api_key:
        parts.extend(["--api-key", args.api_key])
    if args.server_extra_args:
        parts.extend(args.server_extra_args)
    return shell_join(parts)


def runpodctl_create_api_command(args: argparse.Namespace) -> list[str]:
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
        f"{args.server_port}/http",
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
    for name, value in args.server_env:
        command.extend(["--env", f"{name}={value}"])
    command.extend(["--args", server_args(args)])
    return command


def parse_port_mappings(ports: str) -> list[tuple[str, int, int, str]]:
    mappings: list[tuple[str, int, int, str]] = []
    pattern = re.compile(r"([A-Za-z0-9.-]+):(\d+)->(\d+)\s*\(([^)]*)\)")
    for host, public_port, container_port, labels in pattern.findall(ports):
        mappings.append((host, int(public_port), int(container_port), labels.lower()))
    return mappings


def api_endpoint_from_pod(pod: dict[str, str], server_port: int) -> ApiEndpoint | None:
    ports = pod.get("PORTS", "")
    for url in re.findall(r"https?://[A-Za-z0-9./:_-]+", ports):
        if str(server_port) in url or str(server_port) in ports:
            scheme, rest = url.split("://", 1)
            host_port = rest.split("/", 1)[0]
            host, _, port_text = host_port.rpartition(":")
            port = int(port_text) if host and port_text.isdigit() else (443 if scheme == "https" else 80)
            return ApiEndpoint(base_url=url.rstrip("/") + "/v1", host=host or host_port, port=port)
    has_http_mapping = False
    for host, public_port, container_port, labels in parse_port_mappings(pod.get("PORTS", "")):
        if container_port != server_port or "http" not in labels:
            continue
        has_http_mapping = True
        endpoint = ApiEndpoint(base_url=f"http://{host}:{public_port}/v1", host=host, port=public_port)
        if "prv" not in labels:
            return endpoint
    pod_id = pod.get("ID")
    if pod_id and has_http_mapping and str(pod.get("STATUS", "")).upper() == "RUNNING":
        host = f"{pod_id}-{server_port}.proxy.runpod.net"
        return ApiEndpoint(base_url=f"https://{host}/v1", host=host, port=443)
    return None


def wait_for_api_endpoint(
    runner: Runner,
    args: argparse.Namespace,
    pod_id: str,
    timeout_seconds: int,
) -> ApiEndpoint:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        pod = next((item for item in list_pods(runner, args) if item.get("ID") == pod_id), None)
        if pod is not None:
            endpoint = api_endpoint_from_pod(pod, args.server_port)
            if endpoint is not None:
                return endpoint
        time.sleep(10)
    raise TimeoutError(f"pod did not expose public HTTP port {args.server_port} within {timeout_seconds} seconds")


def wait_for_models(endpoint: ApiEndpoint, timeout_seconds: int, api_key: str) -> None:
    deadline = time.monotonic() + timeout_seconds
    url = endpoint.base_url.rstrip("/") + "/models"
    probe = request.Request(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "User-Agent": DEFAULT_USER_AGENT,
        },
    )
    while time.monotonic() < deadline:
        try:
            with request.urlopen(probe, timeout=5) as response:
                if 200 <= response.status < 300:
                    return
        except (error.URLError, TimeoutError):
            pass
        time.sleep(5)
    raise TimeoutError(f"OpenAI-compatible API did not become ready: {endpoint.base_url}")


def local_collect_command(args: argparse.Namespace, endpoint: ApiEndpoint) -> list[str]:
    command = [
        "env",
        f"OPENAI_BASE_URL={endpoint.base_url}",
        f"OPENAI_API_KEY={args.api_key}",
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
            "--timeout-seconds",
            str(args.request_timeout_seconds),
            "--overwrite",
        ]
    )
    if args.thinking_param:
        command.extend(["--thinking-param", args.thinking_param])
    return command


def local_evaluate_command(args: argparse.Namespace) -> list[str]:
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


def run_local_command(
    command: list[str],
    *,
    cwd: Path,
    deadline: float | None,
    secrets: list[str],
    dry_run: bool,
) -> None:
    print(f"$ {redact_command(command, secrets)}")
    if dry_run:
        return
    process = subprocess.Popen(command, cwd=cwd, text=True, start_new_session=True)
    try:
        return_code = process.wait(timeout=remaining_seconds(deadline))
    except BaseException:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise
    if return_code != 0:
        raise CommandError(f"command failed with exit code {return_code}")


def cleanup_named_pods(runner: Runner, args: argparse.Namespace, pod_id: str | None) -> None:
    if runner.dry_run:
        return
    for pod in active_pods(runner, args):
        if pod_id is not None and pod.get("ID") == pod_id:
            continue
        if pod.get("NAME") == args.pod_name:
            runner.run([args.runpodctl, "remove", "pod", str(pod["ID"])], check=False)


def remaining_seconds(deadline: float | None) -> float | None:
    if deadline is None:
        return None
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("maximum runtime was reached")
    return remaining


def redact_command(command: list[str], secrets: list[str]) -> str:
    text = shell_join(command)
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[REDACTED]")
    return text


def parse_env(values: list[str]) -> list[tuple[str, str]]:
    env: list[tuple[str, str]] = []
    for value in values:
        if "=" not in value:
            raise ValueError("--server-env values must use NAME=VALUE")
        name, env_value = value.split("=", 1)
        if not name:
            raise ValueError("--server-env name must not be empty")
        env.append((name, env_value))
    return env


def preflight(args: argparse.Namespace) -> None:
    for task in args.tasks:
        if not (args.repo_root / task).exists():
            raise FileNotFoundError(f"task file does not exist: {task}")
    if args.max_cost > args.max_allowed_cost:
        raise ValueError("--max-cost must not exceed --max-allowed-cost")
    if args.server_port <= 0:
        raise ValueError("--server-port must be positive")
    if args.max_model_len <= 0:
        raise ValueError("--max-model-len must be positive")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    if args.temperature < 0:
        raise ValueError("--temperature must be non-negative")
    if not args.model_label:
        raise ValueError("--model-label must not be empty")
    if not args.dry_run and not Path(args.runpodctl).exists():
        raise FileNotFoundError(f"runpodctl does not exist: {args.runpodctl}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, action="append")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--served-model-name")
    parser.add_argument("--model-label", default=DEFAULT_MODEL_LABEL)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scores-output", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--gpu-type", default="NVIDIA GeForce RTX 4090")
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--max-cost", type=float, default=2.0)
    parser.add_argument("--max-allowed-cost", type=float, default=2.0)
    parser.add_argument("--pod-name-prefix", default=DEFAULT_POD_NAME_PREFIX)
    parser.add_argument("--pod-name")
    parser.add_argument("--container-disk-size", type=int, default=80)
    parser.add_argument("--volume-size", type=int, default=120)
    parser.add_argument("--remote-volume", default="/workspace")
    parser.add_argument("--vcpu", type=int, default=8)
    parser.add_argument("--mem", type=int, default=29)
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--reasoning-parser")
    parser.add_argument("--language-model-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--server-env", action="append", default=["VLLM_ENABLE_CUDA_COMPATIBILITY=1"])
    parser.add_argument("--server-extra-args", nargs=argparse.REMAINDER, default=[])
    parser.add_argument("--thinking-mode", choices=("disabled", "none"), default="none")
    parser.add_argument(
        "--thinking-param",
        choices=("chat_template_kwargs", "enable_thinking"),
        default="chat_template_kwargs",
    )
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--request-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--runpodctl", default=DEFAULT_RUNPODCTL)
    parser.add_argument("--secret-path", type=Path, default=DEFAULT_SECRET_PATH)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--wait-seconds", type=int, default=180)
    parser.add_argument("--api-wait-seconds", type=int, default=600)
    parser.add_argument("--max-runtime-minutes", type=int, default=20)
    parser.add_argument("--secure-cloud", action="store_true")
    parser.add_argument("--allow-existing-pods", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.repo_root = args.repo_root.resolve()
    if args.served_model_name is None:
        args.served_model_name = args.model
    if args.pod_name is None:
        args.pod_name = timestamped_name(args.pod_name_prefix)
    if args.tasks is None:
        args.tasks = DEFAULT_TASKS.copy()
    args.server_env = parse_env(args.server_env)
    args.keep_pod = False
    args.keep_pod_on_failure = False
    return args


def main() -> int:
    args = normalize_args(parse_args())
    preflight(args)
    runpod_api_key = "" if args.dry_run else load_api_key(args.secret_path)
    runner = Runner(
        dry_run=args.dry_run,
        secrets=[runpod_api_key, args.api_key],
        env={"RUNPOD_API_KEY": runpod_api_key} if runpod_api_key else None,
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
        run_with_deadline(runner, runpodctl_create_api_command(args), cwd=args.repo_root, deadline=deadline)
        if args.dry_run:
            endpoint = ApiEndpoint("http://dry-run.runpod.local:8000/v1", "dry-run.runpod.local", 8000)
            pod_id = "dry-run-pod"
        else:
            pods_after_create = list_pods(runner, args)
            pod = find_created_pod(pods_before_create, pods_after_create, args.pod_name)
            pod_id = str(pod["ID"])
            print(f"created pod: {pod_id}")
            endpoint = wait_for_api_endpoint(runner, args, pod_id, args.wait_seconds)
            print(f"api: {endpoint.base_url}")
            wait_for_models(endpoint, args.api_wait_seconds, args.api_key)

        ensure_before_deadline(deadline)
        run_local_command(
            local_collect_command(args, endpoint),
            cwd=args.repo_root,
            deadline=deadline,
            secrets=[args.api_key],
            dry_run=args.dry_run,
        )
        ensure_before_deadline(deadline)
        run_local_command(
            local_evaluate_command(args),
            cwd=args.repo_root,
            deadline=deadline,
            secrets=[args.api_key],
            dry_run=args.dry_run,
        )
        success = True
        return 0
    finally:
        cleanup_named_pods(runner, args, pod_id)
        cleanup_pod(runner, args, pod_id, success=success)
        if not args.dry_run:
            remaining = active_pods(runner, args)
            if remaining:
                raise RuntimeError("RunPod account still has active pods after cleanup")


if __name__ == "__main__":
    raise SystemExit(main())
