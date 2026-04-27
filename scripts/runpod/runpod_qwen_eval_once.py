#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
from pathlib import Path
import shlex
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runpod_train_once import (
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
    redact,
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


DEFAULT_MODEL = "Qwen/Qwen3-14B-FP8"
DEFAULT_MODEL_LABEL = "qwen3_14b_fp8"
DEFAULT_POD_NAME_PREFIX = "llm-qwen-eval-once"
DEFAULT_IMAGE = "vllm/vllm-openai:latest"
DEFAULT_OUTPUT = Path("experiments/leverage/qwen3_14b_fp8_runpod.jsonl")
DEFAULT_SCORES = Path("experiments/leverage/qwen3_14b_fp8_scores.csv")
DEFAULT_SUMMARY = Path("experiments/leverage/qwen3_14b_fp8_summary.csv")
SYNC_DIRS = ("src", "tests", "evals")
SYNC_FILES = ("pyproject.toml", "uv.lock", "README.md", "AGENTS.md", "LICENSE")


@dataclass(frozen=True)
class EvalOutputs:
    predictions: Path
    scores: Path
    summary: Path


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def rsync_to_remote_command(args: argparse.Namespace, connection: PodConnection) -> list[str]:
    sources = [*SYNC_DIRS, *SYNC_FILES]
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
        f"{connection.user}@{connection.host}:{args.remote_dir}/experiments/leverage",
        f"{args.repo_root}/experiments/",
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


def remote_vllm_server_command(args: argparse.Namespace) -> str:
    server = (
        "python -m vllm.entrypoints.openai.api_server "
        f"--model {q(args.model)} "
        f"--served-model-name {q(args.model)} "
        "--host 127.0.0.1 "
        "--port 8000 "
        "--language-model-only "
        f"--max-model-len {args.max_model_len} "
        "--reasoning-parser qwen3 "
        "> /tmp/qwen-vllm.log 2>&1 & "
        "echo $! > /tmp/qwen-vllm.pid"
    )
    wait = (
        "for i in $(seq 1 120); do "
        "if curl -fsS http://127.0.0.1:8000/v1/models >/tmp/qwen-models.json; then exit 0; fi; "
        "if ! kill -0 $(cat /tmp/qwen-vllm.pid) 2>/dev/null; then "
        "cat /tmp/qwen-vllm.log; exit 1; "
        "fi; "
        "sleep 5; "
        "done; "
        "cat /tmp/qwen-vllm.log; exit 1"
    )
    return (
        "set -euo pipefail; "
        "cd \"$REMOTE_DIR\"; "
        "export PATH=\"$HOME/.local/bin:$PATH\"; "
        "export VLLM_ENABLE_CUDA_COMPATIBILITY=1; "
        f"{server}; {wait}"
    )


def remote_collect_command(args: argparse.Namespace) -> str:
    task_args = " ".join(f"--tasks {q(task)}" for task in args.tasks)
    return (
        "set -euo pipefail; "
        "cd \"$REMOTE_DIR\"; "
        "export PATH=\"$HOME/.local/bin:$PATH\"; "
        "export OPENAI_BASE_URL=http://127.0.0.1:8000/v1; "
        "export OPENAI_API_KEY=runpod-local; "
        "uv run python -u -m llm.leverage.collect_openai "
        f"{task_args} "
        f"--model {q(args.model)} "
        f"--model-label {q(args.model_label)} "
        f"--output {q(args.output)} "
        f"--max-tokens {args.max_tokens} "
        f"--temperature {args.temperature} "
        "--thinking-mode none "
        "--overwrite"
    )


def remote_evaluate_command(args: argparse.Namespace) -> str:
    task_args = " ".join(f"--tasks {q(task)}" for task in args.tasks)
    return (
        "set -euo pipefail; "
        "cd \"$REMOTE_DIR\"; "
        "export PATH=\"$HOME/.local/bin:$PATH\"; "
        "uv run python -u -m llm.leverage.evaluate "
        f"{task_args} "
        f"--predictions {q(args.output)} "
        f"--output {q(args.scores_output)} "
        f"--summary-output {q(args.summary_output)}"
    )


def remote_stop_server_command() -> str:
    return (
        "set -euo pipefail; "
        "if test -f /tmp/qwen-vllm.pid; then "
        "kill $(cat /tmp/qwen-vllm.pid) 2>/dev/null || true; "
        "fi"
    )


def preflight(args: argparse.Namespace) -> None:
    for task_path in args.tasks:
        if not (args.repo_root / task_path).exists():
            raise FileNotFoundError(f"task file does not exist: {task_path}")
    for source in ("src", "evals", "pyproject.toml", "uv.lock"):
        if not (args.repo_root / source).exists():
            raise FileNotFoundError(f"required source does not exist: {source}")
    if args.max_cost > 5:
        raise ValueError("--max-cost must be 5 or less for the first Qwen spike")
    if args.max_model_len <= 0:
        raise ValueError("--max-model-len must be positive")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    if args.temperature < 0:
        raise ValueError("--temperature must be non-negative")
    if not args.dry_run:
        if not Path(args.runpodctl).exists():
            raise FileNotFoundError(f"runpodctl does not exist: {args.runpodctl}")
        if not args.ssh_key.exists():
            raise FileNotFoundError(f"SSH private key does not exist: {args.ssh_key}")
    if args.bootstrap_sshd and not args.ssh_public_key.exists():
        raise FileNotFoundError(f"SSH public key does not exist: {args.ssh_public_key}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        type=Path,
        action="append",
        default=[Path("evals/leverage_smoke.jsonl"), Path("evals/project_judgment_v0.jsonl")],
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-label", default=DEFAULT_MODEL_LABEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scores-output", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--gpu-type", default="NVIDIA GeForce RTX 4090")
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--max-cost", type=float, default=2.0)
    parser.add_argument("--pod-name-prefix", default=DEFAULT_POD_NAME_PREFIX)
    parser.add_argument("--pod-name")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--container-disk-size", type=int, default=80)
    parser.add_argument("--volume-size", type=int, default=120)
    parser.add_argument("--remote-volume", default="/workspace")
    parser.add_argument("--remote-dir", default=DEFAULT_REMOTE_DIR)
    parser.add_argument("--vcpu", type=int, default=8)
    parser.add_argument("--mem", type=int, default=30)
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
    preflight(args)
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
        run_with_deadline(runner, ssh_command(args, connection, with_remote_dir(args, remote_vllm_server_command(args))), deadline=deadline)
        ensure_before_deadline(deadline)
        run_with_deadline(runner, ssh_command(args, connection, with_remote_dir(args, remote_collect_command(args))), deadline=deadline)
        ensure_before_deadline(deadline)
        run_with_deadline(runner, ssh_command(args, connection, with_remote_dir(args, remote_evaluate_command(args))), deadline=deadline)
        ensure_before_deadline(deadline)
        run_with_deadline(runner, rsync_from_remote_command(args, connection), cwd=args.repo_root, deadline=deadline)
        success = True
        return 0
    finally:
        if pod_id is not None:
            try:
                run_with_deadline(
                    runner,
                    ssh_command(args, connection, remote_stop_server_command()),
                    deadline=deadline,
                    check=False,
                )
            except Exception as exc:
                print(redact(f"failed to stop remote server: {exc}", [api_key, public_key]), file=sys.stderr)
        cleanup_pod(runner, args, pod_id, success=success)


if __name__ == "__main__":
    raise SystemExit(main())
