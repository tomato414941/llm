from argparse import Namespace
import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "runpod" / "run_once.py"
SPEC = importlib.util.spec_from_file_location("runpod_run_once", MODULE_PATH)
assert SPEC is not None
runpod_run_once = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runpod_run_once)

PodConnection = runpod_run_once.PodConnection
preflight = runpod_run_once.preflight
remote_cuda_smoke_command = runpod_run_once.remote_cuda_smoke_command
remote_user_command = runpod_run_once.remote_user_command
rsync_from_remote_command = runpod_run_once.rsync_from_remote_command
rsync_to_remote_command = runpod_run_once.rsync_to_remote_command
runpodctl_create_command = runpod_run_once.runpodctl_create_command
split_shell_command = runpod_run_once.split_shell_command


def write_repo_shape(tmp_path: Path) -> None:
    for directory in ("src", "tests", "tracks/leverage/configs", "tracks/leverage/evals"):
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)
    for file_name in ("pyproject.toml", "uv.lock", "README.md", "AGENTS.md", "LICENSE"):
        (tmp_path / file_name).write_text("", encoding="utf-8")
    (tmp_path / "runpod.pub").write_text("ssh-ed25519 public-key", encoding="utf-8")


def args(tmp_path: Path) -> Namespace:
    return Namespace(
        name="llm-leverage-sft-smoke",
        pod_name_prefix="llm-leverage-sft-smoke",
        pod_name="llm-leverage-sft-smoke-20260428-000000",
        sync=["tracks/leverage/configs", "tracks/leverage/evals"],
        output=["outputs/leverage-sft-smoke"],
        local=["uv run python -m llm.leverage.sft_smoke_preflight --overwrite"],
        remote=["uv run python -m llm.leverage.train_sft_smoke --config tracks/leverage/configs/leverage-sft-smoke.toml"],
        gpu_type="NVIDIA GeForce RTX 3090",
        gpu_count=1,
        max_cost=0.8,
        image="runpod/pytorch:test",
        container_disk_size=80,
        volume_size=80,
        remote_volume="/workspace",
        remote_dir="/workspace/llm",
        vcpu=8,
        mem=24,
        secure_cloud=False,
        runpodctl="/home/dev/bin/runpodctl",
        secret_path=tmp_path / "runpod",
        ssh_key=Path("/home/dev/.runpod/ssh/RunPod-Key-Go"),
        ssh_public_key=tmp_path / "runpod.pub",
        bootstrap_sshd=True,
        repo_root=tmp_path,
        wait_seconds=900,
        ssh_wait_seconds=180,
        max_runtime_minutes=30,
        allow_existing_pods=False,
        dry_run=True,
        keep_pod=False,
        keep_pod_on_failure=False,
    )


def test_split_shell_command_rejects_empty_command() -> None:
    with pytest.raises(ValueError, match="empty"):
        split_shell_command("")


def test_preflight_requires_remote_and_output(tmp_path: Path) -> None:
    write_repo_shape(tmp_path)
    run_args = args(tmp_path)
    run_args.remote = []

    with pytest.raises(ValueError, match="--remote"):
        preflight(run_args)

    run_args.remote = ["echo ok"]
    run_args.output = []
    with pytest.raises(ValueError, match="--output"):
        preflight(run_args)


def test_preflight_accepts_generic_run_shape(tmp_path: Path) -> None:
    write_repo_shape(tmp_path)

    preflight(args(tmp_path))


def test_runpodctl_create_command_uses_generic_args(tmp_path: Path) -> None:
    write_repo_shape(tmp_path)
    run_args = args(tmp_path)

    command = runpodctl_create_command(run_args)

    assert command[command.index("--gpuType") + 1] == "NVIDIA GeForce RTX 3090"
    assert command[command.index("--cost") + 1] == "0.8"
    assert command[command.index("--imageName") + 1] == "runpod/pytorch:test"


def test_rsync_to_remote_syncs_default_and_explicit_sources(tmp_path: Path) -> None:
    write_repo_shape(tmp_path)

    command = rsync_to_remote_command(args(tmp_path), PodConnection("host", 2222))

    assert "src" in command
    assert "tests" in command
    assert "tracks/leverage/configs" in command
    assert "tracks/leverage/evals" in command
    assert "tracks/from-scratch/data/processed/tokens.pt" not in command


def test_rsync_from_remote_fetches_outputs(tmp_path: Path) -> None:
    write_repo_shape(tmp_path)

    command = rsync_from_remote_command(args(tmp_path), PodConnection("host", 2222))

    assert "root@host:/workspace/llm/outputs/leverage-sft-smoke" in command
    assert command[-1] == f"{tmp_path}/"


def test_remote_cuda_smoke_requires_cuda() -> None:
    command = remote_cuda_smoke_command()

    assert "torch.cuda.is_available()" in command
    assert "cuda_device=" in command


def test_remote_user_command_runs_inside_remote_repo_with_uv_on_path() -> None:
    command = remote_user_command("uv run python -m example")

    assert 'cd "$REMOTE_DIR"' in command
    assert 'export PATH="$HOME/.local/bin:$PATH"' in command
    assert "uv run python -m example" in command
