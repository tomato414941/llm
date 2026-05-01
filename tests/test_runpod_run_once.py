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

COMMON_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "runpod" / "runpod_common.py"
COMMON_SPEC = importlib.util.spec_from_file_location("runpod_common", COMMON_MODULE_PATH)
assert COMMON_SPEC is not None
runpod_common = importlib.util.module_from_spec(COMMON_SPEC)
assert COMMON_SPEC.loader is not None
COMMON_SPEC.loader.exec_module(runpod_common)

PodConnection = runpod_common.PodConnection
normalize_pods = runpod_common.normalize_pods
parse_ssh_info = runpod_common.parse_ssh_info
parse_ssh_info_error = runpod_common.parse_ssh_info_error
public_pod_metadata = runpod_common.public_pod_metadata
preflight = runpod_run_once.preflight
remote_cuda_smoke_command = runpod_run_once.remote_cuda_smoke_command
remote_user_command = runpod_run_once.remote_user_command
rsync_from_remote_command = runpod_run_once.rsync_from_remote_command
rsync_to_remote_command = runpod_run_once.rsync_to_remote_command
runpodctl_create_command = runpod_run_once.runpodctl_create_command
split_shell_command = runpod_run_once.split_shell_command
step_name = runpod_run_once.step_name
wait_for_connection = runpod_common.wait_for_connection


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
        template_id=None,
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

    assert command[:4] == ["/home/dev/bin/runpodctl", "pod", "create", "-o"]
    assert command[command.index("--gpu-id") + 1] == "NVIDIA GeForce RTX 3090"
    assert command[command.index("--image") + 1] == "runpod/pytorch:test"
    assert "--public-ip" in command
    assert "--cost" not in command


def test_runpodctl_create_command_can_use_template_instead_of_image(tmp_path: Path) -> None:
    write_repo_shape(tmp_path)
    run_args = args(tmp_path)
    run_args.template_id = "runpod-torch-v280"

    command = runpodctl_create_command(run_args)

    assert command[command.index("--template-id") + 1] == "runpod-torch-v280"
    assert "--image" not in command


def test_rsync_to_remote_syncs_default_and_explicit_sources(tmp_path: Path) -> None:
    write_repo_shape(tmp_path)

    command = rsync_to_remote_command(args(tmp_path), PodConnection("host", 2222))

    assert "--no-owner" in command
    assert "--no-group" in command
    assert "src" in command
    assert "tests" in command
    assert "tracks/leverage/configs" in command
    assert "tracks/leverage/evals" in command
    assert "tracks/from-scratch/data/processed/tokens.pt" not in command
    assert step_name(command) == "repo_sync"


def test_rsync_from_remote_fetches_outputs(tmp_path: Path) -> None:
    write_repo_shape(tmp_path)

    command = rsync_from_remote_command(args(tmp_path), PodConnection("host", 2222))

    assert "--no-owner" in command
    assert "--no-group" in command
    assert "root@host:/workspace/llm/outputs/leverage-sft-smoke" in command
    assert command[-1] == f"{tmp_path / 'outputs'}/"
    assert step_name(command) == "output_sync"


def test_normalize_pods_formats_v2_port_mappings() -> None:
    pods = normalize_pods(
        [
            {
                "id": "pod123",
                "name": "run",
                "desiredStatus": "RUNNING",
                "imageName": "runpod/pytorch:test",
                "costPerHr": 0.69,
                "uptimeSeconds": 12,
                "machine": {"gpuDisplayName": "RTX 4090", "location": "US"},
                "ports": [
                    {
                        "ip": "104.1.2.3",
                        "publicPort": 45678,
                        "privatePort": 22,
                        "type": "tcp",
                        "isIpPublic": True,
                    }
                ],
            }
        ]
    )

    assert pods[0]["ID"] == "pod123"
    assert pods[0]["NAME"] == "run"
    assert pods[0]["STATUS"] == "RUNNING"
    assert pods[0]["PORTS"] == "104.1.2.3:45678->22 (pub,tcp)"
    assert pods[0]["IMAGE"] == "runpod/pytorch:test"
    assert pods[0]["COST_PER_HR"] == "0.69"
    assert pods[0]["UPTIME_SECONDS"] == "12"
    assert pods[0]["GPU_DISPLAY_NAME"] == "RTX 4090"
    assert pods[0]["LOCATION"] == "US"


def test_public_pod_metadata_keeps_observability_fields_without_secrets() -> None:
    pod = normalize_pods(
        [
            {
                "id": "pod123",
                "name": "run",
                "desiredStatus": "RUNNING",
                "env": {"PUBLIC_KEY": "ssh-ed25519 secret"},
                "ssh": {"error": "pod not ready"},
                "machine": {"gpuDisplayName": "RTX 4090", "location": "US"},
                "imageName": "runpod/pytorch:test",
                "ports": ["22/tcp"],
            }
        ]
    )[0]

    metadata = public_pod_metadata(pod)

    assert metadata["ID"] == "pod123"
    assert metadata["SSH_ERROR"] == "pod not ready"
    assert metadata["GPU_DISPLAY_NAME"] == "RTX 4090"
    assert "env" not in metadata
    assert "PUBLIC_KEY" not in metadata


def test_parse_ssh_info_reads_command_shape() -> None:
    connection = parse_ssh_info('{"command":"ssh root@213.173.108.12 -p 17445 -i ~/.ssh/id_ed25519"}')

    assert connection == PodConnection("213.173.108.12", 17445)


def test_parse_ssh_info_error_reads_runpod_not_ready_shape() -> None:
    assert parse_ssh_info_error('{"error":"pod not ready","status":"RUNNING"}') == "pod not ready"


def test_wait_for_connection_records_poll_events(tmp_path: Path) -> None:
    class FakeRunner:
        def __init__(self) -> None:
            self.calls = 0

        def run(self, command, **kwargs):
            from subprocess import CompletedProcess

            if command[1:3] == ["pod", "get"]:
                return CompletedProcess(
                    command,
                    0,
                    '{"id":"pod1","desiredStatus":"RUNNING","ports":"22/tcp",'
                    '"uptimeSeconds":7,"machine":{"gpuDisplayName":"RTX 4090","location":"US"},'
                    '"memoryInGb":62,"vcpuCount":12}',
                    "",
                )
            if command[1:3] == ["ssh", "info"]:
                self.calls += 1
                if self.calls == 1:
                    return CompletedProcess(command, 0, '{"error":"pod not ready","status":"RUNNING"}', "")
                return CompletedProcess(command, 0, '{"ip":"203.0.113.10","port":10226}', "")
            raise AssertionError(command)

    run_args = args(tmp_path)
    poll_events: list[dict[str, object]] = []

    connection = wait_for_connection(FakeRunner(), run_args, "pod1", 30, poll_events)

    assert connection == PodConnection("203.0.113.10", 10226)
    assert poll_events[0]["pod_status"] == "RUNNING"
    assert poll_events[0]["pod_uptime_seconds"] == "7"
    assert poll_events[0]["pod_location"] == "US"
    assert poll_events[0]["pod_gpu_display_name"] == "RTX 4090"
    assert poll_events[0]["ssh_info_error"] == "pod not ready"
    assert poll_events[-1]["ssh_info_has_connection"] is True
    assert poll_events[-1]["ssh_host"] == "203.0.113.10"
    assert poll_events[-1]["ssh_port"] == 10226


def test_remote_cuda_smoke_requires_cuda() -> None:
    command = remote_cuda_smoke_command()

    assert "torch.cuda.is_available()" in command
    assert "cuda_device=" in command


def test_remote_user_command_runs_inside_remote_repo_with_uv_on_path() -> None:
    command = remote_user_command("uv run python -m example")

    assert 'cd "$REMOTE_DIR"' in command
    assert 'export PATH="$HOME/.local/bin:$PATH"' in command
    assert "uv run python -m example" in command
