from argparse import Namespace
import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "runpod" / "runpod_train_once.py"
SPEC = importlib.util.spec_from_file_location("runpod_train_once", MODULE_PATH)
assert SPEC is not None
runpod_train_once = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runpod_train_once)

PodConnection = runpod_train_once.PodConnection
Runner = runpod_train_once.Runner
cleanup_pod = runpod_train_once.cleanup_pod
ensure_before_deadline = runpod_train_once.ensure_before_deadline
find_created_pod = runpod_train_once.find_created_pod
output_paths = runpod_train_once.output_paths
preflight = runpod_train_once.preflight
redact = runpod_train_once.redact
remote_cuda_preflight_command = runpod_train_once.remote_cuda_preflight_command
remote_scaling_command = runpod_train_once.remote_scaling_command
remote_observe_command = runpod_train_once.remote_observe_command
remote_train_command = runpod_train_once.remote_train_command
resume_paths = runpod_train_once.resume_paths
run_with_deadline = runpod_train_once.run_with_deadline
rsync_from_remote_command = runpod_train_once.rsync_from_remote_command
rsync_to_remote_command = runpod_train_once.rsync_to_remote_command
runpodctl_create_command = runpod_train_once.runpodctl_create_command
ssh_base = runpod_train_once.ssh_base
timestamped_name = runpod_train_once.timestamped_name


def write_config(path: Path) -> None:
    path.write_text(
        """
[data]
tokens = "data/processed/tokens.pt"

[outputs]
checkpoint = "checkpoints/run.pt"
metrics = "experiments/runs/run.csv"
observation = "experiments/observations/run.md"
summary = "experiments/summaries/observations.csv"
""",
        encoding="utf-8",
    )


def args(tmp_path: Path, config: Path) -> Namespace:
    return Namespace(
        config=config.relative_to(tmp_path),
        train_extra_args="",
        observe_extra_args="",
        gpu_type="NVIDIA GeForce RTX 4090",
        gpu_count=1,
        image="runpod/pytorch:test",
        container_disk_size=40,
        volume_size=20,
        remote_volume="/workspace",
        remote_dir="/workspace/llm",
        max_cost=0.4,
        pod_name="llm-train-once-20260427-000000",
        pod_name_prefix="llm-train-once",
        vcpu=8,
        mem=20,
        secure_cloud=False,
        runpodctl="/home/dev/bin/runpodctl",
        ssh_key=Path("/home/dev/.runpod/ssh/RunPod-Key-Go"),
        ssh_public_key=tmp_path / "runpod.pub",
        bootstrap_sshd=True,
        repo_root=tmp_path,
        keep_pod=False,
        keep_pod_on_failure=False,
        dry_run=False,
    )


class RecordingRunner:
    dry_run = True

    def __init__(self) -> None:
        self.commands: list[list[str]] = []
        self.timeouts: list[float | None] = []

    def run(
        self,
        command: list[str],
        *,
        cwd=None,
        capture=None,
        check=True,
        timeout=None,
    ):
        self.commands.append(command)
        self.timeouts.append(timeout)
        return runpod_train_once.subprocess.CompletedProcess(command, 0, "", "")


def test_output_paths_reads_train_observe_and_scaling_paths(tmp_path) -> None:
    config = tmp_path / "configs" / "run.toml"
    config.parent.mkdir()
    write_config(config)

    outputs = output_paths(config)

    assert outputs.checkpoint == "checkpoints/run.pt"
    assert outputs.metrics == "experiments/runs/run.csv"
    assert outputs.observation == "experiments/observations/run.md"
    assert outputs.summary == "experiments/summaries/observations.csv"
    assert outputs.tokens == "data/processed/tokens.pt"


def test_runpodctl_create_command_uses_cost_ceiling_and_gpu(tmp_path) -> None:
    config = tmp_path / "configs" / "run.toml"
    config.parent.mkdir()
    write_config(config)
    (tmp_path / "runpod.pub").write_text("ssh-ed25519 test-key", encoding="utf-8")

    command = runpodctl_create_command(args(tmp_path, config))

    assert command[:3] == ["/home/dev/bin/runpodctl", "create", "pod"]
    assert command[command.index("--gpuType") + 1] == "NVIDIA GeForce RTX 4090"
    assert command[command.index("--cost") + 1] == "0.4"
    assert "--communityCloud" in command
    assert command[command.index("--env") + 1] == "PUBLIC_KEY=ssh-ed25519 test-key"
    assert "openssh-server" in command[command.index("--args") + 1]


def test_remote_commands_use_config_outputs(tmp_path) -> None:
    config = tmp_path / "configs" / "run.toml"
    config.parent.mkdir()
    write_config(config)
    outputs = output_paths(config)

    train = remote_train_command(Path("configs/run.toml"), "--max-iters 20")
    observe = remote_observe_command(Path("configs/run.toml"), "--eval-iters 1")
    scaling = remote_scaling_command(Path("configs/run.toml"), outputs)

    assert "llm.train --config configs/run.toml --max-iters 20 --device cuda" in train
    assert "llm.observe --config configs/run.toml --eval-iters 1 --device cuda" in observe
    assert "--checkpoint checkpoints/run.pt" in scaling
    assert "--summary experiments/summaries/observations.csv" in scaling


def test_remote_commands_keep_explicit_device() -> None:
    train = remote_train_command(Path("configs/run.toml"), "--device cpu --max-iters 20")
    observe = remote_observe_command(Path("configs/run.toml"), "--device=cpu --eval-iters 1")

    assert train.count("--device") == 1
    assert "--device cpu" in train
    assert "--device=cpu" in observe


def test_remote_cuda_preflight_fails_without_cuda() -> None:
    command = remote_cuda_preflight_command()

    assert "torch.cuda.is_available()" in command
    assert "raise SystemExit" in command


def test_rsync_to_remote_uses_allowlist_and_explicit_token_file(tmp_path) -> None:
    config = tmp_path / "configs" / "run.toml"
    config.parent.mkdir()
    write_config(config)
    for path in (
        tmp_path / "src",
        tmp_path / "tests",
        tmp_path / "configs",
        tmp_path / "eval_prompts",
        tmp_path / "data" / "processed",
    ):
        path.mkdir(parents=True, exist_ok=True)
    for file_name in ("pyproject.toml", "uv.lock", "README.md", "AGENTS.md", "LICENSE"):
        (tmp_path / file_name).write_text("", encoding="utf-8")
    (tmp_path / "data" / "processed" / "tokens.pt").write_text("", encoding="utf-8")

    command = rsync_to_remote_command(args(tmp_path, config), PodConnection("host", 2222))

    assert "--relative" in command
    assert "src" in command
    assert "data/processed/tokens.pt" in command
    assert str(tmp_path / ".venv") not in command
    assert str(tmp_path / "checkpoints") not in command


def test_rsync_to_remote_includes_resume_checkpoint(tmp_path) -> None:
    config = tmp_path / "configs" / "run.toml"
    config.parent.mkdir()
    write_config(config)
    for path in (
        tmp_path / "src",
        tmp_path / "tests",
        tmp_path / "configs",
        tmp_path / "eval_prompts",
        tmp_path / "data" / "processed",
        tmp_path / "checkpoints",
    ):
        path.mkdir(parents=True, exist_ok=True)
    for file_name in ("pyproject.toml", "uv.lock", "README.md", "AGENTS.md", "LICENSE"):
        (tmp_path / file_name).write_text("", encoding="utf-8")
    (tmp_path / "data" / "processed" / "tokens.pt").write_text("", encoding="utf-8")
    (tmp_path / "checkpoints" / "run.pt").write_text("", encoding="utf-8")
    run_args = args(tmp_path, config)
    run_args.train_extra_args = "--resume checkpoints/run.pt"

    command = rsync_to_remote_command(run_args, PodConnection("host", 2222))

    assert "checkpoints/run.pt" in command


def test_resume_paths_reads_space_and_equals_forms() -> None:
    assert resume_paths("--resume checkpoints/run.pt --max-iters 30") == ["checkpoints/run.pt"]
    assert resume_paths("--resume=checkpoints/run.pt") == ["checkpoints/run.pt"]


def test_rsync_from_remote_fetches_checkpoint_and_experiments(tmp_path) -> None:
    config = tmp_path / "configs" / "run.toml"
    config.parent.mkdir()
    write_config(config)

    command = rsync_from_remote_command(args(tmp_path, config), PodConnection("host", 2222))

    assert "root@host:/workspace/llm/checkpoints" in command
    assert "root@host:/workspace/llm/experiments" in command
    assert command[-1] == f"{tmp_path}/"


def test_cleanup_policy_removes_on_success_and_failure(tmp_path) -> None:
    config = tmp_path / "configs" / "run.toml"
    config.parent.mkdir()
    write_config(config)
    runner = RecordingRunner()

    cleanup_pod(runner, args(tmp_path, config), "pod123", success=True)
    cleanup_pod(runner, args(tmp_path, config), "pod456", success=False)

    assert runner.commands == [
        ["/home/dev/bin/runpodctl", "remove", "pod", "pod123"],
        ["/home/dev/bin/runpodctl", "remove", "pod", "pod456"],
    ]


def test_cleanup_policy_keeps_failure_when_requested(tmp_path) -> None:
    config = tmp_path / "configs" / "run.toml"
    config.parent.mkdir()
    write_config(config)
    run_args = args(tmp_path, config)
    run_args.keep_pod_on_failure = True
    runner = RecordingRunner()

    cleanup_pod(runner, run_args, "pod456", success=False)

    assert runner.commands == []


def test_redact_hides_api_key() -> None:
    assert redact("https://api.runpod.io/graphql?api_key=secret", ["secret"]) == (
        "https://api.runpod.io/graphql?api_key=[REDACTED]"
    )


def test_redact_hides_public_key() -> None:
    assert redact("PUBLIC_KEY=ssh-ed25519 public-key", ["ssh-ed25519 public-key"]) == (
        "PUBLIC_KEY=[REDACTED]"
    )


def test_timestamped_name_uses_prefix() -> None:
    assert timestamped_name("llm-train-once").startswith("llm-train-once-")


def test_ensure_before_deadline_rejects_expired_deadline() -> None:
    with pytest.raises(TimeoutError, match="maximum runtime"):
        ensure_before_deadline(0)


def test_find_created_pod_uses_before_after_diff() -> None:
    before = [{"ID": "old", "NAME": "same"}]
    after = [*before, {"ID": "new", "NAME": "same"}]

    assert find_created_pod(before, after, "same")["ID"] == "new"


def test_find_created_pod_rejects_missing_created_pod() -> None:
    with pytest.raises(RuntimeError, match="created pod"):
        find_created_pod([], [{"ID": "other", "NAME": "other"}], "same")


def test_ssh_base_uses_non_hanging_options(tmp_path) -> None:
    config = tmp_path / "configs" / "run.toml"
    config.parent.mkdir()
    write_config(config)

    command = ssh_base(args(tmp_path, config), PodConnection("host", 2222))

    assert "BatchMode=yes" in command
    assert "ConnectTimeout=10" in command
    assert "ConnectionAttempts=1" in command
    assert "ServerAliveInterval=15" in command
    assert "ServerAliveCountMax=2" in command


def test_run_with_deadline_passes_timeout(tmp_path, monkeypatch) -> None:
    runner = RecordingRunner()
    monkeypatch.setattr(runpod_train_once.time, "monotonic", lambda: 100.0)

    run_with_deadline(runner, ["echo", "ok"], deadline=130.0)

    assert runner.timeouts == [30.0]


def test_preflight_rejects_missing_tokens_before_pod_creation(tmp_path) -> None:
    config = tmp_path / "configs" / "run.toml"
    config.parent.mkdir()
    write_config(config)
    for path in (tmp_path / "src",):
        path.mkdir(parents=True)
    for file_name in ("pyproject.toml", "uv.lock"):
        (tmp_path / file_name).write_text("", encoding="utf-8")
    (tmp_path / "runpod.pub").write_text("ssh-ed25519 test-key", encoding="utf-8")
    run_args = args(tmp_path, config)
    run_args.dry_run = True

    with pytest.raises(FileNotFoundError, match="token file"):
        preflight(run_args, output_paths(config))
