from argparse import Namespace
import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "runpod" / "runpod_qwen_eval_once.py"
SPEC = importlib.util.spec_from_file_location("runpod_qwen_eval_once", MODULE_PATH)
assert SPEC is not None
runpod_qwen_eval_once = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runpod_qwen_eval_once)

PodConnection = runpod_qwen_eval_once.PodConnection
preflight = runpod_qwen_eval_once.preflight
remote_collect_command = runpod_qwen_eval_once.remote_collect_command
remote_evaluate_command = runpod_qwen_eval_once.remote_evaluate_command
remote_vllm_server_command = runpod_qwen_eval_once.remote_vllm_server_command
rsync_from_remote_command = runpod_qwen_eval_once.rsync_from_remote_command
rsync_to_remote_command = runpod_qwen_eval_once.rsync_to_remote_command
runpodctl_create_command = runpod_qwen_eval_once.runpodctl_create_command


def args(tmp_path: Path) -> Namespace:
    return Namespace(
        tasks=[Path("evals/leverage_smoke.jsonl"), Path("evals/project_judgment_v0.jsonl")],
        model="Qwen/Qwen3-14B-FP8",
        model_label="qwen3_14b_fp8",
        output=Path("experiments/leverage/qwen.jsonl"),
        scores_output=Path("experiments/leverage/qwen_scores.csv"),
        summary_output=Path("experiments/leverage/qwen_summary.csv"),
        max_model_len=8192,
        max_tokens=512,
        temperature=0.0,
        gpu_type="NVIDIA GeForce RTX 4090",
        gpu_count=1,
        max_cost=2.0,
        pod_name_prefix="llm-qwen-eval-once",
        pod_name="llm-qwen-eval-once-20260427-000000",
        image="vllm/vllm-openai:test",
        container_disk_size=80,
        volume_size=120,
        remote_volume="/workspace",
        remote_dir="/workspace/llm",
        vcpu=8,
        mem=29,
        secure_cloud=False,
        runpodctl="/home/dev/bin/runpodctl",
        secret_path=tmp_path / "runpod",
        ssh_key=Path("/home/dev/.runpod/ssh/RunPod-Key-Go"),
        ssh_public_key=tmp_path / "runpod.pub",
        bootstrap_sshd=True,
        repo_root=tmp_path,
        wait_seconds=900,
        ssh_wait_seconds=180,
        max_runtime_minutes=60,
        allow_existing_pods=False,
        dry_run=True,
        keep_pod=False,
        keep_pod_on_failure=False,
    )


def write_repo_shape(tmp_path: Path) -> None:
    for directory in ("src", "tests", "evals"):
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)
    for file_name in ("pyproject.toml", "uv.lock", "README.md", "AGENTS.md", "LICENSE"):
        (tmp_path / file_name).write_text("", encoding="utf-8")
    (tmp_path / "evals" / "leverage_smoke.jsonl").write_text("{}", encoding="utf-8")
    (tmp_path / "evals" / "project_judgment_v0.jsonl").write_text("{}", encoding="utf-8")
    (tmp_path / "runpod.pub").write_text("ssh-ed25519 public-key", encoding="utf-8")


def test_runpodctl_create_command_uses_4090_and_cost_ceiling(tmp_path: Path) -> None:
    write_repo_shape(tmp_path)

    command = runpodctl_create_command(args(tmp_path))

    assert command[:3] == ["/home/dev/bin/runpodctl", "create", "pod"]
    assert command[command.index("--gpuType") + 1] == "NVIDIA GeForce RTX 4090"
    assert command[command.index("--gpuCount") + 1] == "1"
    assert command[command.index("--imageName") + 1] == "vllm/vllm-openai:test"
    assert command[command.index("--cost") + 1] == "2.0"
    assert command[command.index("--mem") + 1] == "29"
    assert command[command.index("--volumeSize") + 1] == "120"


def test_rsync_to_remote_syncs_only_repo_inputs(tmp_path: Path) -> None:
    write_repo_shape(tmp_path)

    command = rsync_to_remote_command(args(tmp_path), PodConnection("host", 2222))

    assert "--relative" in command
    assert "src" in command
    assert "evals" in command
    assert "pyproject.toml" in command
    assert "data/processed/tokens.pt" not in command
    assert command[-1] == "root@host:/workspace/llm/"


def test_remote_vllm_server_uses_fp8_model_and_short_context(tmp_path: Path) -> None:
    command = remote_vllm_server_command(args(tmp_path))

    assert "python -m vllm.entrypoints.openai.api_server" in command
    assert "VLLM_ENABLE_CUDA_COMPATIBILITY=1" in command
    assert "--model Qwen/Qwen3-14B-FP8" in command
    assert "--language-model-only" in command
    assert "--max-model-len 8192" in command
    assert "--reasoning-parser qwen3" in command
    assert "http://127.0.0.1:8000/v1/models" in command


def test_remote_setup_does_not_install_vllm() -> None:
    command = runpod_qwen_eval_once.remote_setup_command()

    assert "uv sync --extra dev" in command
    assert "uv pip install vllm" not in command
    assert "--pre vllm" not in command
    assert "wheels.vllm.ai/nightly" not in command


def test_remote_collect_command_uses_openai_compatible_local_server(tmp_path: Path) -> None:
    command = remote_collect_command(args(tmp_path))

    assert "OPENAI_BASE_URL=http://127.0.0.1:8000/v1" in command
    assert "OPENAI_API_KEY=runpod-local" in command
    assert "--tasks evals/leverage_smoke.jsonl" in command
    assert "--tasks evals/project_judgment_v0.jsonl" in command
    assert "--thinking-mode none" in command
    assert "--overwrite" in command


def test_remote_evaluate_command_scores_saved_predictions(tmp_path: Path) -> None:
    command = remote_evaluate_command(args(tmp_path))

    assert "llm.leverage.evaluate" in command
    assert "--predictions experiments/leverage/qwen.jsonl" in command
    assert "--summary-output experiments/leverage/qwen_summary.csv" in command


def test_rsync_from_remote_fetches_leverage_outputs(tmp_path: Path) -> None:
    command = rsync_from_remote_command(args(tmp_path), PodConnection("host", 2222))

    assert "root@host:/workspace/llm/experiments/leverage" in command
    assert command[-1] == f"{tmp_path}/experiments/"


def test_preflight_rejects_cost_above_first_spike_limit(tmp_path: Path) -> None:
    write_repo_shape(tmp_path)
    run_args = args(tmp_path)
    run_args.max_cost = 5.01

    with pytest.raises(ValueError, match="5 or less"):
        preflight(run_args)


def test_preflight_rejects_missing_task_file(tmp_path: Path) -> None:
    write_repo_shape(tmp_path)
    (tmp_path / "evals" / "project_judgment_v0.jsonl").unlink()

    with pytest.raises(FileNotFoundError, match="task file"):
        preflight(args(tmp_path))
