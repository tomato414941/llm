from argparse import Namespace
import importlib.util
from pathlib import Path
import sys

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "runpod" / "runpod_openai_api_once.py"
SPEC = importlib.util.spec_from_file_location("runpod_openai_api_once", MODULE_PATH)
assert SPEC is not None
runpod_openai_api_once = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runpod_openai_api_once)

ApiEndpoint = runpod_openai_api_once.ApiEndpoint
api_endpoint_from_pod = runpod_openai_api_once.api_endpoint_from_pod
cleanup_pod = runpod_openai_api_once.cleanup_pod
cleanup_named_pods = runpod_openai_api_once.cleanup_named_pods
local_collect_command = runpod_openai_api_once.local_collect_command
local_evaluate_command = runpod_openai_api_once.local_evaluate_command
parse_env = runpod_openai_api_once.parse_env
parse_port_mappings = runpod_openai_api_once.parse_port_mappings
parse_args = runpod_openai_api_once.parse_args
normalize_args = runpod_openai_api_once.normalize_args
preflight = runpod_openai_api_once.preflight
runpodctl_create_api_command = runpod_openai_api_once.runpodctl_create_api_command
server_args = runpod_openai_api_once.server_args
wait_for_models = runpod_openai_api_once.wait_for_models


def args(tmp_path: Path) -> Namespace:
    return Namespace(
        tasks=[Path("evals/leverage_smoke.jsonl"), Path("evals/project_judgment_v0.jsonl")],
        model="Qwen/Qwen3-14B-FP8",
        served_model_name="Qwen/Qwen3-14B-FP8",
        model_label="qwen3_14b_fp8",
        image="vllm/vllm-openai:test",
        output=Path("experiments/leverage/qwen.jsonl"),
        scores_output=Path("experiments/leverage/qwen_scores.csv"),
        summary_output=Path("experiments/leverage/qwen_summary.csv"),
        gpu_type="NVIDIA GeForce RTX 4090",
        gpu_count=1,
        max_cost=2.0,
        max_allowed_cost=2.0,
        pod_name_prefix="llm-openai-api-once",
        pod_name="llm-openai-api-once-20260427-000000",
        container_disk_size=80,
        volume_size=120,
        remote_volume="/workspace",
        vcpu=8,
        mem=29,
        server_port=8000,
        max_model_len=8192,
        max_tokens=512,
        temperature=0.0,
        reasoning_parser="qwen3",
        language_model_only=True,
        server_env=[("VLLM_ENABLE_CUDA_COMPATIBILITY", "1")],
        server_extra_args=[],
        thinking_mode="none",
        thinking_param="chat_template_kwargs",
        api_key="runpod-local",
        request_timeout_seconds=60.0,
        runpodctl="/home/dev/bin/runpodctl",
        secret_path=tmp_path / "runpod",
        repo_root=tmp_path,
        wait_seconds=900,
        api_wait_seconds=900,
        max_runtime_minutes=60,
        secure_cloud=False,
        allow_existing_pods=False,
        dry_run=True,
        keep_pod=False,
        keep_pod_on_failure=False,
    )


def write_repo_shape(tmp_path: Path) -> None:
    (tmp_path / "evals").mkdir()
    (tmp_path / "evals" / "leverage_smoke.jsonl").write_text("{}", encoding="utf-8")
    (tmp_path / "evals" / "project_judgment_v0.jsonl").write_text("{}", encoding="utf-8")


class RecordingRunner:
    dry_run = True

    def __init__(self) -> None:
        self.commands: list[list[str]] = []

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
        return runpod_openai_api_once.subprocess.CompletedProcess(command, 0, "", "")


def test_parse_args_defaults_to_qwen_on_vllm_latest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["runpod_openai_api_once.py"])

    parsed = normalize_args(parse_args())

    assert parsed.model == "Qwen/Qwen3-14B-FP8"
    assert parsed.model_label == "qwen3_14b_fp8"
    assert parsed.image == "vllm/vllm-openai:latest"
    assert parsed.tasks == [
        Path("evals/leverage_smoke.jsonl"),
        Path("evals/project_judgment_v0.jsonl"),
    ]
    assert parsed.server_port == 8000
    assert parsed.wait_seconds == 180
    assert parsed.api_wait_seconds == 600
    assert parsed.max_runtime_minutes == 20


def test_server_args_are_model_generic() -> None:
    run_args = args(Path("/tmp"))
    run_args.model = "provider/model"
    run_args.served_model_name = "served"
    run_args.reasoning_parser = ""

    command = server_args(run_args)

    assert command.startswith("--model provider/model")
    assert "vllm serve" not in command
    assert "--served-model-name served" in command
    assert "--host 0.0.0.0" in command
    assert "--port 8000" in command
    assert "--max-model-len 8192" in command
    assert "--language-model-only" in command
    assert "--api-key runpod-local" in command
    assert "--reasoning-parser" not in command


def test_server_args_include_custom_port_and_extra_vllm_args() -> None:
    run_args = args(Path("/tmp"))
    run_args.server_port = 8080
    run_args.server_extra_args = ["--disable-log-requests", "--uvicorn-log-level", "warning"]

    command = server_args(run_args)

    assert "--host 0.0.0.0" in command
    assert "--port 8080" in command
    assert "--disable-log-requests" in command
    assert "--uvicorn-log-level warning" in command


def test_runpodctl_create_api_command_exposes_http_without_ssh(tmp_path: Path) -> None:
    command = runpodctl_create_api_command(args(tmp_path))
    server_command = command[command.index("--args") + 1]

    assert command[:3] == ["/home/dev/bin/runpodctl", "create", "pod"]
    assert command[command.index("--imageName") + 1] == "vllm/vllm-openai:test"
    assert command[command.index("--ports") + 1] == "8000/http"
    assert command[command.index("--gpuType") + 1] == "NVIDIA GeForce RTX 4090"
    assert command[command.index("--cost") + 1] == "2.0"
    assert command[command.index("--mem") + 1] == "29"
    assert command[command.index("--env") + 1] == "VLLM_ENABLE_CUDA_COMPATIBILITY=1"
    assert server_command.startswith("--model Qwen/Qwen3-14B-FP8")
    assert "--api-key" in server_command
    assert "vllm serve" not in server_command
    assert "openssh-server" not in command
    assert "rsync" not in command


def test_cleanup_named_pods_removes_matching_active_pod(tmp_path: Path, monkeypatch) -> None:
    runner = RecordingRunner()
    runner.dry_run = False
    run_args = args(tmp_path)

    monkeypatch.setattr(
        runpod_openai_api_once,
        "active_pods",
        lambda _runner, _args: [
            {"ID": "target", "NAME": "llm-openai-api-once-20260427-000000"},
            {"ID": "other", "NAME": "unrelated"},
        ],
    )

    cleanup_named_pods(runner, run_args, None)

    assert runner.commands == [["/home/dev/bin/runpodctl", "remove", "pod", "target"]]


def test_runpodctl_create_api_command_keeps_runpod_http_port_in_sync_with_server_port(tmp_path: Path) -> None:
    run_args = args(tmp_path)
    run_args.server_port = 8080

    command = runpodctl_create_api_command(run_args)
    server_command = command[command.index("--args") + 1]

    assert command[command.index("--ports") + 1] == "8080/http"
    assert "--port 8080" in server_command
    assert "8000/http" not in command
    assert "8080/tcp" not in command


def test_runpodctl_create_api_command_keeps_server_extra_args_inside_container_args(tmp_path: Path) -> None:
    run_args = args(tmp_path)
    run_args.server_extra_args = ["--uvicorn-log-level", "warning"]

    command = runpodctl_create_api_command(run_args)
    server_command = command[command.index("--args") + 1]

    assert "--uvicorn-log-level warning" in server_command
    assert "--uvicorn-log-level" not in command[:-1]
    assert "warning" not in command[:-1]


def test_parse_port_mappings_reads_runpod_public_http_mapping() -> None:
    ports = "10.0.0.1:12345->8000 (prv,http),104.1.2.3:45678->8000 (pub,http)"

    assert parse_port_mappings(ports) == [
        ("10.0.0.1", 12345, 8000, "prv,http"),
        ("104.1.2.3", 45678, 8000, "pub,http"),
    ]


def test_api_endpoint_from_pod_uses_public_http_mapping() -> None:
    pod = {"PORTS": "10.0.0.1:12345->8000 (prv,http),104.1.2.3:45678->8000 (pub,http)"}

    endpoint = api_endpoint_from_pod(pod, 8000)

    assert endpoint == ApiEndpoint("http://104.1.2.3:45678/v1", "104.1.2.3", 45678)


def test_api_endpoint_from_pod_falls_back_to_runpod_proxy_url() -> None:
    pod = {"ID": "pod123", "STATUS": "RUNNING", "PORTS": ""}

    endpoint = api_endpoint_from_pod(pod, 8000)

    assert endpoint == ApiEndpoint(
        "https://pod123-8000.proxy.runpod.net/v1",
        "pod123-8000.proxy.runpod.net",
        443,
    )


def test_api_endpoint_from_pod_ignores_public_tcp_mapping_without_http_label() -> None:
    pod = {"PORTS": "104.1.2.3:45678->8000 (pub,tcp)"}

    assert api_endpoint_from_pod(pod, 8000) is None


def test_local_collect_command_targets_endpoint_and_model(tmp_path: Path) -> None:
    command = local_collect_command(args(tmp_path), ApiEndpoint("http://host:8000/v1", "host", 8000))

    assert command[:4] == [
        "env",
        "OPENAI_BASE_URL=http://host:8000/v1",
        "OPENAI_API_KEY=runpod-local",
        "uv",
    ]
    assert "llm.leverage.collect_openai" in command
    assert "--model" in command
    assert command[command.index("--model") + 1] == "Qwen/Qwen3-14B-FP8"
    assert "--thinking-mode" in command
    assert command[command.index("--thinking-mode") + 1] == "none"


def test_local_evaluate_command_scores_saved_predictions(tmp_path: Path) -> None:
    command = local_evaluate_command(args(tmp_path))

    assert "llm.leverage.evaluate" in command
    assert command[command.index("--predictions") + 1] == "experiments/leverage/qwen.jsonl"
    assert command[command.index("--summary-output") + 1] == "experiments/leverage/qwen_summary.csv"


def test_wait_for_models_sends_authorization_header(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_headers: list[str] = []

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(probe, timeout):
        seen_headers.append(probe.get_header("Authorization"))
        assert timeout == 5
        return Response()

    monkeypatch.setattr(runpod_openai_api_once.request, "urlopen", fake_urlopen)

    wait_for_models(ApiEndpoint("http://host:8000/v1", "host", 8000), 1, "runpod-local")

    assert seen_headers == ["Bearer runpod-local"]


def test_parse_env_requires_name_value_pairs() -> None:
    assert parse_env(["A=1", "B=two"]) == [("A", "1"), ("B", "two")]
    with pytest.raises(ValueError, match="NAME=VALUE"):
        parse_env(["BAD"])


def test_preflight_rejects_cost_over_allowed_limit(tmp_path: Path) -> None:
    write_repo_shape(tmp_path)
    run_args = args(tmp_path)
    run_args.max_cost = 2.01

    with pytest.raises(ValueError, match="max-allowed-cost"):
        preflight(run_args)


def test_preflight_rejects_missing_task_file(tmp_path: Path) -> None:
    write_repo_shape(tmp_path)
    (tmp_path / "evals" / "project_judgment_v0.jsonl").unlink()

    with pytest.raises(FileNotFoundError, match="task file"):
        preflight(args(tmp_path))


def test_cleanup_policy_removes_pod_on_success_and_failure(tmp_path: Path) -> None:
    runner = RecordingRunner()

    cleanup_pod(runner, args(tmp_path), "pod-success", success=True)
    cleanup_pod(runner, args(tmp_path), "pod-failure", success=False)

    assert runner.commands == [
        ["/home/dev/bin/runpodctl", "remove", "pod", "pod-success"],
        ["/home/dev/bin/runpodctl", "remove", "pod", "pod-failure"],
    ]


def test_cleanup_policy_keeps_failure_when_requested(tmp_path: Path) -> None:
    runner = RecordingRunner()
    run_args = args(tmp_path)
    run_args.keep_pod_on_failure = True

    cleanup_pod(runner, run_args, "pod-failure", success=False)

    assert runner.commands == []
