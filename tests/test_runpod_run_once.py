import importlib.util
from pathlib import Path
import sys

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "runpod" / "run_once.py"
SPEC = importlib.util.spec_from_file_location("runpod_run_once", MODULE_PATH)
assert SPEC is not None
runpod_run_once = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.path.insert(0, str(MODULE_PATH.parent))
try:
    SPEC.loader.exec_module(runpod_run_once)
finally:
    sys.path.pop(0)


def write_repo_shape(tmp_path: Path) -> None:
    for directory in ("src", "tests", "tracks/leverage/configs", "tracks/leverage/evals"):
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)
    for file_name in ("pyproject.toml", "uv.lock", "README.md", "AGENTS.md", "LICENSE"):
        (tmp_path / file_name).write_text("", encoding="utf-8")
    (tmp_path / "run_job.py").write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    (tmp_path / "runpod.pub").write_text("ssh-ed25519 public-key", encoding="utf-8")


def parse_with(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, extra: list[str]):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_once.py",
            "--repo-root",
            str(tmp_path),
            "--shared-runner",
            str(tmp_path / "run_job.py"),
            "--ssh-public-key",
            str(tmp_path / "runpod.pub"),
            "--output",
            "outputs/run",
            "--remote",
            "echo ok",
            *extra,
        ],
    )
    return runpod_run_once.parse_args()


def test_parse_args_defaults_to_runpod_pytorch_280_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_repo_shape(tmp_path)

    parsed = parse_with(monkeypatch, tmp_path, [])

    assert parsed.template_id == "runpod-torch-v280"
    assert parsed.image == "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"


def test_parse_args_explicit_image_disables_default_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_repo_shape(tmp_path)

    parsed = parse_with(monkeypatch, tmp_path, ["--image", "runpod/pytorch:test"])

    assert parsed.template_id is None
    assert parsed.image == "runpod/pytorch:test"


def test_preflight_requires_remote_and_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_repo_shape(tmp_path)
    run_args = parse_with(monkeypatch, tmp_path, [])
    run_args.remote = []

    with pytest.raises(ValueError, match="--remote"):
        runpod_run_once.preflight(run_args)

    run_args.remote = ["echo ok"]
    run_args.output = []
    with pytest.raises(ValueError, match="--output"):
        runpod_run_once.preflight(run_args)


def test_preflight_accepts_generic_run_shape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_repo_shape(tmp_path)

    runpod_run_once.preflight(parse_with(monkeypatch, tmp_path, []))


def test_remote_cuda_smoke_command_checks_cuda() -> None:
    command = runpod_run_once.remote_cuda_smoke_command()

    assert command.startswith("set -euo pipefail")
    assert 'cd "$REMOTE_DIR"' in command
    assert 'export PATH="$HOME/.local/bin:$PATH"' in command
    assert "torch.cuda.is_available()" in command
    assert "cuda_device=" in command


def test_remote_user_command_adds_shell_safety_and_uv_path() -> None:
    command = runpod_run_once.remote_user_command("echo ok")

    assert command == 'set -euo pipefail; cd "$REMOTE_DIR"; export PATH="$HOME/.local/bin:$PATH"; echo ok'


def test_build_shared_runner_command_preserves_llm_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_repo_shape(tmp_path)
    args = parse_with(
        monkeypatch,
        tmp_path,
        [
            "--dry-run",
            "--secure-cloud",
            "--gpu-type",
            "NVIDIA GeForce RTX 5090",
            "--allowed-cuda-version",
            "12.8",
            "--sync",
            "tracks/leverage/evals",
            "--local",
            "uv run pytest tests/test_leverage_harmony.py",
        ],
    )

    command = runpod_run_once.build_shared_runner_command(args)

    assert command[:2] == [sys.executable, str(tmp_path / "run_job.py")]
    assert command[command.index("--repo-root") + 1] == str(tmp_path)
    assert command[command.index("--template-id") + 1] == "runpod-torch-v280"
    assert command[command.index("--gpu-type") + 1] == "NVIDIA GeForce RTX 5090"
    assert command[command.index("--remote-dir") + 1] == "/workspace/llm"
    assert command[command.index("--timings-output") + 1] == "outputs/run/runpod-timings.json"
    assert "--dry-run" in command
    assert "--secure-cloud" in command
    assert command.count("--sync") == 8
    assert "tracks/leverage/evals" in command
    assert command.count("--remote") == 2
    assert command[command.index("--remote") + 1].startswith("set -euo pipefail")
    assert command[command.index("--allowed-cuda-version") + 1] == "12.8"
    assert command[command.index("--local") + 1] == "uv run pytest tests/test_leverage_harmony.py"


def test_build_shared_runner_command_uses_image_when_template_is_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_repo_shape(tmp_path)
    args = parse_with(monkeypatch, tmp_path, ["--image", "runpod/pytorch:test"])

    command = runpod_run_once.build_shared_runner_command(args)

    assert "--template-id" not in command
    assert command[command.index("--image") + 1] == "runpod/pytorch:test"


def test_load_api_key_reads_quoted_dotenv_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        'export RUNPOD_API_KEY="example-runpod-key" # local credential\n',
        encoding="utf-8",
    )

    assert runpod_run_once.load_api_key(env_file) == "example-runpod-key"


def test_load_api_key_prefers_injected_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "injected-runpod-key")
    env_file = tmp_path / ".env"
    env_file.write_text("RUNPOD_API_KEY=local-runpod-key\n", encoding="utf-8")

    assert runpod_run_once.load_api_key(env_file) == "injected-runpod-key"


@pytest.mark.parametrize(
    "content",
    [None, "", "OTHER=value\n", "RUNPOD_API_KEY=\n", "RUNPOD_API_KEY\n", "RUNPOD_API_KEY='   '\n"],
)
def test_load_api_key_reports_missing_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: str | None
) -> None:
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    if content is not None:
        env_file.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="RUNPOD_API_KEY"):
        runpod_run_once.load_api_key(env_file)


def test_main_delivers_dotenv_key_to_runner_from_repo_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    write_repo_shape(tmp_path)
    (tmp_path / ".env").write_text("RUNPOD_API_KEY=example-runpod-key\n", encoding="utf-8")
    (tmp_path / "run_job.py").write_text(
        "import os\n"
        "raise SystemExit(7 if os.environ.get('RUNPOD_API_KEY') == 'example-runpod-key' else 9)\n",
        encoding="utf-8",
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / ".env").write_text("RUNPOD_API_KEY=other-project-key\n", encoding="utf-8")
    monkeypatch.chdir(elsewhere)
    parse_with(monkeypatch, tmp_path, [])

    assert runpod_run_once.main() == 7


def test_main_reports_missing_runpod_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    write_repo_shape(tmp_path)
    parse_with(monkeypatch, tmp_path, [])

    assert runpod_run_once.main() == 2
    assert "RUNPOD_API_KEY" in capsys.readouterr().err


def test_main_dry_run_runs_without_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    write_repo_shape(tmp_path)
    (tmp_path / "run_job.py").write_text(
        "import sys\nraise SystemExit(0 if '--dry-run' in sys.argv else 9)\n",
        encoding="utf-8",
    )
    parse_with(monkeypatch, tmp_path, ["--dry-run"])

    assert runpod_run_once.main() == 0
