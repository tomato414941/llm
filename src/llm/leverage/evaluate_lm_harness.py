import argparse
from datetime import datetime, timezone
import json
import shutil
import subprocess
import time
from pathlib import Path

from llm.leverage.evaluate_sft_adapter import DEFAULT_CONFIG, config_defaults


DEFAULT_TASK = "ifeval"
DEFAULT_OUTPUT_ROOT = Path("outputs/leverage-lm-harness")
# Generation timing is observational: lm-evaluation-harness may change or hide
# this progress line, so missing generation_seconds is not an error.
GENERATE_MARKER = "Running generate_until requests:"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def model_args(
    *,
    base_model: str,
    adapter_dir: Path | None,
    enable_thinking: bool | None,
    think_end_token: str | None,
) -> str:
    args = [f"pretrained={base_model}", "trust_remote_code=True"]
    if adapter_dir is not None:
        args.append(f"peft={adapter_dir}")
    if enable_thinking is not None:
        args.append(f"enable_thinking={enable_thinking}")
    if think_end_token is not None:
        args.append(f"think_end_token={think_end_token}")
    return ",".join(args)


def build_lm_eval_command(
    *,
    base_model: str,
    adapter_dir: Path | None,
    tasks: list[str],
    output_path: Path,
    device: str,
    batch_size: str,
    apply_chat_template: bool,
    limit: int | None,
    log_samples: bool,
    enable_thinking: bool | None,
    think_end_token: str | None,
    max_gen_toks: int | None = None,
) -> list[str]:
    command = [
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        model_args(
            base_model=base_model,
            adapter_dir=adapter_dir,
            enable_thinking=enable_thinking,
            think_end_token=think_end_token,
        ),
        "--tasks",
        ",".join(tasks),
        "--device",
        device,
        "--batch_size",
        batch_size,
        "--output_path",
        str(output_path),
    ]
    if apply_chat_template:
        command.append("--apply_chat_template")
    if limit is not None:
        command.extend(["--limit", str(limit)])
    if log_samples:
        command.append("--log_samples")
    if max_gen_toks is not None:
        command.extend(["--gen_kwargs", f"max_gen_toks={max_gen_toks}"])
    return command


def shell_join(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def update_generation_timing(line: str, timing: dict[str, object], started: float) -> None:
    if GENERATE_MARKER not in line:
        return
    elapsed = round(time.monotonic() - started, 3)
    if timing["generation_started_after_seconds"] is None:
        timing["generation_started_after_seconds"] = elapsed
    timing["generation_last_seen_after_seconds"] = elapsed


def write_timing(path: Path, timing: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(timing, indent=2) + "\n", encoding="utf-8")


def run_command_with_timing(command: list[str], timing_output: Path | None) -> None:
    started = time.monotonic()
    timing: dict[str, object] = {
        "command": shell_join(command),
        "started_at": utc_now(),
        "finished_at": None,
        "returncode": None,
        "elapsed_seconds": None,
        "generation_started_after_seconds": None,
        "generation_last_seen_after_seconds": None,
        "generation_seconds": None,
    }
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        update_generation_timing(line, timing, started)
    returncode = process.wait()
    elapsed = round(time.monotonic() - started, 3)
    timing["finished_at"] = utc_now()
    timing["returncode"] = returncode
    timing["elapsed_seconds"] = elapsed
    generation_started = timing["generation_started_after_seconds"]
    generation_last_seen = timing["generation_last_seen_after_seconds"]
    if isinstance(generation_started, float) and isinstance(generation_last_seen, float):
        timing["generation_seconds"] = round(generation_last_seen - generation_started, 3)
    if timing_output is not None:
        write_timing(timing_output, timing)
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)


def run_lm_harness(
    *,
    base_model: str,
    adapter_dir: Path,
    output_root: Path,
    tasks: list[str],
    device: str,
    batch_size: str,
    apply_chat_template: bool,
    limit: int | None,
    log_samples: bool,
    enable_thinking: bool | None,
    think_end_token: str | None,
    variant: str,
    dry_run: bool,
    timing_output: Path | None = None,
    max_gen_toks: int | None = None,
) -> list[str]:
    lines: list[str] = []
    target_adapter = adapter_dir if variant == "adapter" else None
    output_path = output_root / variant
    command = build_lm_eval_command(
        base_model=base_model,
        adapter_dir=target_adapter,
        tasks=tasks,
        output_path=output_path,
        device=device,
        batch_size=batch_size,
        apply_chat_template=apply_chat_template,
        limit=limit,
        log_samples=log_samples,
        enable_thinking=enable_thinking,
        think_end_token=think_end_token,
        max_gen_toks=max_gen_toks,
    )
    lines.append(f"{variant}: {shell_join(command)}")
    if dry_run:
        return lines
    if variant == "adapter" and not adapter_dir.exists():
        raise FileNotFoundError(f"adapter directory does not exist: {adapter_dir}")
    if shutil.which("lm_eval") is None:
        raise RuntimeError("lm_eval is not installed; install lm-evaluation-harness in the run environment")
    output_path.mkdir(parents=True, exist_ok=True)
    run_command_with_timing(command, timing_output)
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--base-model")
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--task", action="append", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--log-samples", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--think-end-token")
    parser.add_argument("--max-gen-toks", type=int)
    parser.add_argument("--variant", choices=("base", "adapter"), required=True)
    parser.add_argument("--apply-chat-template", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timing-output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    defaults = config_defaults(args.config)
    base_model = args.base_model if args.base_model is not None else defaults["base_model"]
    adapter_dir = args.adapter_dir if args.adapter_dir is not None else defaults["adapter_dir"]
    tasks = args.task if args.task is not None else [DEFAULT_TASK]
    for line in run_lm_harness(
        base_model=base_model,
        adapter_dir=adapter_dir,
        output_root=args.output_root,
        tasks=tasks,
        device=args.device,
        batch_size=args.batch_size,
        apply_chat_template=args.apply_chat_template,
        limit=args.limit,
        log_samples=args.log_samples,
        enable_thinking=args.enable_thinking,
        think_end_token=args.think_end_token,
        max_gen_toks=args.max_gen_toks,
        variant=args.variant,
        dry_run=args.dry_run,
        timing_output=args.timing_output,
    ):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
