import argparse
import shutil
import subprocess
from pathlib import Path

from llm.leverage.evaluate_sft_adapter import DEFAULT_CONFIG, config_defaults


DEFAULT_TASK = "ifeval"
DEFAULT_OUTPUT_ROOT = Path("outputs/leverage-lm-harness")


def model_args(*, base_model: str, adapter_dir: Path | None) -> str:
    args = [f"pretrained={base_model}", "trust_remote_code=True"]
    if adapter_dir is not None:
        args.append(f"peft={adapter_dir}")
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
) -> list[str]:
    command = [
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        model_args(base_model=base_model, adapter_dir=adapter_dir),
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
    return command


def shell_join(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def selected_runs(run: str) -> list[str]:
    if run == "both":
        return ["base", "adapter"]
    return [run]


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
    run: str,
    dry_run: bool,
) -> list[str]:
    lines: list[str] = []
    for target in selected_runs(run):
        target_adapter = adapter_dir if target == "adapter" else None
        output_path = output_root / target
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
        )
        lines.append(f"{target}: {shell_join(command)}")
        if dry_run:
            continue
        if target == "adapter" and not adapter_dir.exists():
            raise FileNotFoundError(f"adapter directory does not exist: {adapter_dir}")
        if shutil.which("lm_eval") is None:
            raise RuntimeError("lm_eval is not installed; install lm-evaluation-harness in the run environment")
        output_path.mkdir(parents=True, exist_ok=True)
        subprocess.run(command, check=True)
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
    parser.add_argument("--run", choices=("base", "adapter", "both"), default="both")
    parser.add_argument("--apply-chat-template", action=argparse.BooleanOptionalAction, default=True)
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
        run=args.run,
        dry_run=args.dry_run,
    ):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
