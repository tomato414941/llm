import argparse
from pathlib import Path
from typing import Any

from llm.config import load_toml
from llm.leverage.export_reviewed_instructions import export_dataset
from llm.leverage.validate_reviewed_instructions import load_jsonl


DEFAULT_CONFIG = Path("tracks/leverage/configs/leverage-sft-smoke.toml")


def require_table(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"config section [{name}] must be a table")
    return value


def require_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty path string")
    return Path(value)


def require_int(value: Any, label: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def preflight(config_path: Path, *, overwrite: bool) -> list[str]:
    config = load_toml(config_path)
    experiment = require_table(config, "experiment")
    data = require_table(config, "data")
    method = require_table(config, "method")
    runpod = require_table(config, "runpod")
    outputs = require_table(config, "outputs")

    if experiment.get("track") != "leverage":
        raise ValueError("experiment.track must be 'leverage'")
    if experiment.get("status") != "planned":
        raise ValueError("experiment.status must be 'planned' until training is implemented")

    reviewed_path = require_path(data.get("reviewed-instructions"), "data.reviewed-instructions")
    train_export_path = require_path(data.get("train_export"), "data.train_export")
    eval_tasks = data.get("eval_tasks")
    if not isinstance(eval_tasks, list) or not eval_tasks:
        raise ValueError("data.eval_tasks must be a non-empty list")
    eval_task_paths = [require_path(path, "data.eval_tasks[]") for path in eval_tasks]

    max_train_examples = require_int(method.get("max_train_examples"), "method.max_train_examples")
    max_runtime_minutes = require_int(method.get("max_runtime_minutes"), "method.max_runtime_minutes")
    batch_size = require_int(method.get("batch_size"), "method.batch_size")
    gradient_accumulation_steps = require_int(
        method.get("gradient_accumulation_steps", 1),
        "method.gradient_accumulation_steps",
    )
    log_every_steps = require_int(method.get("log_every_steps", 50), "method.log_every_steps")
    if max_train_examples <= 0:
        raise ValueError("method.max_train_examples must be positive")
    if batch_size <= 0:
        raise ValueError("method.batch_size must be positive")
    if gradient_accumulation_steps <= 0:
        raise ValueError("method.gradient_accumulation_steps must be positive")
    if log_every_steps <= 0:
        raise ValueError("method.log_every_steps must be positive")
    if max_runtime_minutes > 60:
        raise ValueError("method.max_runtime_minutes must be <= 60 for the smoke run")

    if require_bool(runpod.get("required"), "runpod.required"):
        raise ValueError("runpod.required must be false for local preflight")
    if not require_bool(runpod.get("cleanup_required"), "runpod.cleanup_required"):
        raise ValueError("runpod.cleanup_required must be true")

    output_root = require_path(outputs.get("root"), "outputs.root")
    if output_root.is_absolute() or output_root.parts[:1] != ("outputs",):
        raise ValueError("outputs.root must be a relative path under outputs/")

    if not reviewed_path.exists():
        raise FileNotFoundError(f"reviewed instruction file not found: {reviewed_path}")
    for eval_task_path in eval_task_paths:
        if not eval_task_path.exists():
            raise FileNotFoundError(f"eval task file not found: {eval_task_path}")

    row_count = export_dataset(
        reviewed_path,
        train_export_path,
        eval_dir=Path("tracks/leverage/evals"),
        include_id=True,
        overwrite=overwrite,
    )
    if row_count > max_train_examples:
        raise ValueError(
            f"exported {row_count} rows, exceeding max_train_examples={max_train_examples}"
        )
    exported_rows = load_jsonl(train_export_path)
    if len(exported_rows) != row_count:
        raise ValueError("exported row count did not match written training JSONL")

    return [
        f"validated reviewed instructions: {reviewed_path}",
        f"exported training rows: {row_count} -> {train_export_path}",
        f"checked eval tasks: {len(eval_task_paths)}",
        f"checked local output root: {output_root}",
        "runpod.required=false",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for line in preflight(args.config, overwrite=args.overwrite):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
