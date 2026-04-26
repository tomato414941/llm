import argparse
import csv
from pathlib import Path
from typing import Any

import torch


SCALING_FIELDS = (
    "run_id",
    "checkpoint_path",
    "tokens_path",
    "checkpoint_step",
    "tokens_seen",
    "parameter_count",
    "block_size",
    "embedding_dim",
    "num_heads",
    "num_layers",
    "dropout",
    "max_iters",
    "batch_size",
    "learning_rate",
    "validation_loss",
    "validation_perplexity",
    "prompt_id",
    "prompt",
    "seed",
    "samples",
    "max_new_tokens",
    "temperature",
    "top_k",
    "note",
)


def load_checkpoint_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must be a dictionary")
    return payload


def metadata_value(metadata: object, key: str) -> object:
    if isinstance(metadata, dict):
        return metadata.get(key, "")
    return ""


def checkpoint_run_id(path: Path, checkpoint: dict[str, Any]) -> str:
    run_id = metadata_value(checkpoint.get("metadata", {}), "run_id")
    if isinstance(run_id, str) and run_id:
        return run_id
    return path.stem


def read_summary_rows(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as summary_file:
        return list(csv.DictReader(summary_file))


def latest_summary_row(rows: list[dict[str, str]], checkpoint_path: Path) -> dict[str, str] | None:
    checkpoint_text = str(checkpoint_path)
    checkpoint_name = checkpoint_path.name
    matches = [
        row
        for row in rows
        if row.get("checkpoint_path") == checkpoint_text
        or Path(row.get("checkpoint_path", "")).name == checkpoint_name
    ]
    if not matches:
        return None
    return matches[-1]


def scaling_row(
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    summary_row: dict[str, str] | None,
    note: str,
) -> dict[str, object]:
    config = checkpoint.get("config", {})
    if not isinstance(config, dict):
        config = {}
    metadata = checkpoint.get("metadata", {})
    row = {
        "run_id": checkpoint_run_id(checkpoint_path, checkpoint),
        "checkpoint_path": str(checkpoint_path),
        "tokens_path": metadata_value(metadata, "tokens"),
        "checkpoint_step": checkpoint.get("step", ""),
        "tokens_seen": checkpoint.get("tokens_seen", ""),
        "parameter_count": metadata_value(metadata, "parameter_count"),
        "block_size": config.get("block_size", ""),
        "embedding_dim": config.get("embedding_dim", ""),
        "num_heads": config.get("num_heads", ""),
        "num_layers": config.get("num_layers", ""),
        "dropout": config.get("dropout", ""),
        "max_iters": metadata_value(metadata, "max_iters"),
        "batch_size": metadata_value(metadata, "batch_size"),
        "learning_rate": metadata_value(metadata, "learning_rate"),
        "validation_loss": "",
        "validation_perplexity": "",
        "prompt_id": "",
        "prompt": "",
        "seed": "",
        "samples": "",
        "max_new_tokens": "",
        "temperature": "",
        "top_k": "",
        "note": note,
    }
    if summary_row is not None:
        for field in (
            "validation_loss",
            "validation_perplexity",
            "prompt_id",
            "prompt",
            "seed",
            "samples",
            "max_new_tokens",
            "temperature",
            "top_k",
        ):
            row[field] = summary_row.get(field, "")
    return row


def read_scaling_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as scaling_file:
        return list(csv.DictReader(scaling_file))


def upsert_scaling_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = read_scaling_rows(path)
    run_id = str(row["run_id"])
    output_rows = [existing for existing in rows if existing.get("run_id") != run_id]
    output_rows.append({field: row.get(field, "") for field in SCALING_FIELDS})
    with path.open("w", encoding="utf-8", newline="") as scaling_file:
        writer = csv.DictWriter(scaling_file, fieldnames=SCALING_FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--output", type=Path, default=Path("experiments/summaries/scaling.csv"))
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    checkpoint = load_checkpoint_payload(args.checkpoint)
    summary_row = latest_summary_row(read_summary_rows(args.summary), args.checkpoint)
    row = scaling_row(
        checkpoint_path=args.checkpoint,
        checkpoint=checkpoint,
        summary_row=summary_row,
        note=args.note,
    )
    upsert_scaling_row(args.output, row)
    print(f"scaling row written to {args.output}")


if __name__ == "__main__":
    main()
