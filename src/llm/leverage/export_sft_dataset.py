import argparse
import json
from pathlib import Path
from typing import Any

from llm.leverage.validate_sft_candidates import load_jsonl, validate_file


def export_rows(rows: list[tuple[int, dict[str, Any]]], *, include_id: bool) -> list[dict[str, Any]]:
    exported: list[dict[str, Any]] = []
    for _line_number, row in rows:
        record: dict[str, Any] = {"messages": row["messages"]}
        if include_id:
            record = {"id": row["id"], **record}
        exported.append(record)
    return exported


def write_jsonl(path: Path, rows: list[dict[str, Any]], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def export_dataset(
    input_path: Path,
    output_path: Path,
    *,
    eval_dir: Path,
    include_id: bool,
    overwrite: bool,
) -> int:
    errors = validate_file(input_path, eval_dir=eval_dir)
    if errors:
        raise ValueError("\n".join(errors))
    rows = load_jsonl(input_path)
    exported = export_rows(rows, include_id=include_id)
    write_jsonl(output_path, exported, overwrite=overwrite)
    return len(exported)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("datasets/sft_candidates/leverage_sft_v0.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/sft/leverage_sft_v0.train.jsonl"),
    )
    parser.add_argument("--eval-dir", type=Path, default=Path("evals"))
    parser.add_argument("--include-id", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    row_count = export_dataset(
        args.input,
        args.output,
        eval_dir=args.eval_dir,
        include_id=args.include_id,
        overwrite=args.overwrite,
    )
    print(f"exported {row_count} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
