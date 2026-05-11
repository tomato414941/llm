import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


DEFAULT_DATASET = "Open-Orca/OpenOrca"
DEFAULT_SPLIT = "train"
DEFAULT_OUTPUT = Path("tracks/leverage/sft/openorca.train.jsonl")
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def string_field(row: dict[str, Any], names: tuple[str, ...]) -> str:
    for name in names:
        value = row.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def messages_from_openorca_row(row: dict[str, Any], *, default_system_prompt: str) -> list[dict[str, str]]:
    system = string_field(row, ("system_prompt", "system", "instruction_context")) or default_system_prompt
    user = string_field(row, ("question", "instruction", "prompt", "user"))
    assistant = string_field(row, ("response", "answer", "output", "assistant"))
    if not user:
        raise ValueError("OpenOrca row is missing a question/instruction field")
    if not assistant:
        raise ValueError("OpenOrca row is missing a response/answer field")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


def convert_openorca_row(
    row: dict[str, Any],
    *,
    index: int,
    default_system_prompt: str,
    include_metadata: bool,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": f"openorca_{index:08d}",
        "messages": messages_from_openorca_row(row, default_system_prompt=default_system_prompt),
    }
    if include_metadata:
        record["source"] = {
            "dataset": DEFAULT_DATASET,
            "license": "mit",
            "row_id": row.get("id"),
        }
    return record


def write_openorca_sft(
    rows: Iterable[dict[str, Any]],
    output_path: Path,
    *,
    default_system_prompt: str,
    include_metadata: bool,
    limit: int | None,
    overwrite: bool,
) -> int:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", encoding="utf-8") as output_file:
        for index, row in enumerate(rows):
            if limit is not None and written >= limit:
                break
            record = convert_openorca_row(
                row,
                index=index,
                default_system_prompt=default_system_prompt,
                include_metadata=include_metadata,
            )
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    return written


def load_openorca_rows(dataset_name: str, split: str, *, streaming: bool) -> Iterable[dict[str, Any]]:
    from datasets import load_dataset

    return load_dataset(dataset_name, split=split, streaming=streaming)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--default-system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-streaming", action="store_true")
    parser.add_argument("--include-metadata", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_openorca_rows(args.dataset, args.split, streaming=not args.no_streaming)
    row_count = write_openorca_sft(
        rows,
        args.output,
        default_system_prompt=args.default_system_prompt,
        include_metadata=args.include_metadata,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    print(f"exported {row_count} OpenOrca rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
