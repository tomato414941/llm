import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from llm.leverage.collect_instructions import InstructionSeed, load_seeds, select_seeds
from llm.leverage.instruction_contract import build_instruction_contract
from llm.leverage.validate_reviewed_instructions import load_jsonl


DEFAULT_SEEDS = Path("tracks/leverage/prompts/instruction-seeds.jsonl")
DEFAULT_REVIEWED = Path("tracks/leverage/datasets/reviewed-instructions/bootstrap.jsonl")


@dataclass(frozen=True)
class DuplicateRow:
    duplicate_type: str
    seed_id: str
    duplicate_id: str
    prompt: str


def seed_user_prompt(seed: InstructionSeed) -> str:
    return build_instruction_contract(
        prompt=seed.prompt,
        output_format=seed.output_format,
        constraints=seed.constraints,
    )


def reviewed_user_prompts(path: Path) -> dict[str, str]:
    prompts: dict[str, str] = {}
    for line_number, row in load_jsonl(path):
        row_id = row.get("id")
        messages = row.get("messages")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError(f"{path}:{line_number}: id must be a non-empty string")
        if not isinstance(messages, list) or len(messages) < 2 or not isinstance(messages[1], dict):
            raise ValueError(f"{path}:{line_number}: messages must include a user message")
        user_prompt = messages[1].get("content")
        if not isinstance(user_prompt, str) or not user_prompt:
            raise ValueError(f"{path}:{line_number}: user message content must be a non-empty string")
        prompts[user_prompt] = row_id
    return prompts


def duplicate_rows(
    seeds: dict[str, InstructionSeed],
    *,
    reviewed_prompts: dict[str, str],
) -> list[DuplicateRow]:
    rows: list[DuplicateRow] = []
    seen_seed_prompts: dict[str, str] = {}
    for seed in seeds.values():
        prompt = seed_user_prompt(seed)
        previous_seed_id = seen_seed_prompts.get(prompt)
        if previous_seed_id is not None:
            rows.append(
                DuplicateRow(
                    duplicate_type="selected_seed",
                    seed_id=seed.id,
                    duplicate_id=previous_seed_id,
                    prompt=prompt,
                )
            )
        else:
            seen_seed_prompts[prompt] = seed.id

        reviewed_id = reviewed_prompts.get(prompt)
        if reviewed_id is not None:
            rows.append(
                DuplicateRow(
                    duplicate_type="reviewed",
                    seed_id=seed.id,
                    duplicate_id=reviewed_id,
                    prompt=prompt,
                )
            )
    return rows


def write_csv(path: Path, rows: list[DuplicateRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=["duplicate_type", "seed_id", "duplicate_id", "prompt"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "duplicate_type": row.duplicate_type,
                    "seed_id": row.seed_id,
                    "duplicate_id": row.duplicate_id,
                    "prompt": row.prompt,
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=Path, default=DEFAULT_SEEDS)
    parser.add_argument("--reviewed", type=Path, default=DEFAULT_REVIEWED)
    parser.add_argument("--seed-id", action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    selected = select_seeds(load_seeds(args.seeds), args.seed_id)
    rows = duplicate_rows(selected, reviewed_prompts=reviewed_user_prompts(args.reviewed))
    if args.output is not None:
        write_csv(args.output, rows)

    selected_seed_count = len(selected)
    duplicate_seed_count = len({row.seed_id for row in rows})
    print(f"selected_seed_count,{selected_seed_count}")
    print(f"duplicate_seed_count,{duplicate_seed_count}")
    print(f"duplicate_row_count,{len(rows)}")
    if args.output is not None:
        print(f"wrote,{args.output}")


if __name__ == "__main__":
    main()
