import csv
import json
from pathlib import Path

from llm.leverage.collect_instructions import InstructionSeed
from llm.leverage.instruction_contract import build_instruction_contract
from llm.leverage.report_instruction_seed_duplicates import (
    duplicate_rows,
    reviewed_user_prompts,
    seed_user_prompt,
    write_csv,
)


def seed(seed_id: str, prompt: str) -> InstructionSeed:
    return InstructionSeed(
        id=seed_id,
        capability="reasoning",
        purpose="Generate a reasoning example.",
        system_prompt="Answer directly.",
        prompt=prompt,
        output_format="short_answer",
        constraints=["be concise"],
    )


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def reviewed_row(row_id: str, user_prompt: str) -> dict[str, object]:
    return {
        "id": row_id,
        "source_prompt_id": "lt_seed_existing",
        "capability": "reasoning",
        "task_shape": "direct_answer",
        "messages": [
            {"role": "system", "content": "Answer directly."},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": "Answer."},
        ],
        "review": {
            "status": "accepted_instruction",
            "author": "test",
            "notes": "test row",
            "source": "manual",
        },
    }


def test_duplicate_rows_reports_selected_seed_duplicates() -> None:
    seeds = {
        "lt_seed_001": seed("lt_seed_001", "Why did the deploy fail?"),
        "lt_seed_002": seed("lt_seed_002", "Why did the deploy fail?"),
    }

    rows = duplicate_rows(seeds, reviewed_prompts={})

    assert [(row.duplicate_type, row.seed_id, row.duplicate_id) for row in rows] == [
        ("selected_seed", "lt_seed_002", "lt_seed_001")
    ]


def test_duplicate_rows_reports_reviewed_prompt_overlap(tmp_path: Path) -> None:
    selected_seed = seed("lt_seed_001", "Why did the deploy fail?")
    reviewed_path = tmp_path / "reviewed.jsonl"
    write_jsonl(reviewed_path, [reviewed_row("instr_001", seed_user_prompt(selected_seed))])

    rows = duplicate_rows(
        {"lt_seed_001": selected_seed},
        reviewed_prompts=reviewed_user_prompts(reviewed_path),
    )

    assert [(row.duplicate_type, row.seed_id, row.duplicate_id) for row in rows] == [
        ("reviewed", "lt_seed_001", "instr_001")
    ]


def test_seed_user_prompt_uses_full_instruction_contract() -> None:
    selected_seed = seed("lt_seed_001", "Why did the deploy fail?")

    assert seed_user_prompt(selected_seed) == build_instruction_contract(
        prompt="Why did the deploy fail?",
        output_format="short_answer",
        constraints=["be concise"],
    )


def test_write_csv_writes_stable_columns(tmp_path: Path) -> None:
    output = tmp_path / "duplicates.csv"
    rows = duplicate_rows(
        {
            "lt_seed_001": seed("lt_seed_001", "Why did the deploy fail?"),
            "lt_seed_002": seed("lt_seed_002", "Why did the deploy fail?"),
        },
        reviewed_prompts={},
    )

    write_csv(output, rows)

    with output.open(encoding="utf-8", newline="") as input_file:
        csv_rows = list(csv.DictReader(input_file))
    assert csv_rows == [
        {
            "duplicate_type": "selected_seed",
            "seed_id": "lt_seed_002",
            "duplicate_id": "lt_seed_001",
            "prompt": seed_user_prompt(seed("lt_seed_002", "Why did the deploy fail?")),
        }
    ]
