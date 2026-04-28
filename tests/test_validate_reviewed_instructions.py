import json
from pathlib import Path

from llm.leverage.validate_reviewed_instructions import validate_file


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def valid_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "id": "instr_test_001",
        "source_prompt_id": "lt_seed_test",
        "category": "resource_judgment",
        "messages": [
            {"role": "system", "content": "Answer concisely."},
            {"role": "user", "content": "Explain when to use hosted inference."},
            {"role": "assistant", "content": "Use hosted inference for evaluation before spending GPU time."},
        ],
        "review": {
            "status": "accepted_instruction",
            "author": "tester",
            "notes": "Good minimal row.",
        },
    }
    row.update(overrides)
    return row


def test_validate_file_reports_duplicate_ids(tmp_path: Path) -> None:
    dataset = tmp_path / "reviewed_instructions.jsonl"
    eval_dir = tmp_path / "evals"
    eval_dir.mkdir()
    write_jsonl(dataset, [valid_row(), valid_row()])

    errors = validate_file(dataset, eval_dir=eval_dir)

    assert any("duplicate id" in error for error in errors)


def test_validate_file_rejects_eval_prompt_reuse(tmp_path: Path) -> None:
    dataset = tmp_path / "reviewed_instructions.jsonl"
    eval_dir = tmp_path / "evals"
    eval_dir.mkdir()
    write_jsonl(eval_dir / "heldout.jsonl", [{"id": "eval_1", "prompt": "Do not reuse me."}])
    row = valid_row(
        messages=[
            {"role": "system", "content": "Answer concisely."},
            {"role": "user", "content": "Do not reuse me."},
            {"role": "assistant", "content": "This should fail."},
        ]
    )
    write_jsonl(dataset, [row])

    errors = validate_file(dataset, eval_dir=eval_dir)

    assert any("duplicates a held-out eval prompt" in error for error in errors)


def test_validate_file_rejects_secret_markers(tmp_path: Path) -> None:
    dataset = tmp_path / "reviewed_instructions.jsonl"
    eval_dir = tmp_path / "evals"
    eval_dir.mkdir()
    write_jsonl(dataset, [valid_row(source_prompt_id="OPENAI_API_KEY")])

    errors = validate_file(dataset, eval_dir=eval_dir)

    assert any("secret marker" in error for error in errors)
