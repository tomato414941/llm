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
        "capability": "tool_use",
        "task_shape": "explanation",
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


def test_validate_file_rejects_duplicate_user_prompts(tmp_path: Path) -> None:
    dataset = tmp_path / "reviewed_instructions.jsonl"
    eval_dir = tmp_path / "evals"
    eval_dir.mkdir()
    write_jsonl(
        dataset,
        [
            valid_row(id="instr_test_001"),
            valid_row(
                id="instr_test_002",
                source_prompt_id="lt_seed_other",
                messages=[
                    {"role": "system", "content": "Answer briefly."},
                    {"role": "user", "content": "Explain when to use hosted inference."},
                    {"role": "assistant", "content": "Use it for cheap evaluation."},
                ],
            ),
        ],
    )

    errors = validate_file(dataset, eval_dir=eval_dir)

    assert any("user prompt duplicates reviewed row instr_test_001" in error for error in errors)


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


def test_validate_file_does_not_reject_task_solving_text(tmp_path: Path) -> None:
    dataset = tmp_path / "reviewed_instructions.jsonl"
    eval_dir = tmp_path / "evals"
    eval_dir.mkdir()
    write_jsonl(
        dataset,
        [
            valid_row(
                messages=[
                    {"role": "system", "content": "Answer concisely."},
                    {"role": "user", "content": "Explain evaluation."},
                    {"role": "assistant", "content": "Measure task-solving behavior on held-out data."},
                ]
            )
        ],
    )

    errors = validate_file(dataset, eval_dir=eval_dir)

    assert errors == []


def test_validate_file_rejects_incomplete_candidate_provenance(tmp_path: Path) -> None:
    dataset = tmp_path / "reviewed_instructions.jsonl"
    eval_dir = tmp_path / "evals"
    eval_dir.mkdir()
    row = valid_row()
    row["review"] = {
        "status": "accepted_instruction",
        "author": "tester",
        "source": "edited_candidate",
        "notes": "Missing generator and judge provenance.",
    }
    write_jsonl(dataset, [row])

    errors = validate_file(dataset, eval_dir=eval_dir)

    assert any("review.generator_model is required" in error for error in errors)
    assert any("review.judge_model is required" in error for error in errors)
    assert any("review.judge_decision is required" in error for error in errors)


def test_validate_file_rejects_secret_like_openai_key(tmp_path: Path) -> None:
    dataset = tmp_path / "reviewed_instructions.jsonl"
    eval_dir = tmp_path / "evals"
    eval_dir.mkdir()
    write_jsonl(
        dataset,
        [
            valid_row(
                messages=[
                    {"role": "system", "content": "Answer concisely."},
                    {"role": "user", "content": "Explain secret handling."},
                    {"role": "assistant", "content": "Never commit sk-abcdefghijklmnopqrstuvwxyz123456."},
                ]
            )
        ],
    )

    errors = validate_file(dataset, eval_dir=eval_dir)

    assert any("secret marker" in error for error in errors)


def test_validate_file_rejects_unknown_task_shape(tmp_path: Path) -> None:
    dataset = tmp_path / "reviewed_instructions.jsonl"
    eval_dir = tmp_path / "evals"
    eval_dir.mkdir()
    write_jsonl(dataset, [valid_row(task_shape="legacy_shape")])

    errors = validate_file(dataset, eval_dir=eval_dir)

    assert any("task_shape must be one of" in error for error in errors)
