import json
from pathlib import Path

from llm.leverage.collect_instructions import InstructionSeed
from llm.leverage.filter_instruction_outputs import (
    filter_row,
    filter_rows,
    write_candidate_jsonl,
    write_csv,
)


def seed() -> InstructionSeed:
    return InstructionSeed(
        id="lt_seed_test",
        category="resource_judgment",
        purpose="Generate a resource judgment example.",
        system_prompt="Answer concisely.",
        prompt="When should hosted inference be used?",
        output_format="short_answer",
        constraints=["mention cost"],
    )


def raw_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "source_prompt_id": "lt_seed_test",
        "category": "resource_judgment",
        "purpose": "Generate a resource judgment example.",
        "model": "teacher",
        "messages": [
            {"role": "system", "content": "Answer concisely."},
            {"role": "user", "content": "When should hosted inference be used?"},
            {"role": "assistant", "content": "Use hosted inference before renting GPUs."},
        ],
        "raw_response": "Use hosted inference before renting GPUs.",
        "output_format": "short_answer",
        "constraints": ["mention cost"],
        "review": {"status": "raw"},
    }
    row.update(overrides)
    return row


def test_filter_row_marks_clean_raw_output_for_judge() -> None:
    result = filter_row(
        raw_row(),
        seeds={"lt_seed_test": seed()},
        max_response_chars=200,
    )

    assert result.decision == "needs_judge"
    assert result.issues == []


def test_filter_row_rejects_unknown_seed() -> None:
    result = filter_row(
        raw_row(source_prompt_id="unknown"),
        seeds={"lt_seed_test": seed()},
        max_response_chars=200,
    )

    assert result.decision == "reject"
    assert "unknown_source_prompt_id" in result.issues


def test_filter_row_flags_overlong_response_without_rejecting() -> None:
    long_response = "x" * 201
    result = filter_row(
        raw_row(
            messages=[
                {"role": "system", "content": "Answer concisely."},
                {"role": "user", "content": "When should hosted inference be used?"},
                {"role": "assistant", "content": long_response},
            ],
            raw_response=long_response,
        ),
        seeds={"lt_seed_test": seed()},
        max_response_chars=200,
    )

    assert result.decision == "needs_judge"
    assert result.issues == ["response_too_long"]


def test_filter_row_rejects_secret_markers() -> None:
    result = filter_row(
        raw_row(raw_response="OPENAI_API_KEY should not appear"),
        seeds={"lt_seed_test": seed()},
        max_response_chars=200,
    )

    assert result.decision == "reject"
    assert "secret_marker:OPENAI_API_KEY" in result.issues


def test_write_csv_and_candidate_jsonl(tmp_path: Path) -> None:
    clean = raw_row()
    bad = raw_row(source_prompt_id="unknown")
    results = filter_rows(
        [(1, clean), (2, bad)],
        seeds={"lt_seed_test": seed()},
        max_response_chars=200,
    )
    csv_path = tmp_path / "filter.csv"
    candidates_path = tmp_path / "candidates.jsonl"

    write_csv(csv_path, results)
    write_candidate_jsonl(candidates_path, results)

    csv_lines = csv_path.read_text(encoding="utf-8").splitlines()
    assert csv_lines[0].startswith("line_number,source_prompt_id")
    assert "unknown_source_prompt_id" in csv_lines[2]
    candidates = [json.loads(line) for line in candidates_path.read_text(encoding="utf-8").splitlines()]
    assert candidates == [clean]
