import json
from pathlib import Path

import pytest

from llm.leverage.import_openorca import (
    convert_openorca_row,
    messages_from_openorca_row,
    write_openorca_sft,
)


def test_messages_from_openorca_row_uses_question_and_response() -> None:
    messages = messages_from_openorca_row(
        {
            "system_prompt": "You are precise.",
            "question": "What is 2 + 2?",
            "response": "4",
        },
        default_system_prompt="You are helpful.",
    )

    assert messages == [
        {"role": "system", "content": "You are precise."},
        {"role": "user", "content": "What is 2 + 2?"},
        {"role": "assistant", "content": "4"},
    ]


def test_messages_from_openorca_row_falls_back_to_default_system_prompt() -> None:
    messages = messages_from_openorca_row(
        {
            "question": "What is 2 + 2?",
            "response": "4",
        },
        default_system_prompt="You are helpful.",
    )

    assert messages[0] == {"role": "system", "content": "You are helpful."}


def test_messages_from_openorca_row_requires_user_and_assistant_text() -> None:
    with pytest.raises(ValueError, match="question/instruction"):
        messages_from_openorca_row({"response": "4"}, default_system_prompt="sys")

    with pytest.raises(ValueError, match="response/answer"):
        messages_from_openorca_row({"question": "q"}, default_system_prompt="sys")


def test_convert_openorca_row_can_include_metadata() -> None:
    record = convert_openorca_row(
        {
            "id": "source-row",
            "question": "What is 2 + 2?",
            "response": "4",
        },
        index=3,
        default_system_prompt="sys",
        include_metadata=True,
    )

    assert record["id"] == "openorca_00000003"
    assert record["source"] == {
        "dataset": "Open-Orca/OpenOrca",
        "license": "mit",
        "row_id": "source-row",
    }


def test_write_openorca_sft_writes_jsonl_with_limit(tmp_path: Path) -> None:
    output = tmp_path / "openorca.train.jsonl"
    rows = [
        {"id": "a", "question": "q1", "response": "a1"},
        {"id": "b", "question": "q2", "response": "a2"},
    ]

    count = write_openorca_sft(
        rows,
        output,
        default_system_prompt="sys",
        include_metadata=False,
        limit=1,
        overwrite=False,
    )

    exported = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert count == 1
    assert exported == [
        {
            "id": "openorca_00000000",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
            ],
        }
    ]


def test_write_openorca_sft_skips_invalid_rows(tmp_path: Path) -> None:
    output = tmp_path / "openorca.train.jsonl"
    rows = [
        {"id": "bad", "question": "q"},
        {"id": "good", "question": "q", "response": "a"},
    ]

    count = write_openorca_sft(
        rows,
        output,
        default_system_prompt="sys",
        include_metadata=False,
        limit=None,
        overwrite=False,
    )

    exported = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert count == 1
    assert exported[0]["id"] == "openorca_00000001"


def test_write_openorca_sft_refuses_existing_output_without_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "openorca.train.jsonl"
    output.write_text("existing\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        write_openorca_sft(
            [],
            output,
            default_system_prompt="sys",
            include_metadata=True,
            limit=None,
            overwrite=False,
        )
