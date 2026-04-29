import json
from pathlib import Path
from typing import Any

from llm.leverage.validate_reviewed_instructions import validate_file

DATASET_PATH = Path("tracks/leverage/datasets/reviewed-instructions/bootstrap.jsonl")
REQUIRED_ROW_FIELDS = {"id", "source_prompt_id", "category", "messages", "review"}
REQUIRED_ROLES = ["system", "user", "assistant"]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        assert line.strip(), f"{path}:{line_number} must not be blank"
        payload = json.loads(line)
        assert isinstance(payload, dict), f"{path}:{line_number} must be a JSON object"
        rows.append(payload)
    return rows


def test_reviewed_instruction_dataset_has_reviewed_chat_schema() -> None:
    rows = load_jsonl(DATASET_PATH)
    assert len(rows) == 17

    seen_ids: set[str] = set()
    for row in rows:
        assert REQUIRED_ROW_FIELDS <= row.keys()
        assert isinstance(row["id"], str) and row["id"]
        assert row["id"] not in seen_ids
        seen_ids.add(row["id"])
        assert isinstance(row["source_prompt_id"], str) and row["source_prompt_id"].startswith("lt_seed_")
        assert isinstance(row["category"], str) and row["category"]

        messages = row["messages"]
        assert isinstance(messages, list)
        assert [message["role"] for message in messages] == REQUIRED_ROLES
        for message in messages:
            assert isinstance(message["content"], str) and message["content"]

        review = row["review"]
        assert isinstance(review, dict)
        assert review.get("status") == "accepted_instruction"
        assert review.get("author") == "codex"
        assert isinstance(review.get("notes"), str) and review["notes"]


def test_reviewed_instructions_do_not_reuse_eval_prompts_verbatim() -> None:
    dataset_prompts = {row["messages"][1]["content"] for row in load_jsonl(DATASET_PATH)}
    eval_prompts: set[str] = set()
    for eval_path in Path("tracks/leverage/evals").glob("*.jsonl"):
        for row in load_jsonl(eval_path):
            prompt = row.get("prompt")
            if isinstance(prompt, str):
                eval_prompts.add(prompt)

    assert dataset_prompts.isdisjoint(eval_prompts)


def test_reviewed_instruction_validator_accepts_current_dataset() -> None:
    assert validate_file(DATASET_PATH, eval_dir=Path("tracks/leverage/evals")) == []
