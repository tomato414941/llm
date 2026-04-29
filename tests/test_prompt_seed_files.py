import json
from pathlib import Path
from typing import Any

from llm.leverage.capabilities import ALLOWED_CAPABILITIES


PROMPT_FILES = [Path("tracks/leverage/prompts/instruction-seeds.jsonl")]
REQUIRED_FIELDS = {
    "id",
    "capability",
    "purpose",
    "system_prompt",
    "prompt",
    "output_format",
    "constraints",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        assert line.strip(), f"{path}:{line_number} must not be blank"
        payload = json.loads(line)
        assert isinstance(payload, dict), f"{path}:{line_number} must be a JSON object"
        rows.append(payload)
    return rows


def test_prompt_seed_files_have_stable_schema_and_unique_ids() -> None:
    seen_ids: set[str] = set()
    for path in PROMPT_FILES:
        rows = load_jsonl(path)
        assert rows, f"{path} must contain at least one prompt"
        for row in rows:
            assert REQUIRED_FIELDS <= row.keys()
            assert isinstance(row["id"], str) and row["id"]
            assert row["id"] not in seen_ids
            seen_ids.add(row["id"])
            assert isinstance(row["capability"], str) and row["capability"]
            assert row["capability"] in ALLOWED_CAPABILITIES
            assert isinstance(row["purpose"], str) and row["purpose"]
            assert isinstance(row["system_prompt"], str) and row["system_prompt"]
            assert isinstance(row["prompt"], str) and row["prompt"]
            assert isinstance(row["output_format"], str) and row["output_format"]
            assert isinstance(row["constraints"], list) and row["constraints"]
            assert all(isinstance(item, str) and item for item in row["constraints"])


def test_training_seed_prompts_cover_current_decision_areas() -> None:
    rows = load_jsonl(Path("tracks/leverage/prompts/instruction-seeds.jsonl"))
    capabilities = {row["capability"] for row in rows}

    assert len(rows) >= 50
    assert capabilities <= ALLOWED_CAPABILITIES
    assert {"instruction_following", "reasoning", "coding", "tool_use"} <= capabilities
