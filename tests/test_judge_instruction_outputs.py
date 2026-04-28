import json
from pathlib import Path

import pytest

from llm.leverage import judge_instruction_outputs as judge


def raw_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "source_prompt_id": "lt_seed_test",
        "category": "resource_judgment",
        "model": "generator_a",
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


def judge_json(decision: str = "accept") -> str:
    return json.dumps(
        {
            "scores": {
                "correctness": 2,
                "instruction_following": 2,
                "conciseness": 2,
                "safety": 2,
            },
            "decision": decision,
            "reason": "Good answer.",
        }
    )


def test_answer_id_separates_generator_model() -> None:
    assert judge.answer_id(raw_row()) == "lt_seed_test:generator_a"


def test_build_payload_contains_rubric_and_candidate_answer() -> None:
    payload = judge.build_payload(raw_row(), judge_model="judge/model", max_tokens=256, temperature=0.0)

    assert payload["model"] == "judge/model"
    user_message = payload["messages"][1]["content"]
    assert "candidate_answer" in user_message
    assert "Use hosted inference before renting GPUs." in user_message
    assert "decision must be one of accept, needs_edit, reject" in user_message


def test_parse_judge_response_accepts_fenced_json() -> None:
    payload = judge.parse_judge_response(f"```json\n{judge_json()}\n```")

    assert payload["decision"] == "accept"


def test_normalize_judgment_rejects_bad_score() -> None:
    payload = json.loads(judge_json())
    payload["scores"]["correctness"] = 3

    with pytest.raises(ValueError, match="correctness"):
        judge.normalize_judgment(payload)


def test_judgment_record_uses_generator_and_judge_fields() -> None:
    record = judge.judgment_record(
        raw_row(),
        judge_model="judge/api-model",
        judge_label="judge_label",
        judge_response=judge_json("needs_edit"),
    )

    assert record["source_prompt_id"] == "lt_seed_test"
    assert record["answer_id"] == "lt_seed_test:generator_a"
    assert record["generator_model"] == "generator_a"
    assert record["judge_model"] == "judge_label"
    assert record["judge_api_model"] == "judge/api-model"
    assert record["decision"] == "needs_edit"
    assert record["scores"]["safety"] == 2


def test_judge_rows_honors_limit() -> None:
    calls: list[dict[str, object]] = []

    def client(payload):
        calls.append(payload)
        return judge_json()

    records = judge.judge_rows(
        [(1, raw_row(source_prompt_id="a")), (2, raw_row(source_prompt_id="b"))],
        client=client,
        judge_model="judge/model",
        judge_label="judge_label",
        max_tokens=256,
        temperature=0.0,
        limit=1,
    )

    assert len(records) == 1
    assert len(calls) == 1


def test_write_outputs(tmp_path: Path) -> None:
    records = [
        judge.judgment_record(
            raw_row(),
            judge_model="judge/api-model",
            judge_label="judge_label",
            judge_response=judge_json(),
        )
    ]
    jsonl_path = tmp_path / "judgments.jsonl"
    csv_path = tmp_path / "judgments.csv"

    judge.write_jsonl(jsonl_path, records, overwrite=False)
    judge.write_csv(csv_path, records, overwrite=False)

    assert json.loads(jsonl_path.read_text(encoding="utf-8"))["decision"] == "accept"
    assert "instruction_following" in csv_path.read_text(encoding="utf-8").splitlines()[0]
