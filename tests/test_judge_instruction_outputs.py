import json
from pathlib import Path

import pytest

from llm.leverage import judge_instruction_outputs as judge
from llm.leverage.collect_openai import ChatResult
from llm.leverage.instruction_contract import build_instruction_contract


def raw_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "source_prompt_id": "lt_seed_test",
        "capability": "tool_use",
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
    payload = judge.build_payload(
        raw_row(),
        judge_model="judge/model",
        max_tokens=256,
        temperature=0.0,
        reasoning_effort="none",
        exclude_reasoning=True,
    )

    assert payload["model"] == "judge/model"
    assert payload["reasoning"] == {"exclude": True, "effort": "none"}
    user_message = payload["messages"][1]["content"]
    assert "candidate_answer" in user_message
    assert "Use hosted inference before renting GPUs." in user_message
    assert '"safety": 2' in user_message
    assert 'Do not use synonyms such as "safe".' in user_message
    assert "Do not apply it as a requirement for the candidate answer." in user_message
    assert "Judge the candidate answer only against the source instruction contract" in user_message
    assert "source_system_prompt:\nAnswer concisely." in user_message
    assert "instruction_contract" in user_message
    assert "decision must be one of accept, needs_edit, reject" in user_message


def test_build_judge_prompt_uses_shared_instruction_contract() -> None:
    row = raw_row()
    expected_contract = build_instruction_contract(
        prompt="When should hosted inference be used?",
        output_format="short_answer",
        constraints=["mention cost"],
    )

    user_message = judge.build_judge_prompt(row)

    assert "source_system_prompt:\nAnswer concisely.\n\n" in user_message
    assert f"instruction_contract:\n{expected_contract}\n\ncandidate_answer:" in user_message


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


def test_failed_judgment_record_preserves_raw_response() -> None:
    record = judge.failed_judgment_record(
        raw_row(),
        judge_model="judge/api-model",
        judge_label="judge_label",
        judge_response="{bad json",
        error=ValueError("bad response"),
    )

    assert record["answer_id"] == "lt_seed_test:generator_a"
    assert record["decision"] == "parse_error"
    assert record["scores"]["correctness"] == 0
    assert record["raw_judge_response"] == "{bad json"


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
        reasoning_effort="none",
        exclude_reasoning=True,
        limit=1,
    )

    assert len(records) == 1
    assert len(calls) == 1


def test_judge_rows_resume_skips_existing_answer_ids() -> None:
    calls: list[dict[str, object]] = []
    existing = judge.judgment_record(
        raw_row(source_prompt_id="a"),
        judge_model="judge/api-model",
        judge_label="judge_label",
        judge_response=judge_json(),
    )

    def client(payload):
        calls.append(payload)
        return judge_json("needs_edit")

    records = judge.judge_rows(
        [(1, raw_row(source_prompt_id="a")), (2, raw_row(source_prompt_id="b"))],
        client=client,
        judge_model="judge/model",
        judge_label="judge_label",
        max_tokens=256,
        temperature=0.0,
        reasoning_effort="none",
        exclude_reasoning=True,
        limit=None,
        existing_records=[existing],
    )

    assert len(records) == 2
    assert len(calls) == 1
    assert records[0]["answer_id"] == "a:generator_a"
    assert records[1]["answer_id"] == "b:generator_a"
    assert records[1]["decision"] == "needs_edit"


def test_judge_rows_records_parse_errors_and_continues() -> None:
    responses = iter(["not json", judge_json("accept")])

    records = judge.judge_rows(
        [(1, raw_row(source_prompt_id="a")), (2, raw_row(source_prompt_id="b"))],
        client=lambda _payload: next(responses),
        judge_model="judge/model",
        judge_label="judge_label",
        max_tokens=256,
        temperature=0.0,
        reasoning_effort="none",
        exclude_reasoning=True,
        limit=None,
    )

    assert [record["decision"] for record in records] == ["parse_error", "accept"]


def test_judge_rows_records_client_errors_and_continues() -> None:
    calls = iter([ValueError("API response message content must be text"), judge_json("accept")])

    def client(_payload):
        value = next(calls)
        if isinstance(value, Exception):
            raise value
        return value

    records = judge.judge_rows(
        [(1, raw_row(source_prompt_id="a")), (2, raw_row(source_prompt_id="b"))],
        client=client,
        judge_model="judge/model",
        judge_label="judge_label",
        max_tokens=256,
        temperature=0.0,
        reasoning_effort="none",
        exclude_reasoning=True,
        limit=None,
    )

    assert [record["decision"] for record in records] == ["parse_error", "accept"]
    assert "API response message content must be text" in records[0]["reason"]


def test_judge_rows_can_choose_random_non_self_judges_per_row() -> None:
    calls: list[dict[str, object]] = []

    def client(payload):
        calls.append(payload)
        return judge_json("accept")

    records = judge.judge_rows(
        [
            (1, raw_row(source_prompt_id="a", model="generator_a")),
            (2, raw_row(source_prompt_id="b", model="judge_a")),
            (3, raw_row(source_prompt_id="c", model="judge_b")),
        ],
        client=client,
        judge_model="fallback/model",
        judge_label="fallback",
        max_tokens=256,
        temperature=0.0,
        reasoning_effort="none",
        exclude_reasoning=True,
        limit=None,
        judge_candidates=[("judge_a", "model/a", 1.0), ("judge_b", "model/b", 1.0)],
        random_seed=7,
    )

    assert len(records) == 3
    assert calls[0]["model"] in {"model/a", "model/b"}
    assert records[1]["judge_model"] == "judge_b"
    assert records[1]["judge_api_model"] == "model/b"
    assert records[2]["judge_model"] == "judge_a"
    assert records[2]["judge_api_model"] == "model/a"


def test_parse_judge_candidate_accepts_optional_weight() -> None:
    assert judge.parse_judge_candidate("judge_a=model/a:0.25") == ("judge_a", "model/a", 0.25)
    assert judge.parse_judge_candidate("judge_a=model/a") == ("judge_a", "model/a", 1.0)
    with pytest.raises(ValueError, match="label=model"):
        judge.parse_judge_candidate("judge_a")


def test_parse_args_defaults_to_random_judge_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "judge_instruction_outputs",
            "--input",
            "candidates.jsonl",
            "--output",
            "judgments.jsonl",
        ],
    )

    args = judge.parse_args()
    judge.validate_args(args)

    assert args.judge_candidates == [
        judge.parse_judge_candidate(value) for value in judge.DEFAULT_JUDGE_CANDIDATES
    ]


def test_judge_rows_accepts_chat_result_client_response() -> None:
    records = judge.judge_rows(
        [(1, raw_row())],
        client=lambda _payload: ChatResult(judge_json("accept"), "stop", {"completion_tokens": 24}),
        judge_model="judge/model",
        judge_label="judge_label",
        max_tokens=256,
        temperature=0.0,
        reasoning_effort="none",
        exclude_reasoning=True,
        limit=None,
    )

    assert records[0]["decision"] == "accept"


def test_load_existing_records_rejects_duplicate_answer_ids(tmp_path: Path) -> None:
    path = tmp_path / "judgments.jsonl"
    row = judge.judgment_record(
        raw_row(),
        judge_model="judge/api-model",
        judge_label="judge_label",
        judge_response=judge_json(),
    )
    path.write_text(
        "\n".join(json.dumps(item) for item in [row, row]) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate answer_id"):
        judge.load_existing_records(path)


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


def test_summary_rows_counts_decisions_and_scores() -> None:
    records = [
        judge.judgment_record(
            raw_row(source_prompt_id="a"),
            judge_model="judge/api-model",
            judge_label="judge_label",
            judge_response=judge_json("accept"),
        ),
        judge.failed_judgment_record(
            raw_row(source_prompt_id="b"),
            judge_model="judge/api-model",
            judge_label="judge_label",
            judge_response="not json",
            error=ValueError("bad response"),
        ),
    ]

    rows = judge.summary_rows(records)

    assert {"scope": "overall", "name": "total", "value": 2, "rate": "1.000"} in rows
    assert {"scope": "decision", "name": "accept", "value": 1, "rate": "0.500"} in rows
    assert {"scope": "decision", "name": "parse_error", "value": 1, "rate": "0.500"} in rows
    assert {"scope": "judge_model", "name": "judge_label", "value": 2, "rate": "1.000"} in rows
    assert {"scope": "avg_score", "name": "correctness", "value": "1.000", "rate": ""} in rows


def test_write_summary_csv(tmp_path: Path) -> None:
    records = [
        judge.judgment_record(
            raw_row(),
            judge_model="judge/api-model",
            judge_label="judge_label",
            judge_response=judge_json(),
        )
    ]
    summary_path = tmp_path / "summary.csv"

    judge.write_summary_csv(summary_path, records, overwrite=False)

    lines = summary_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "scope,name,value,rate"
    assert "decision,accept,1,1.000" in lines
