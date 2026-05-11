import csv
import json
from pathlib import Path

import pytest

from llm.leverage.gpt_oss_gsm8k_eval import (
    Gsm8kTask,
    build_messages,
    extract_expected_answer,
    extract_numeric_prediction,
    run_gsm8k_eval,
)


def test_extract_expected_answer_reads_gsm8k_final_marker() -> None:
    assert extract_expected_answer("Work here.\n#### 1,234") == "1234"


def test_extract_expected_answer_rejects_missing_marker() -> None:
    with pytest.raises(ValueError, match="missing final marker"):
        extract_expected_answer("The answer is 42.")


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("42", "42"),
        ("The answer is 1,234.", "1234"),
        ("First 10, then final -7.5", "-7.5"),
        ("No numeric answer", ""),
    ],
)
def test_extract_numeric_prediction_uses_last_number(response: str, expected: str) -> None:
    assert extract_numeric_prediction(response) == expected


def test_build_messages_requests_numeric_only_answer() -> None:
    messages = build_messages("What is 2+2?", "Reasoning: low")

    assert messages[0] == {"role": "system", "content": "Reasoning: low"}
    assert "Return only the final numeric answer" in messages[1]["content"]
    assert "What is 2+2?" in messages[1]["content"]


def test_run_gsm8k_eval_writes_outputs_with_fake_pipeline(tmp_path: Path) -> None:
    tasks = [
        Gsm8kTask(id="gsm8k_test_0000", question="What is 20 + 22?", answer="20 + 22 = 42\n#### 42"),
        Gsm8kTask(id="gsm8k_test_0001", question="What is 7 + 8?", answer="7 + 8 = 15\n#### 15"),
    ]

    def task_loader(split: str, limit: int | None) -> list[Gsm8kTask]:
        assert split == "test"
        assert limit is None
        return tasks

    class FakePipeline:
        def __init__(self) -> None:
            self.responses = [
                "analysis compute assistantfinalThe answer is 42.",
                "analysis compute assistantfinal16",
            ]

        def __call__(self, messages, *, max_new_tokens: int, do_sample: bool):  # noqa: ANN001
            assert max_new_tokens == 128
            assert do_sample is False
            content = self.responses.pop(0)
            return [{"generated_text": [*messages, {"role": "assistant", "content": content}]}]

    def pipeline_factory(*args, **kwargs):  # noqa: ANN002, ANN003
        assert args == ("text-generation",)
        assert kwargs["model"] == "fake-model"
        return FakePipeline()

    lines = run_gsm8k_eval(
        output_root=tmp_path,
        model="fake-model",
        system_prompt="Reasoning: low",
        split="test",
        limit=None,
        max_new_tokens=128,
        task_loader=task_loader,
        pipeline_factory=pipeline_factory,
    )

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    predictions = [
        json.loads(line)
        for line in (tmp_path / "predictions.harmony-final.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    with (tmp_path / "scores.harmony-final.csv").open(encoding="utf-8", newline="") as file:
        scores = list(csv.DictReader(file))

    assert lines[0] == "evaluated 2 GSM8K tasks"
    assert metadata["task_count"] == 2
    assert metadata["passed_count"] == 1
    assert metadata["accuracy"] == 0.5
    assert metadata["harmony"] == {
        "task_count": 2,
        "missing_final_marker_count": 0,
        "empty_final_response_count": 0,
        "non_final_channel_in_final_count": 0,
    }
    assert predictions == [
        {
            "task_id": "gsm8k_test_0000",
            "model": "fake-model",
            "expected": "42",
            "prediction": "42",
            "response": "The answer is 42.",
            "passed": True,
        },
        {
            "task_id": "gsm8k_test_0001",
            "model": "fake-model",
            "expected": "15",
            "prediction": "16",
            "response": "16",
            "passed": False,
        },
    ]
    assert [score["passed"] for score in scores] == ["true", "false"]
