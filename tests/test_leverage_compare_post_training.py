import json
from pathlib import Path

from llm.leverage.compare_post_training import (
    ComparedTask,
    grouped_by_category,
    grouped_by_model_outcome,
    score_predictions,
    write_report,
)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_score_predictions_compares_raw_decoded_and_qwen_final_parse(tmp_path: Path) -> None:
    tasks_path = tmp_path / "tasks.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    write_jsonl(
        tasks_path,
        [
            {
                "id": "task-1",
                "category": "qa",
                "prompt": "Return ok.",
                "scoring": {"type": "exact", "expected": "ok"},
            }
        ],
    )
    write_jsonl(
        predictions_path,
        [
            {"task_id": "task-1", "model": "base", "response": "</think>\n\nok"},
            {"task_id": "task-1", "model": "adapter", "response": "wrong"},
        ],
    )

    rows = score_predictions([tasks_path], predictions_path)

    assert rows == [
        ComparedTask(
            model="base",
            task_id="task-1",
            suite="tasks",
            category="qa",
            raw_passed=False,
            parsed_passed=True,
            raw_reason="response did not exactly match expected text",
            parsed_reason="response exactly matched expected text",
        ),
        ComparedTask(
            model="adapter",
            task_id="task-1",
            suite="tasks",
            category="qa",
            raw_passed=False,
            parsed_passed=False,
            raw_reason="response did not exactly match expected text",
            parsed_reason="response did not exactly match expected text",
        ),
    ]


def test_grouped_by_category_preserves_rows() -> None:
    rows = [
        ComparedTask("base", "task-1", "suite", "qa", False, True, "raw failed", "parsed passed"),
        ComparedTask("base", "task-2", "suite", "json", True, True, "raw passed", "parsed passed"),
    ]

    assert grouped_by_category(rows) == {"qa": [rows[0]], "json": [rows[1]]}


def test_grouped_by_model_outcome_classifies_base_and_adapter() -> None:
    rows = [
        ComparedTask("base", "adapter-only", "suite", "qa", False, False, "raw failed", "parsed failed"),
        ComparedTask("adapter", "adapter-only", "suite", "qa", False, True, "raw failed", "parsed passed"),
        ComparedTask("base", "base-only", "suite", "qa", False, True, "raw failed", "parsed passed"),
        ComparedTask("adapter", "base-only", "suite", "qa", False, False, "raw failed", "parsed failed"),
        ComparedTask("base", "both-pass", "suite", "qa", True, True, "raw passed", "parsed passed"),
        ComparedTask("adapter", "both-pass", "suite", "qa", True, True, "raw passed", "parsed passed"),
        ComparedTask("base", "both-fail", "suite", "qa", False, False, "raw failed", "parsed failed"),
        ComparedTask("adapter", "both-fail", "suite", "qa", False, False, "raw failed", "parsed failed"),
    ]

    groups = grouped_by_model_outcome(rows, base_model="base", adapter_model="adapter")

    assert groups == {
        "adapter_only": ["adapter-only"],
        "base_only": ["base-only"],
        "both_pass": ["both-pass"],
        "both_fail": ["both-fail"],
    }


def test_write_report_contains_category_sections_and_statuses(tmp_path: Path) -> None:
    report_path = tmp_path / "report.md"
    rows = [
        ComparedTask("base", "task-1", "suite-a", "qa", False, False, "raw failed", "parsed failed"),
        ComparedTask("adapter", "task-1", "suite-a", "qa", False, True, "raw failed", "parsed passed"),
        ComparedTask("base", "task-2", "suite-b", "json", True, True, "raw passed", "parsed passed"),
        ComparedTask("adapter", "task-2", "suite-b", "json", True, False, "raw passed", "parsed failed"),
    ]

    write_report(report_path, rows, base_model="base", adapter_model="adapter")

    text = report_path.read_text(encoding="utf-8")
    assert "| 2 | 2 | 4 |" in text
    assert "## Adapter Only" in text
    assert "- `task-1`" in text
    assert "## json" in text
    assert "## qa" in text
    assert "| `adapter` | `suite-a` | `task-1` | 0 | 1 | `parsed_only` |" in text
    assert "| `adapter` | `suite-b` | `task-2` | 1 | 0 | `raw_only` |" in text
