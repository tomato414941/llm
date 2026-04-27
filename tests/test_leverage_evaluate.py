import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from llm.leverage import evaluate


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_score_response_contains_all_passes_only_when_all_terms_are_present() -> None:
    scoring = {"type": "contains_all", "phrases": ["red", "blue"]}

    assert evaluate.score_response(scoring, "red and blue are primary colors.")[1] is True
    assert evaluate.score_response(scoring, "red is a primary color.")[1] is False


def test_score_response_exact_requires_exact_match() -> None:
    scoring = {"type": "exact", "expected": "4"}

    assert evaluate.score_response(scoring, "4")[1] is True
    assert evaluate.score_response(scoring, " 4 ")[1] is False


def test_score_response_regex_uses_pattern_match() -> None:
    scoring = {"type": "regex", "pattern": r"^\d{4}-\d{2}-\d{2}$"}

    assert evaluate.score_response(scoring, "2026-04-27")[1] is True
    assert evaluate.score_response(scoring, "April 27, 2026")[1] is False


@pytest.mark.parametrize(
    "task",
    [
        {
            "category": "qa",
            "prompt": "Missing id.",
            "scoring": {"type": "exact", "expected": "ok"},
        },
        {
            "id": "missing-prompt",
            "category": "qa",
            "scoring": {"type": "exact", "expected": "ok"},
        },
        {"id": "missing-scoring", "category": "qa", "prompt": "Missing scoring."},
    ],
)
def test_load_tasks_rejects_missing_required_fields(
    tmp_path: Path, task: dict[str, object]
) -> None:
    tasks_path = tmp_path / "tasks.jsonl"
    write_jsonl(tasks_path, [task])

    with pytest.raises(ValueError, match="missing|required"):
        evaluate.load_tasks(tasks_path)


def test_load_tasks_rejects_duplicate_task_ids(tmp_path: Path) -> None:
    tasks_path = tmp_path / "tasks.jsonl"
    write_jsonl(
        tasks_path,
        [
            {
                "id": "duplicate",
                "category": "qa",
                "prompt": "First.",
                "scoring": {"type": "exact", "expected": "yes"},
            },
            {
                "id": "duplicate",
                "category": "qa",
                "prompt": "Second.",
                "scoring": {"type": "exact", "expected": "no"},
            },
        ],
    )

    with pytest.raises(ValueError, match="duplicate"):
        evaluate.load_tasks(tasks_path)


def test_load_tasks_rejects_unknown_scoring_type(tmp_path: Path) -> None:
    tasks_path = tmp_path / "tasks.jsonl"
    write_jsonl(
        tasks_path,
        [
            {
                "id": "unknown",
                "category": "qa",
                "prompt": "Use an unsupported scorer.",
                "scoring": {"type": "semantic", "expected": "ok"},
            }
        ],
    )

    with pytest.raises(ValueError, match="scoring|unknown"):
        evaluate.load_tasks(tasks_path)


def test_load_predictions_rejects_prediction_for_missing_task(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    write_jsonl(
        predictions_path,
        [
            {"task_id": "known", "model": "example", "response": "ok"},
            {"task_id": "missing", "model": "example", "response": "ok"},
        ],
    )

    with pytest.raises(ValueError, match="missing|unknown"):
        evaluate.load_predictions(predictions_path, {"known"})


def test_evaluate_predictions_rejects_model_with_missing_task() -> None:
    tasks = {
        "first": evaluate.Task(
            id="first",
            category="qa",
            prompt="First?",
            scoring={"type": "exact", "expected": "yes"},
        ),
        "second": evaluate.Task(
            id="second",
            category="qa",
            prompt="Second?",
            scoring={"type": "exact", "expected": "no"},
        ),
    }
    predictions = [
        evaluate.Prediction(task_id="first", model="example", response="yes"),
    ]

    with pytest.raises(ValueError, match="missing predictions"):
        evaluate.evaluate_predictions(tasks, predictions)


def test_cli_writes_csv_scores(tmp_path: Path) -> None:
    tasks_path = tmp_path / "tasks.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    output_path = tmp_path / "scores.csv"
    write_jsonl(
        tasks_path,
        [
            {
                "id": "contains",
                "category": "qa",
                "prompt": "Name two primary colors.",
                "scoring": {"type": "contains_all", "phrases": ["red", "blue"]},
            },
            {
                "id": "exact",
                "category": "reasoning",
                "prompt": "What is 2 + 2?",
                "scoring": {"type": "exact", "expected": "4"},
            },
            {
                "id": "regex",
                "category": "instruction",
                "prompt": "Return an ISO date.",
                "scoring": {"type": "regex", "pattern": r"^\d{4}-\d{2}-\d{2}$"},
            },
        ],
    )
    write_jsonl(
        predictions_path,
        [
            {"task_id": "contains", "model": "example", "response": "red and blue."},
            {"task_id": "exact", "model": "example", "response": "four"},
            {"task_id": "regex", "model": "example", "response": "2026-04-27"},
        ],
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(Path.cwd() / "src"), env["PYTHONPATH"]]
        if env.get("PYTHONPATH")
        else [str(Path.cwd() / "src")]
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "llm.leverage.evaluate",
            "--tasks",
            str(tasks_path),
            "--predictions",
            str(predictions_path),
            "--output",
            str(output_path),
        ],
        check=False,
        cwd=Path.cwd(),
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    with output_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert [row["task_id"] for row in rows] == ["contains", "exact", "regex"]
    assert [row["passed"] for row in rows] == ["true", "false", "true"]
