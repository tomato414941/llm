import csv
import json
from pathlib import Path

from llm.leverage.compare_post_training import (
    grouped_tasks,
    read_scores,
    score_predictions,
    write_score_artifacts,
    write_report,
)


def write_scores(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "suite", "task_id", "category", "score", "passed", "reason"],
        )
        writer.writeheader()
        for model in ("base", "adapter"):
            for task_id, passed in (
                ("both-fail-then-adapter", model == "adapter"),
                ("base-only", model == "base"),
                ("both-pass", True),
                ("both-fail", False),
            ):
                writer.writerow(
                    {
                        "model": model,
                        "suite": "suite",
                        "task_id": task_id,
                        "category": "category",
                        "score": "1.0" if passed else "0.0",
                        "passed": str(passed).lower(),
                        "reason": "ok" if passed else "failed",
                    }
                )


def write_raw_scores(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "suite", "task_id", "category", "score", "passed", "reason"],
        )
        writer.writeheader()
        for model in ("base", "adapter"):
            for task_id in ("both-fail-then-adapter", "base-only", "both-pass", "both-fail"):
                writer.writerow(
                    {
                        "model": model,
                        "suite": "suite",
                        "task_id": task_id,
                        "category": "category",
                        "score": "1.0" if task_id == "both-pass" else "0.0",
                        "passed": str(task_id == "both-pass").lower(),
                        "reason": "ok" if task_id == "both-pass" else "failed",
                    }
                )


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_grouped_tasks_classifies_base_adapter_and_recovered(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.csv"
    parsed_path = tmp_path / "parsed.csv"
    write_raw_scores(raw_path)
    write_scores(parsed_path)

    groups = grouped_tasks(
        base_model="base",
        adapter_model="adapter",
        raw_scores=read_scores(raw_path),
        parsed_scores=read_scores(parsed_path),
    )

    assert groups["recovered_by_qwen_final_parse"] == ["both-fail-then-adapter", "base-only"]
    assert groups["adapter_only"] == []
    assert groups["base_only"] == []
    assert groups["both_pass"] == ["both-pass"]
    assert groups["both_fail"] == ["both-fail"]


def test_write_report_contains_summary_and_next_data_target(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.csv"
    parsed_path = tmp_path / "parsed.csv"
    report_path = tmp_path / "report.md"
    write_raw_scores(raw_path)
    write_scores(parsed_path)

    write_report(
        path=report_path,
        base_model="base",
        adapter_model="adapter",
        raw_scores=read_scores(raw_path),
        parsed_scores=read_scores(parsed_path),
    )

    text = report_path.read_text(encoding="utf-8")
    assert "| `base` | 1 | 2 | 4 |" in text
    assert "| `adapter` | 1 | 2 | 4 |" in text
    assert "Prioritize `both_fail` project-judgment tasks" in text


def test_score_predictions_derives_raw_and_qwen_final_scores_from_predictions(tmp_path: Path) -> None:
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

    raw = score_predictions(tasks_paths=[tasks_path], predictions_path=predictions_path, parse_qwen_final=False)
    parsed = score_predictions(tasks_paths=[tasks_path], predictions_path=predictions_path, parse_qwen_final=True)

    assert raw[("base", "task-1")].passed is False
    assert parsed[("base", "task-1")].passed is True
    assert parsed[("adapter", "task-1")].passed is False


def test_write_score_artifacts_writes_raw_and_qwen_final_outputs(tmp_path: Path) -> None:
    tasks_path = tmp_path / "tasks.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    raw_output = tmp_path / "raw.csv"
    parsed_output = tmp_path / "parsed.csv"
    summary_output = tmp_path / "summary.csv"
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
    write_jsonl(predictions_path, [{"task_id": "task-1", "model": "base", "response": "</think>\n\nok"}])

    write_score_artifacts(
        raw_output=raw_output,
        parsed_output=parsed_output,
        summary_output=summary_output,
        tasks_paths=[tasks_path],
        predictions_path=predictions_path,
    )

    assert read_scores(raw_output)[("base", "task-1")].passed is False
    assert read_scores(parsed_output)[("base", "task-1")].passed is True
    assert summary_output.exists()
