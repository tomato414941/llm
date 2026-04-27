import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Task:
    id: str
    category: str
    prompt: str
    scoring: dict[str, Any]
    suite: str = ""


@dataclass(frozen=True)
class Prediction:
    task_id: str
    model: str
    response: str


@dataclass(frozen=True)
class ScoreResult:
    model: str
    task_id: str
    category: str
    score: float
    passed: bool
    reason: str
    suite: str = ""


def load_tasks(path: Path) -> dict[str, Task]:
    tasks: dict[str, Task] = {}
    suite = path.stem
    for line_number, payload in load_jsonl(path):
        require_keys(payload, {"id", "category", "prompt", "scoring"}, "task", line_number)
        task_id = require_string(payload["id"], f"task line {line_number} field 'id'")
        if task_id in tasks:
            raise ValueError(f"duplicate task id {task_id!r} in {path} at line {line_number}")
        category = require_string(payload["category"], f"task {task_id!r} field 'category'")
        prompt = require_string(payload["prompt"], f"task {task_id!r} field 'prompt'")
        scoring = payload["scoring"]
        if not isinstance(scoring, dict):
            raise ValueError(f"task {task_id!r} field 'scoring' must be an object")
        validate_scoring(scoring, task_id)
        tasks[task_id] = Task(
            id=task_id,
            category=category,
            prompt=prompt,
            scoring=scoring,
            suite=suite,
        )
    return tasks


def load_task_suites(paths: list[Path]) -> dict[str, Task]:
    tasks: dict[str, Task] = {}
    sources: dict[str, Path] = {}
    for path in paths:
        for task_id, task in load_tasks(path).items():
            if task_id in tasks:
                raise ValueError(
                    f"duplicate task id {task_id!r} in {path}; already loaded from {sources[task_id]}"
                )
            tasks[task_id] = task
            sources[task_id] = path
    return tasks


def load_predictions(path: Path, task_ids: set[str] | None = None) -> list[Prediction]:
    predictions: list[Prediction] = []
    seen: set[tuple[str, str]] = set()
    for line_number, payload in load_jsonl(path):
        require_keys(payload, {"task_id", "model", "response"}, "prediction", line_number)
        task_id = require_string(payload["task_id"], f"prediction line {line_number} field 'task_id'")
        if task_ids is not None and task_id not in task_ids:
            raise ValueError(f"prediction references missing task id {task_id!r} at line {line_number}")
        model = require_string(payload["model"], f"prediction line {line_number} field 'model'")
        key = (model, task_id)
        if key in seen:
            raise ValueError(
                f"duplicate prediction for model {model!r} and task id {task_id!r} at line {line_number}"
            )
        seen.add(key)
        response = require_string(payload["response"], f"prediction line {line_number} field 'response'")
        predictions.append(Prediction(task_id=task_id, model=model, response=response))
    return predictions


def evaluate_predictions(tasks: dict[str, Task], predictions: list[Prediction]) -> list[ScoreResult]:
    validate_no_duplicate_predictions(predictions)
    validate_prediction_coverage(set(tasks), predictions)
    results: list[ScoreResult] = []
    for prediction in predictions:
        task = tasks.get(prediction.task_id)
        if task is None:
            raise ValueError(f"prediction references missing task id {prediction.task_id!r}")
        score, passed, reason = score_response(task.scoring, prediction.response)
        results.append(
            ScoreResult(
                model=prediction.model,
                task_id=prediction.task_id,
                suite=task.suite,
                category=task.category,
                score=score,
                passed=passed,
                reason=reason,
            )
        )
    return results


def validate_no_duplicate_predictions(predictions: list[Prediction]) -> None:
    seen: set[tuple[str, str]] = set()
    for prediction in predictions:
        key = (prediction.model, prediction.task_id)
        if key in seen:
            raise ValueError(
                f"duplicate prediction for model {prediction.model!r} and task id {prediction.task_id!r}"
            )
        seen.add(key)


def validate_prediction_coverage(task_ids: set[str], predictions: list[Prediction]) -> None:
    if not predictions:
        raise ValueError("prediction file must contain at least one prediction")
    by_model: dict[str, set[str]] = {}
    for prediction in predictions:
        by_model.setdefault(prediction.model, set()).add(prediction.task_id)
    for model, predicted_task_ids in by_model.items():
        missing = sorted(task_ids - predicted_task_ids)
        if missing:
            raise ValueError(
                f"model {model!r} is missing predictions for task ids: {', '.join(missing)}"
            )


def write_results(path: Path, results: list[ScoreResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["model", "suite", "task_id", "category", "score", "passed", "reason"],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "model": result.model,
                    "suite": result.suite,
                    "task_id": result.task_id,
                    "category": result.category,
                    "score": f"{result.score:.1f}",
                    "passed": str(result.passed).lower(),
                    "reason": result.reason,
                }
            )


def write_summary(path: Path, results: list[ScoreResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "suite",
        "category",
        "task_count",
        "passed_count",
        "avg_score",
        "pass_rate",
    ]
    groups: dict[tuple[str, str, str], list[ScoreResult]] = {}
    for result in results:
        groups.setdefault((result.model, result.suite, result.category), []).append(result)
        groups.setdefault((result.model, result.suite, "__overall__"), []).append(result)
        groups.setdefault((result.model, "__overall__", "__overall__"), []).append(result)

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(groups):
            group_results = groups[key]
            task_count = len(group_results)
            passed_count = sum(1 for result in group_results if result.passed)
            avg_score = sum(result.score for result in group_results) / task_count
            pass_rate = passed_count / task_count
            model, suite, category = key
            writer.writerow(
                {
                    "model": model,
                    "suite": suite,
                    "category": category,
                    "task_count": str(task_count),
                    "passed_count": str(passed_count),
                    "avg_score": f"{avg_score:.3f}",
                    "pass_rate": f"{pass_rate:.3f}",
                }
            )


def score_response(scoring: dict[str, Any], response: str) -> tuple[float, bool, str]:
    scoring_type = scoring.get("type")
    if scoring_type == "contains_all":
        phrases = contains_all_phrases(scoring)
        missing = [phrase for phrase in phrases if phrase not in response]
        if missing:
            return 0.0, False, f"missing required text: {', '.join(repr(item) for item in missing)}"
        return 1.0, True, "contains all required text"
    if scoring_type == "exact":
        expected = exact_expected(scoring)
        if response == expected:
            return 1.0, True, "response exactly matched expected text"
        return 0.0, False, "response did not exactly match expected text"
    if scoring_type == "regex":
        pattern = regex_pattern(scoring)
        if re.search(pattern, response):
            return 1.0, True, "response matched regex"
        return 0.0, False, "response did not match regex"
    raise ValueError(f"unknown scoring type {scoring_type!r}")


def load_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    records: list[tuple[int, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path} line {line_number} is not valid JSON: {error.msg}") from error
            if not isinstance(payload, dict):
                raise ValueError(f"{path} line {line_number} must be a JSON object")
            records.append((line_number, payload))
    return records


def require_keys(
    payload: dict[str, Any],
    required: set[str],
    label: str,
    line_number: int,
) -> None:
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"{label} line {line_number} missing required fields: {', '.join(missing)}")


def require_string(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    return value


def validate_scoring(scoring: dict[str, Any], task_id: str) -> None:
    scoring_type = scoring.get("type")
    if not isinstance(scoring_type, str):
        raise ValueError(f"task {task_id!r} scoring field 'type' must be a string")
    if scoring_type == "contains_all":
        contains_all_phrases(scoring)
    elif scoring_type == "exact":
        exact_expected(scoring)
    elif scoring_type == "regex":
        regex_pattern(scoring)
    else:
        raise ValueError(f"unknown scoring type {scoring_type!r} for task {task_id!r}")


def contains_all_phrases(scoring: dict[str, Any]) -> list[str]:
    phrases = scoring.get("phrases", scoring.get("expected"))
    if not isinstance(phrases, list) or not all(isinstance(item, str) for item in phrases):
        raise ValueError("contains_all scoring requires field 'phrases' as a list of strings")
    return phrases


def exact_expected(scoring: dict[str, Any]) -> str:
    expected = scoring.get("expected")
    if not isinstance(expected, str):
        raise ValueError("exact scoring requires field 'expected' as a string")
    return expected


def regex_pattern(scoring: dict[str, Any]) -> str:
    pattern = scoring.get("pattern")
    if not isinstance(pattern, str):
        raise ValueError("regex scoring requires field 'pattern' as a string")
    try:
        re.compile(pattern)
    except re.error as error:
        raise ValueError(f"regex scoring pattern is invalid: {error}") from error
    return pattern


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True, action="append")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = load_task_suites(args.tasks)
    predictions = load_predictions(args.predictions, set(tasks))
    results = evaluate_predictions(tasks, predictions)
    write_results(args.output, results)
    if args.summary_output is not None:
        write_summary(args.summary_output, results)


if __name__ == "__main__":
    main()
