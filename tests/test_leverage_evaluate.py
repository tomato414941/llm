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


def task(
    task_id: str,
    *,
    category: str = "qa",
    expected: str = "ok",
) -> dict[str, object]:
    return {
        "id": task_id,
        "category": category,
        "prompt": f"Return {expected}.",
        "scoring": {"type": "exact", "expected": expected},
    }


def prediction(
    task_id: str,
    *,
    model: str = "example",
    response: str = "ok",
) -> dict[str, object]:
    return {"task_id": task_id, "model": model, "response": response}


def cli_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(Path.cwd() / "src"), env["PYTHONPATH"]]
        if env.get("PYTHONPATH")
        else [str(Path.cwd() / "src")]
    )
    return env


def run_evaluate_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "llm.leverage.evaluate", *args],
        check=False,
        cwd=Path.cwd(),
        env=cli_env(),
        text=True,
        capture_output=True,
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def test_score_response_json_fields_checks_required_and_array_values() -> None:
    scoring = {
        "type": "json_fields",
        "required": {"decision": "do_not_run_yet"},
        "array_contains": {"required_controls": ["dry_run", "cost_cap"]},
    }

    assert (
        evaluate.score_response(
            scoring,
            '{"decision":"do_not_run_yet","required_controls":["cost_cap","dry_run","extra"]}',
        )[1]
        is True
    )
    assert (
        evaluate.score_response(
            scoring,
            '{"decision":"run_now","required_controls":["cost_cap","dry_run"]}',
        )[1]
        is False
    )
    assert evaluate.score_response(scoring, "not json")[1] is False


def test_score_response_json_fields_rejects_fenced_json() -> None:
    scoring = {
        "type": "json_fields",
        "required": {"operation_type": "judging"},
    }

    score, passed, reason = evaluate.score_response(
        scoring,
        '```json\n{"operation_type":"judging"}\n```',
    )

    assert score == 0.0
    assert passed is False
    assert "not valid JSON" in reason


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


def test_cli_accepts_multiple_task_files_and_writes_suite_column(tmp_path: Path) -> None:
    alpha_tasks_path = tmp_path / "alpha.jsonl"
    beta_tasks_path = tmp_path / "beta.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    output_path = tmp_path / "detail.csv"
    write_jsonl(alpha_tasks_path, [task("alpha-one")])
    write_jsonl(beta_tasks_path, [task("beta-one")])
    write_jsonl(
        predictions_path,
        [prediction("alpha-one"), prediction("beta-one")],
    )

    result = run_evaluate_cli(
        [
            "--tasks",
            str(alpha_tasks_path),
            "--tasks",
            str(beta_tasks_path),
            "--predictions",
            str(predictions_path),
            "--output",
            str(output_path),
        ]
    )

    assert result.returncode == 0, result.stderr
    rows = read_csv(output_path)
    assert [row["task_id"] for row in rows] == ["alpha-one", "beta-one"]
    assert [row["suite"] for row in rows] == ["alpha", "beta"]
    assert [row["passed"] for row in rows] == ["true", "true"]


def test_cli_rejects_duplicate_task_ids_across_task_files(tmp_path: Path) -> None:
    alpha_tasks_path = tmp_path / "alpha.jsonl"
    beta_tasks_path = tmp_path / "beta.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    output_path = tmp_path / "detail.csv"
    write_jsonl(alpha_tasks_path, [task("duplicate")])
    write_jsonl(beta_tasks_path, [task("duplicate")])
    write_jsonl(predictions_path, [prediction("duplicate")])

    result = run_evaluate_cli(
        [
            "--tasks",
            str(alpha_tasks_path),
            "--tasks",
            str(beta_tasks_path),
            "--predictions",
            str(predictions_path),
            "--output",
            str(output_path),
        ]
    )

    assert result.returncode != 0
    assert "duplicate" in result.stderr


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


def test_load_predictions_rejects_duplicate_model_task_prediction(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    write_jsonl(
        predictions_path,
        [
            prediction("only"),
            prediction("only", response="ok again"),
        ],
    )

    with pytest.raises(ValueError, match="duplicate"):
        evaluate.load_predictions(predictions_path, {"only"})


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


def test_evaluate_predictions_rejects_empty_predictions() -> None:
    tasks = {
        "first": evaluate.Task(
            id="first",
            category="qa",
            prompt="First?",
            scoring={"type": "exact", "expected": "yes"},
        )
    }

    with pytest.raises(ValueError, match="at least one prediction"):
        evaluate.evaluate_predictions(tasks, [])


def test_cli_combined_task_coverage_requires_every_task_for_each_model(tmp_path: Path) -> None:
    alpha_tasks_path = tmp_path / "alpha.jsonl"
    beta_tasks_path = tmp_path / "beta.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    output_path = tmp_path / "detail.csv"
    write_jsonl(alpha_tasks_path, [task("alpha-one")])
    write_jsonl(beta_tasks_path, [task("beta-one")])
    write_jsonl(predictions_path, [prediction("alpha-one")])

    result = run_evaluate_cli(
        [
            "--tasks",
            str(alpha_tasks_path),
            "--tasks",
            str(beta_tasks_path),
            "--predictions",
            str(predictions_path),
            "--output",
            str(output_path),
        ]
    )

    assert result.returncode != 0
    assert "missing predictions" in result.stderr
    assert "beta-one" in result.stderr


def test_cli_writes_summary_output_with_overall_rows(tmp_path: Path) -> None:
    alpha_tasks_path = tmp_path / "alpha.jsonl"
    beta_tasks_path = tmp_path / "beta.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    detail_path = tmp_path / "detail.csv"
    summary_path = tmp_path / "summary.csv"
    write_jsonl(alpha_tasks_path, [task("alpha-qa", category="qa")])
    write_jsonl(beta_tasks_path, [task("beta-reasoning", category="reasoning", expected="4")])
    write_jsonl(
        predictions_path,
        [
            prediction("alpha-qa"),
            prediction("beta-reasoning", response="wrong"),
        ],
    )

    result = run_evaluate_cli(
        [
            "--tasks",
            str(alpha_tasks_path),
            "--tasks",
            str(beta_tasks_path),
            "--predictions",
            str(predictions_path),
            "--output",
            str(detail_path),
            "--summary-output",
            str(summary_path),
        ]
    )

    assert result.returncode == 0, result.stderr
    rows = read_csv(summary_path)
    row_keys = {(row["model"], row["suite"], row["category"]) for row in rows}
    assert ("example", "alpha", "qa") in row_keys
    assert ("example", "beta", "reasoning") in row_keys
    assert ("example", "alpha", "__overall__") in row_keys
    assert ("example", "beta", "__overall__") in row_keys
    overall = next(
        row
        for row in rows
        if row["model"] == "example"
        and row["suite"] == "alpha"
        and row["category"] == "__overall__"
    )
    assert overall["task_count"] == "1"
    assert overall["passed_count"] == "1"
    assert overall["pass_rate"] == "1.000"


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

    result = run_evaluate_cli(
        [
            "--tasks",
            str(tasks_path),
            "--predictions",
            str(predictions_path),
            "--output",
            str(output_path),
        ]
    )

    assert result.returncode == 0, result.stderr
    rows = read_csv(output_path)

    assert [row["task_id"] for row in rows] == ["contains", "exact", "regex"]
    assert [row["passed"] for row in rows] == ["true", "false", "true"]


def test_real_leverage_smoke_eval_contract() -> None:
    tasks_path = Path("evals/leverage-smoke.jsonl")
    predictions_path = Path("experiments/leverage/predictions.example.jsonl")

    tasks = evaluate.load_tasks(tasks_path)
    predictions = evaluate.load_predictions(predictions_path, set(tasks))
    results = evaluate.evaluate_predictions(tasks, predictions)

    assert set(tasks) == {
        "qa_capital_france",
        "qa_water_freezing",
        "summary_mission",
        "summary_runpod",
        "instruction_json",
        "instruction_bullets",
        "reasoning_arithmetic",
        "reasoning_order",
        "coding_python_function",
        "coding_sql_count",
        "qa_author",
        "instruction_lowercase",
    }
    assert {task.category for task in tasks.values()} == {
        "coding",
        "instruction",
        "qa",
        "reasoning",
        "summarization",
    }
    assert len(predictions) == len(tasks)
    assert {prediction.model for prediction in predictions} == {"example-baseline"}
    assert all(result.model == "example-baseline" for result in results)
    assert all(result.passed for result in results)


def test_real_project_judgment_eval_contract() -> None:
    tasks_path = Path("evals/project-judgment-v0.jsonl")
    predictions_path = Path("experiments/leverage/project-judgment-v0.example.jsonl")

    tasks = evaluate.load_tasks(tasks_path)
    predictions = evaluate.load_predictions(predictions_path, set(tasks))
    results = evaluate.evaluate_predictions(tasks, predictions)

    assert len(tasks) == 18
    assert {task.category for task in tasks.values()} == {
        "coding_repo_reasoning",
        "eval_design",
        "experiment_judgment",
        "loss_curve_interpretation",
        "runpod_cost_awareness",
        "track_distinction",
    }
    assert len(predictions) == len(tasks)
    assert {prediction.model for prediction in predictions} == {"example-baseline"}
    assert all(result.suite == "project-judgment-v0" for result in results)
    assert all(result.passed for result in results)


def test_real_leverage_model_spec_eval_contract() -> None:
    tasks_path = Path("evals/leverage-model-spec.jsonl")
    predictions_path = Path("experiments/leverage/leverage-model-spec.example.jsonl")

    tasks = evaluate.load_tasks(tasks_path)
    predictions = evaluate.load_predictions(predictions_path, set(tasks))
    results = evaluate.evaluate_predictions(tasks, predictions)

    assert len(tasks) == 18
    assert {task.category for task in tasks.values()} == {
        "conciseness",
        "cost_awareness",
        "data_quality",
        "eval_hygiene",
        "instruction_hierarchy",
        "overbuild_avoidance",
        "run_recovery",
        "training_distinction",
    }
    assert len(predictions) == len(tasks)
    assert {prediction.model for prediction in predictions} == {"example-baseline"}
    assert all(result.suite == "leverage-model-spec" for result in results)
    assert all(result.passed for result in results)
    assert sum(1 for task in tasks.values() if task.scoring["type"] == "json_fields") == 6


def test_real_two_layer_eval_contract() -> None:
    tasks = evaluate.load_task_suites(
        [
            Path("evals/leverage-smoke.jsonl"),
            Path("evals/project-judgment-v0.jsonl"),
        ]
    )
    predictions = evaluate.load_predictions(
        Path("experiments/leverage/two-layer.example.jsonl"),
        set(tasks),
    )
    results = evaluate.evaluate_predictions(tasks, predictions)

    assert len(tasks) == 30
    assert len(predictions) == len(tasks)
    assert {result.suite for result in results} == {"leverage-smoke", "project-judgment-v0"}
    assert all(result.passed for result in results)
