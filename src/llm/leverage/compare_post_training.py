import argparse
from dataclasses import dataclass
from pathlib import Path

from llm.leverage.evaluate import (
    Prediction,
    ScoreResult,
    evaluate_predictions,
    load_predictions,
    load_task_suites,
)
from llm.leverage.evaluate_sft_adapter import extract_qwen_final_response


DEFAULT_TASKS = [
    Path("tracks/leverage/evals/leverage-smoke.jsonl"),
    Path("tracks/leverage/evals/project-judgment.jsonl"),
]
DEFAULT_PREDICTIONS = Path("outputs/leverage-sft-smoke/post-training-predictions.jsonl")
DEFAULT_REPORT = Path("tracks/leverage/runs/leverage-sft-smoke-diff.md")
DEFAULT_BASE_MODEL = "qwen3.5-0.8b-base"
DEFAULT_ADAPTER_MODEL = "qwen3.5-0.8b-lora-smoke"


@dataclass(frozen=True)
class ComparedTask:
    model: str
    task_id: str
    suite: str
    capability: str
    raw_passed: bool
    parsed_passed: bool
    raw_reason: str
    parsed_reason: str


def qwen_final_predictions(predictions: list[Prediction]) -> list[Prediction]:
    return [
        Prediction(
            task_id=prediction.task_id,
            model=prediction.model,
            response=extract_qwen_final_response(prediction.response),
        )
        for prediction in predictions
    ]


def score_predictions(tasks_paths: list[Path], predictions_path: Path) -> list[ComparedTask]:
    tasks = load_task_suites(tasks_paths)
    predictions = load_predictions(predictions_path, set(tasks))
    raw = evaluate_predictions(tasks, predictions)
    parsed = evaluate_predictions(tasks, qwen_final_predictions(predictions))
    return compare_scores(raw, parsed)


def compare_scores(raw: list[ScoreResult], parsed: list[ScoreResult]) -> list[ComparedTask]:
    parsed_by_key = {(result.model, result.task_id): result for result in parsed}
    compared: list[ComparedTask] = []
    for raw_result in raw:
        parsed_result = parsed_by_key[(raw_result.model, raw_result.task_id)]
        compared.append(
            ComparedTask(
                model=raw_result.model,
                task_id=raw_result.task_id,
                suite=raw_result.suite,
                capability=raw_result.capability,
                raw_passed=raw_result.passed,
                parsed_passed=parsed_result.passed,
                raw_reason=raw_result.reason,
                parsed_reason=parsed_result.reason,
            )
        )
    return compared


def grouped_by_capability(rows: list[ComparedTask]) -> dict[str, list[ComparedTask]]:
    groups: dict[str, list[ComparedTask]] = {}
    for row in rows:
        groups.setdefault(row.capability, []).append(row)
    return groups


def grouped_by_model_outcome(
    rows: list[ComparedTask],
    *,
    base_model: str,
    adapter_model: str,
) -> dict[str, list[str]]:
    parsed_by_key = {(row.model, row.task_id): row.parsed_passed for row in rows}
    task_ids = list(dict.fromkeys(row.task_id for row in rows if row.model == base_model))
    groups: dict[str, list[str]] = {
        "adapter_only": [],
        "base_only": [],
        "both_pass": [],
        "both_fail": [],
    }
    for task_id in task_ids:
        base_passed = parsed_by_key[(base_model, task_id)]
        adapter_passed = parsed_by_key[(adapter_model, task_id)]
        if adapter_passed and not base_passed:
            groups["adapter_only"].append(task_id)
        elif base_passed and not adapter_passed:
            groups["base_only"].append(task_id)
        elif base_passed and adapter_passed:
            groups["both_pass"].append(task_id)
        else:
            groups["both_fail"].append(task_id)
    return groups


def pass_count(rows: list[ComparedTask], *, parsed: bool) -> int:
    return sum(1 for row in rows if (row.parsed_passed if parsed else row.raw_passed))


def status(raw_passed: bool, parsed_passed: bool) -> str:
    if raw_passed and parsed_passed:
        return "pass"
    if raw_passed and not parsed_passed:
        return "raw_only"
    if not raw_passed and parsed_passed:
        return "parsed_only"
    return "fail"


def markdown_list(items: list[str]) -> list[str]:
    return [f"- `{item}`" for item in items] if items else ["- none"]


def write_report(path: Path, rows: list[ComparedTask], *, base_model: str, adapter_model: str) -> None:
    outcome_groups = grouped_by_model_outcome(rows, base_model=base_model, adapter_model=adapter_model)
    lines = [
        "# Leverage Post-Training Compare",
        "",
        "Compares raw decoded responses with Qwen final-response parsing.",
        "",
        "## Summary",
        "",
        "| raw passed | qwen-final passed | total |",
        "| ---: | ---: | ---: |",
        f"| {pass_count(rows, parsed=False)} | {pass_count(rows, parsed=True)} | {len(rows)} |",
        "",
        "## Adapter Only",
        "",
        *markdown_list(outcome_groups["adapter_only"]),
        "",
        "## Base Only",
        "",
        *markdown_list(outcome_groups["base_only"]),
        "",
        "## Both Pass",
        "",
        *markdown_list(outcome_groups["both_pass"]),
        "",
        "## Both Fail",
        "",
        *markdown_list(outcome_groups["both_fail"]),
        "",
        "## Details",
        "",
    ]
    for capability, capability_rows in sorted(grouped_by_capability(rows).items()):
        lines.extend(
            [
                f"## {capability}",
                "",
                "| model | suite | task | raw | qwen-final | status |",
                "| --- | --- | --- | ---: | ---: | --- |",
            ]
        )
        for row in capability_rows:
            lines.append(
                f"| `{row.model}` | `{row.suite}` | `{row.task_id}` | {format_passed(row.raw_passed)} | "
                f"{format_passed(row.parsed_passed)} | `{status(row.raw_passed, row.parsed_passed)}` |"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def format_passed(passed: bool) -> str:
    return "1" if passed else "0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, action="append", default=DEFAULT_TASKS)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter-model", default=DEFAULT_ADAPTER_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = score_predictions(args.tasks, args.predictions)
    write_report(args.output, rows, base_model=args.base_model, adapter_model=args.adapter_model)
    print(f"wrote report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
