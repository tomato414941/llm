import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from llm.leverage.evaluate import (
    Prediction,
    ScoreResult,
    evaluate_predictions,
    load_predictions,
    load_task_suites,
    write_results,
    write_summary,
)
from llm.leverage.evaluate_sft_adapter import extract_qwen_final_response


DEFAULT_OUTPUT_ROOT = Path("outputs/leverage-sft-smoke")
DEFAULT_TASKS = [
    Path("tracks/leverage/evals/leverage-smoke.jsonl"),
    Path("tracks/leverage/evals/project-judgment-v0.jsonl"),
]
DEFAULT_REPORT = Path("tracks/leverage/runs/leverage-sft-smoke-diff.md")


@dataclass(frozen=True)
class ScoreRow:
    model: str
    suite: str
    task_id: str
    category: str
    passed: bool
    reason: str


def score_row(result: ScoreResult) -> ScoreRow:
    return ScoreRow(
        model=result.model,
        suite=result.suite,
        task_id=result.task_id,
        category=result.category,
        passed=result.passed,
        reason=result.reason,
    )


def score_predictions(
    *,
    tasks_paths: list[Path],
    predictions_path: Path,
    parse_qwen_final: bool,
) -> dict[tuple[str, str], ScoreRow]:
    tasks = load_task_suites(tasks_paths)
    predictions = load_predictions(predictions_path, set(tasks))
    if parse_qwen_final:
        predictions = [
            Prediction(
                task_id=prediction.task_id,
                model=prediction.model,
                response=extract_qwen_final_response(prediction.response),
            )
            for prediction in predictions
        ]
    return {
        (result.model, result.task_id): score_row(result)
        for result in evaluate_predictions(tasks, predictions)
    }


def read_scores(path: Path) -> dict[tuple[str, str], ScoreRow]:
    rows: dict[tuple[str, str], ScoreRow] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            score = ScoreRow(
                model=row["model"],
                suite=row["suite"],
                task_id=row["task_id"],
                category=row["category"],
                passed=row["passed"] == "true",
                reason=row["reason"],
            )
            rows[(score.model, score.task_id)] = score
    return rows


def task_order(scores: dict[tuple[str, str], ScoreRow]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for _key, score in scores.items():
        if score.task_id not in seen:
            seen.add(score.task_id)
            ordered.append(score.task_id)
    return ordered


def classify_task(
    *,
    task_id: str,
    base_model: str,
    adapter_model: str,
    raw_scores: dict[tuple[str, str], ScoreRow],
    parsed_scores: dict[tuple[str, str], ScoreRow],
) -> str:
    base_raw = raw_scores[(base_model, task_id)].passed
    adapter_raw = raw_scores[(adapter_model, task_id)].passed
    base_parsed = parsed_scores[(base_model, task_id)].passed
    adapter_parsed = parsed_scores[(adapter_model, task_id)].passed
    if not base_raw and not adapter_raw and (base_parsed or adapter_parsed):
        return "recovered_by_qwen_final_parse"
    if base_parsed and not adapter_parsed:
        return "base_only"
    if adapter_parsed and not base_parsed:
        return "adapter_only"
    if base_parsed and adapter_parsed:
        return "both_pass"
    return "both_fail"


def grouped_tasks(
    *,
    base_model: str,
    adapter_model: str,
    raw_scores: dict[tuple[str, str], ScoreRow],
    parsed_scores: dict[tuple[str, str], ScoreRow],
) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {
        "recovered_by_qwen_final_parse": [],
        "adapter_only": [],
        "base_only": [],
        "both_pass": [],
        "both_fail": [],
    }
    for task_id in task_order(raw_scores):
        if (base_model, task_id) not in raw_scores:
            continue
        groups[
            classify_task(
                task_id=task_id,
                base_model=base_model,
                adapter_model=adapter_model,
                raw_scores=raw_scores,
                parsed_scores=parsed_scores,
            )
        ].append(task_id)
    return groups


def pass_count(scores: dict[tuple[str, str], ScoreRow], model: str) -> int:
    return sum(1 for (score_model, _task_id), row in scores.items() if score_model == model and row.passed)


def task_count(scores: dict[tuple[str, str], ScoreRow], model: str) -> int:
    return sum(1 for score_model, _task_id in scores if score_model == model)


def markdown_list(items: list[str]) -> list[str]:
    if not items:
        return ["- none"]
    return [f"- `{item}`" for item in items]


def write_score_artifacts(
    *,
    raw_output: Path,
    parsed_output: Path,
    summary_output: Path,
    tasks_paths: list[Path],
    predictions_path: Path,
) -> None:
    tasks = load_task_suites(tasks_paths)
    raw_predictions = load_predictions(predictions_path, set(tasks))
    parsed_predictions = [
        Prediction(
            task_id=prediction.task_id,
            model=prediction.model,
            response=extract_qwen_final_response(prediction.response),
        )
        for prediction in raw_predictions
    ]
    raw_results = evaluate_predictions(tasks, raw_predictions)
    parsed_results = evaluate_predictions(tasks, parsed_predictions)
    write_results(raw_output, raw_results)
    write_results(parsed_output, parsed_results)
    write_summary(summary_output, parsed_results)


def write_report(
    *,
    path: Path,
    base_model: str,
    adapter_model: str,
    raw_scores: dict[tuple[str, str], ScoreRow],
    parsed_scores: dict[tuple[str, str], ScoreRow],
) -> None:
    groups = grouped_tasks(
        base_model=base_model,
        adapter_model=adapter_model,
        raw_scores=raw_scores,
        parsed_scores=parsed_scores,
    )
    total = task_count(raw_scores, base_model)
    lines = [
        "# Leverage SFT Smoke Diff",
        "",
        "This report compares raw decoded scoring with Qwen final-response scoring.",
        "",
        "## Summary",
        "",
        "| model | raw passed | qwen-final passed | total |",
        "| --- | ---: | ---: | ---: |",
        f"| `{base_model}` | {pass_count(raw_scores, base_model)} | {pass_count(parsed_scores, base_model)} | {total} |",
        f"| `{adapter_model}` | {pass_count(raw_scores, adapter_model)} | {pass_count(parsed_scores, adapter_model)} | {total} |",
        "",
        "## Recovered By Qwen Final Parse",
        "",
        *markdown_list(groups["recovered_by_qwen_final_parse"]),
        "",
        "## Adapter Only",
        "",
        *markdown_list(groups["adapter_only"]),
        "",
        "## Base Only",
        "",
        *markdown_list(groups["base_only"]),
        "",
        "## Both Pass",
        "",
        *markdown_list(groups["both_pass"]),
        "",
        "## Both Fail",
        "",
        *markdown_list(groups["both_fail"]),
        "",
        "## Next Data Target",
        "",
        "Prioritize `both_fail` project-judgment tasks before another training run.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, action="append", default=DEFAULT_TASKS)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "post-training-predictions.jsonl",
    )
    parser.add_argument("--base-model", default="qwen3-0.6b-base")
    parser.add_argument("--adapter-model", default="qwen3-0.6b-lora-smoke")
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--raw-scores-output", type=Path, default=DEFAULT_OUTPUT_ROOT / "post-training-scores.raw.csv")
    parser.add_argument(
        "--qwen-final-scores-output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "post-training-scores.qwen-final.csv",
    )
    parser.add_argument(
        "--qwen-final-summary-output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "post-training-summary.qwen-final.csv",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_scores = score_predictions(
        tasks_paths=args.tasks,
        predictions_path=args.predictions,
        parse_qwen_final=False,
    )
    parsed_scores = score_predictions(
        tasks_paths=args.tasks,
        predictions_path=args.predictions,
        parse_qwen_final=True,
    )
    write_score_artifacts(
        raw_output=args.raw_scores_output,
        parsed_output=args.qwen_final_scores_output,
        summary_output=args.qwen_final_summary_output,
        tasks_paths=args.tasks,
        predictions_path=args.predictions,
    )
    write_report(
        path=args.output,
        base_model=args.base_model,
        adapter_model=args.adapter_model,
        raw_scores=raw_scores,
        parsed_scores=parsed_scores,
    )
    print(f"wrote report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
