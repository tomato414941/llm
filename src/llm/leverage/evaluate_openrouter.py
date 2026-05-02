import argparse
import json
from pathlib import Path

from llm.leverage.collect_openai import (
    DEFAULT_SYSTEM_PROMPT,
    chat_completions_client,
    collect_predictions,
    environment_value,
)
from llm.leverage.evaluate import evaluate_predictions, load_predictions, load_task_suites, write_results, write_summary


DEFAULT_MODELS = [
    "gpt-5-5-openrouter=openai/gpt-5.5",
    "claude-opus-4-7-openrouter=anthropic/claude-opus-4.7",
    "gemini-3-1-pro-preview-openrouter=google/gemini-3.1-pro-preview",
    "deepseek-v4-pro-openrouter=deepseek/deepseek-v4-pro",
    "qwen3-6-plus-openrouter=qwen/qwen3.6-plus",
]


def parse_model(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError("--model must use label=api_model")
    label, api_model = value.split("=", 1)
    if not label or not api_model:
        raise ValueError("--model must use non-empty label=api_model")
    return label, api_model


def output_paths(output_root: Path) -> tuple[Path, Path, Path, Path]:
    return (
        output_root / "openrouter-predictions.jsonl",
        output_root / "openrouter-scores.csv",
        output_root / "openrouter-summary.csv",
        output_root / "openrouter-run.json",
    )


def existing_prediction_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    return {(prediction.model, prediction.task_id) for prediction in load_predictions(path)}


def run_eval(
    *,
    tasks_paths: list[Path],
    models: list[tuple[str, str]],
    output_root: Path,
    max_tokens: int,
    temperature: float,
    system_prompt: str,
    timeout_seconds: float,
    thinking_mode: str,
    thinking_param: str,
    reasoning_effort: str,
    exclude_reasoning: bool,
    overwrite: bool,
    resume: bool,
    dry_run: bool,
) -> list[str]:
    tasks = load_task_suites(tasks_paths)
    predictions_path, scores_path, summary_path, run_metadata_path = output_paths(output_root)
    if dry_run:
        return [
            f"would evaluate {len(tasks)} tasks",
            "models: " + ", ".join(f"{label}={api_model}" for label, api_model in models),
            f"predictions output: {predictions_path}",
            f"scores output: {scores_path}",
            f"summary output: {summary_path}",
            f"run metadata output: {run_metadata_path}",
        ]

    if predictions_path.exists() and not overwrite and not resume:
        raise FileExistsError(f"output already exists: {predictions_path}")
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    if not resume:
        predictions_path.write_text("", encoding="utf-8")

    base_url = environment_value("OPENAI_BASE_URL")
    api_key = environment_value("OPENAI_API_KEY")
    client = chat_completions_client(base_url, api_key, timeout_seconds)
    for label, api_model in models:
        existing_keys = existing_prediction_keys(predictions_path) if resume else set()
        missing_tasks = {
            task_id: task
            for task_id, task in tasks.items()
            if (label, task_id) not in existing_keys
        }
        if not missing_tasks:
            continue
        collect_predictions(
            missing_tasks,
            client=client,
            output_path=predictions_path,
            api_model=api_model,
            model_label=label,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            thinking_mode=thinking_mode,
            thinking_param=thinking_param,
            reasoning_effort=reasoning_effort,
            exclude_reasoning=exclude_reasoning,
            overwrite=True,
            append=True,
        )

    predictions = load_predictions(predictions_path, set(tasks))
    results = evaluate_predictions(tasks, predictions)
    write_results(scores_path, results)
    write_summary(summary_path, results)
    write_run_metadata(
        run_metadata_path,
        tasks_paths=tasks_paths,
        models=models,
        max_tokens=max_tokens,
        temperature=temperature,
        system_prompt=system_prompt,
        timeout_seconds=timeout_seconds,
        thinking_mode=thinking_mode,
        thinking_param=thinking_param,
        reasoning_effort=reasoning_effort,
        exclude_reasoning=exclude_reasoning,
        overwrite=overwrite,
        resume=resume,
    )
    return [
        f"evaluated {len(tasks)} tasks with {len(models)} models",
        f"wrote predictions: {predictions_path}",
        f"wrote scores: {scores_path}",
        f"wrote summary: {summary_path}",
        f"wrote run metadata: {run_metadata_path}",
    ]


def write_run_metadata(
    path: Path,
    *,
    tasks_paths: list[Path],
    models: list[tuple[str, str]],
    max_tokens: int,
    temperature: float,
    system_prompt: str,
    timeout_seconds: float,
    thinking_mode: str,
    thinking_param: str,
    reasoning_effort: str,
    exclude_reasoning: bool,
    overwrite: bool,
    resume: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "tasks": [str(path) for path in tasks_paths],
                "models": [{"label": label, "api_model": api_model} for label, api_model in models],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system_prompt": system_prompt,
                "timeout_seconds": timeout_seconds,
                "thinking_mode": thinking_mode,
                "thinking_param": thinking_param,
                "reasoning_effort": reasoning_effort,
                "exclude_reasoning": exclude_reasoning,
                "overwrite": overwrite,
                "resume": resume,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True, action="append")
    parser.add_argument("--model", action="append", default=None)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/leverage-openrouter-eval"))
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument(
        "--thinking-mode",
        choices=("disabled", "none"),
        default="none",
        help="Use provider-specific thinking controls only when evaluating one known-compatible model.",
    )
    parser.add_argument(
        "--thinking-param",
        choices=("chat_template_kwargs", "enable_thinking"),
        default="chat_template_kwargs",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("provider_default", "none", "minimal", "low", "medium", "high", "xhigh"),
        default="provider_default",
    )
    parser.add_argument("--exclude-reasoning", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    if args.temperature < 0:
        raise ValueError("--temperature must be non-negative")
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive")


def main() -> int:
    args = parse_args()
    validate_args(args)
    model_args = args.model if args.model is not None else DEFAULT_MODELS
    models = [parse_model(value) for value in model_args]
    for line in run_eval(
        tasks_paths=args.tasks,
        models=models,
        output_root=args.output_root,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        system_prompt=args.system_prompt,
        timeout_seconds=args.timeout_seconds,
        thinking_mode=args.thinking_mode,
        thinking_param=args.thinking_param,
        reasoning_effort=args.reasoning_effort,
        exclude_reasoning=args.exclude_reasoning,
        overwrite=args.overwrite,
        resume=args.resume,
        dry_run=args.dry_run,
    ):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
