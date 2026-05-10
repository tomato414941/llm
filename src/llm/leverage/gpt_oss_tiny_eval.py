import argparse
import json
import time
from pathlib import Path
from typing import Any

from llm.leverage.evaluate import (
    Prediction,
    evaluate_predictions,
    load_task_suites,
    write_results,
    write_summary,
)
from llm.leverage.harmony import extract_harmony_final_response


DEFAULT_MODEL = "openai/gpt-oss-20b"
DEFAULT_OUTPUT_ROOT = Path("outputs/leverage-gpt-oss-20b-tiny-eval")
DEFAULT_SYSTEM_PROMPT = "Reasoning: low"


def selected_tasks(paths: list[Path], limit: int | None) -> dict[str, Any]:
    tasks = load_task_suites(paths)
    if limit is None:
        return tasks
    return dict(list(tasks.items())[:limit])


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_messages(prompt: str, system_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]


def generated_message(output: Any) -> Any:
    if isinstance(output, list) and output and isinstance(output[0], dict):
        generated = output[0].get("generated_text")
        if isinstance(generated, list) and generated:
            return generated[-1]
    return output


def run_tiny_eval(
    *,
    task_paths: list[Path],
    output_root: Path,
    model: str,
    system_prompt: str,
    limit: int | None,
    max_new_tokens: int,
) -> list[str]:
    import torch
    from transformers import pipeline

    tasks = selected_tasks(task_paths, limit)
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    pipe = pipeline("text-generation", model=model, torch_dtype="auto", device_map="auto")
    load_seconds = round(time.monotonic() - started, 3)
    raw_rows: list[dict[str, Any]] = []
    predictions: list[Prediction] = []
    for task in tasks.values():
        messages = build_messages(task.prompt, system_prompt)
        task_started = time.monotonic()
        output = pipe(messages, max_new_tokens=max_new_tokens, do_sample=False)
        raw_output = generated_message(output)
        final_response = extract_harmony_final_response(raw_output)
        elapsed = round(time.monotonic() - task_started, 3)
        raw_rows.append(
            {
                "task_id": task.id,
                "capability": task.capability,
                "prompt": task.prompt,
                "scoring": task.scoring,
                "messages": messages,
                "raw_output": raw_output,
                "final_response": final_response,
                "seconds": elapsed,
            }
        )
        predictions.append(Prediction(task_id=task.id, model=model, response=final_response))
    raw_path = output_root / "raw-generations.jsonl"
    predictions_path = output_root / "predictions.harmony-final.jsonl"
    scores_path = output_root / "scores.harmony-final.csv"
    summary_path = output_root / "summary.harmony-final.csv"
    metadata_path = output_root / "metadata.json"
    write_jsonl(raw_path, raw_rows)
    write_jsonl(
        predictions_path,
        [
            {"task_id": prediction.task_id, "model": prediction.model, "response": prediction.response}
            for prediction in predictions
        ],
    )
    results = evaluate_predictions(tasks, predictions)
    write_results(scores_path, results)
    write_summary(summary_path, results)
    metadata = {
        "model": model,
        "backend": "transformers.pipeline",
        "system_prompt": system_prompt,
        "task_paths": [str(path) for path in task_paths],
        "task_count": len(tasks),
        "load_seconds": load_seconds,
        "total_seconds": round(time.monotonic() - started, 3),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    if torch.cuda.is_available():
        metadata["max_memory_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return [
        f"evaluated {len(tasks)} tasks",
        f"wrote raw generations: {raw_path}",
        f"wrote final predictions: {predictions_path}",
        f"wrote scores: {scores_path}",
        f"wrote summary: {summary_path}",
        f"wrote metadata: {metadata_path}",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", action="append", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for line in run_tiny_eval(
        task_paths=args.task,
        output_root=args.output_root,
        model=args.model,
        system_prompt=args.system_prompt,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
    ):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
