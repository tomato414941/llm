import argparse
import csv
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from llm.leverage.harmony import HarmonyExtraction, analyze_harmony_response


DEFAULT_MODEL = "openai/gpt-oss-20b"
DEFAULT_OUTPUT_ROOT = Path("outputs/leverage-gpt-oss-20b-gsm8k")
DEFAULT_SYSTEM_PROMPT = "Reasoning: low"
ANSWER_RE = re.compile(r"####\s*([^\n]+)")
NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


@dataclass(frozen=True)
class Gsm8kTask:
    id: str
    question: str
    answer: str


@dataclass(frozen=True)
class Gsm8kPrediction:
    task_id: str
    expected: str
    prediction: str
    final_response: str
    passed: bool


def normalize_number(value: str) -> str:
    return value.strip().replace(",", "")


def extract_expected_answer(answer: str) -> str:
    match = ANSWER_RE.search(answer)
    if not match:
        raise ValueError(f"GSM8K answer is missing final marker: {answer!r}")
    return normalize_number(match.group(1))


def extract_numeric_prediction(response: str) -> str:
    matches = NUMBER_RE.findall(response)
    if not matches:
        return ""
    return normalize_number(matches[-1])


def build_messages(question: str, system_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "Solve the grade-school math problem. "
                "Return only the final numeric answer, with no explanation.\n\n"
                f"Problem: {question}"
            ),
        },
    ]


def generated_message(output: Any) -> Any:
    if isinstance(output, list) and output and isinstance(output[0], dict):
        generated = output[0].get("generated_text")
        if isinstance(generated, list) and generated:
            return generated[-1]
    return output


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_gsm8k_tasks(split: str, limit: int | None) -> list[Gsm8kTask]:
    from datasets import load_dataset

    dataset = load_dataset("gsm8k", "main", split=split)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    tasks: list[Gsm8kTask] = []
    for index, row in enumerate(dataset):
        tasks.append(
            Gsm8kTask(
                id=f"gsm8k_{split}_{index:04d}",
                question=row["question"],
                answer=row["answer"],
            )
        )
    return tasks


def summarize_extractions(extractions: list[HarmonyExtraction]) -> dict[str, int]:
    return {
        "task_count": len(extractions),
        "missing_final_marker_count": sum(1 for extraction in extractions if not extraction.final_marker_found),
        "empty_final_response_count": sum(1 for extraction in extractions if extraction.final_response_empty),
        "non_final_channel_in_final_count": sum(
            1 for extraction in extractions if extraction.non_final_channel_in_final
        ),
    }


def write_scores(path: Path, predictions: list[Gsm8kPrediction]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["task_id", "expected", "prediction", "passed", "final_response"],
        )
        writer.writeheader()
        for prediction in predictions:
            writer.writerow(
                {
                    "task_id": prediction.task_id,
                    "expected": prediction.expected,
                    "prediction": prediction.prediction,
                    "passed": str(prediction.passed).lower(),
                    "final_response": prediction.final_response,
                }
            )


def run_gsm8k_eval(
    *,
    output_root: Path,
    model: str,
    system_prompt: str,
    split: str,
    limit: int | None,
    max_new_tokens: int,
    task_loader: Callable[[str, int | None], list[Gsm8kTask]] = load_gsm8k_tasks,
    pipeline_factory: Callable[..., Any] | None = None,
) -> list[str]:
    import torch

    output_root.mkdir(parents=True, exist_ok=True)
    tasks = task_loader(split, limit)
    started = time.monotonic()
    if pipeline_factory is None:
        from transformers import pipeline

        pipeline_factory = pipeline
    pipe = pipeline_factory("text-generation", model=model, torch_dtype="auto", device_map="auto")
    load_seconds = round(time.monotonic() - started, 3)

    raw_rows: list[dict[str, Any]] = []
    predictions: list[Gsm8kPrediction] = []
    extractions: list[HarmonyExtraction] = []
    for task in tasks:
        expected = extract_expected_answer(task.answer)
        messages = build_messages(task.question, system_prompt)
        task_started = time.monotonic()
        output = pipe(messages, max_new_tokens=max_new_tokens, do_sample=False)
        raw_output = generated_message(output)
        extraction = analyze_harmony_response(raw_output)
        prediction = extract_numeric_prediction(extraction.final_response)
        passed = prediction == expected
        elapsed = round(time.monotonic() - task_started, 3)
        extractions.append(extraction)
        predictions.append(
            Gsm8kPrediction(
                task_id=task.id,
                expected=expected,
                prediction=prediction,
                final_response=extraction.final_response,
                passed=passed,
            )
        )
        raw_rows.append(
            {
                "task_id": task.id,
                "question": task.question,
                "answer": task.answer,
                "expected": expected,
                "messages": messages,
                "raw_output": raw_output,
                "final_response": extraction.final_response,
                "prediction": prediction,
                "passed": passed,
                "harmony": {
                    "final_marker_found": extraction.final_marker_found,
                    "final_response_empty": extraction.final_response_empty,
                    "non_final_channel_in_final": extraction.non_final_channel_in_final,
                },
                "seconds": elapsed,
            }
        )

    raw_path = output_root / "raw-generations.jsonl"
    predictions_path = output_root / "predictions.harmony-final.jsonl"
    scores_path = output_root / "scores.harmony-final.csv"
    metadata_path = output_root / "metadata.json"
    write_jsonl(raw_path, raw_rows)
    write_jsonl(
        predictions_path,
        [
            {
                "task_id": prediction.task_id,
                "model": model,
                "expected": prediction.expected,
                "prediction": prediction.prediction,
                "response": prediction.final_response,
                "passed": prediction.passed,
            }
            for prediction in predictions
        ],
    )
    write_scores(scores_path, predictions)

    passed_count = sum(1 for prediction in predictions if prediction.passed)
    task_count = len(predictions)
    metadata: dict[str, Any] = {
        "model": model,
        "backend": "transformers.pipeline",
        "system_prompt": system_prompt,
        "dataset": "gsm8k",
        "dataset_config": "main",
        "split": split,
        "task_count": task_count,
        "passed_count": passed_count,
        "accuracy": passed_count / task_count if task_count else 0.0,
        "load_seconds": load_seconds,
        "total_seconds": round(time.monotonic() - started, 3),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "harmony": summarize_extractions(extractions),
    }
    if torch.cuda.is_available():
        metadata["max_memory_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return [
        f"evaluated {task_count} GSM8K tasks",
        f"accuracy: {passed_count}/{task_count} = {metadata['accuracy']:.3f}",
        f"wrote raw generations: {raw_path}",
        f"wrote final predictions: {predictions_path}",
        f"wrote scores: {scores_path}",
        f"wrote metadata: {metadata_path}",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for line in run_gsm8k_eval(
        output_root=args.output_root,
        model=args.model,
        system_prompt=args.system_prompt,
        split=args.split,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
    ):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
