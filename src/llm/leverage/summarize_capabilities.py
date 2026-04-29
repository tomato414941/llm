import argparse
import csv
import json
from collections import Counter
from itertools import combinations
from pathlib import Path
import re
from typing import Any

from llm.leverage.capabilities import ALLOWED_CAPABILITIES
from llm.leverage.validate_reviewed_instructions import TASK_SHAPES


DEFAULT_SEEDS = Path("tracks/leverage/prompts/instruction-seeds.jsonl")
DEFAULT_REVIEWED = Path("tracks/leverage/datasets/reviewed-instructions/bootstrap.jsonl")
DEFAULT_EVALS = [
    Path("tracks/leverage/evals/leverage-smoke.jsonl"),
    Path("tracks/leverage/evals/project-judgment.jsonl"),
    Path("tracks/leverage/evals/leverage-model-spec.jsonl"),
]
DEFAULT_OUTPUT = Path("tracks/leverage/runs/capability-distribution.csv")
DEFAULT_PROVENANCE_OUTPUT = Path("tracks/leverage/runs/reviewed-provenance-distribution.csv")
DEFAULT_TASK_SHAPE_OUTPUT = Path("tracks/leverage/runs/reviewed-task-shape-distribution.csv")
DEFAULT_NEAR_DUPLICATE_OUTPUT = Path("tracks/leverage/runs/reviewed-near-duplicates.csv")
REVIEWED_TARGETS = {
    "instruction_following": 80,
    "reasoning": 80,
    "coding": 50,
    "knowledge_qa": 30,
    "summarization_transformation": 25,
    "tool_use": 35,
}
DEFAULT_REVIEW_SOURCE = "historical_reviewed"
TOKEN_PATTERN = re.compile(r"[a-z0-9_]+")


def load_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"{path}:{line_number}: blank lines are not allowed")
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number}: row must be a JSON object")
        rows.append((line_number, payload))
    return rows


def count_capabilities(path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    for line_number, row in load_jsonl(path):
        capability = row.get("capability")
        if not isinstance(capability, str) or not capability:
            raise ValueError(f"{path}:{line_number}: capability must be a non-empty string")
        if capability not in ALLOWED_CAPABILITIES:
            raise ValueError(f"{path}:{line_number}: unknown capability: {capability}")
        counts[capability] += 1
    return counts


def add_counts(total: Counter[str], path: Path) -> None:
    total.update(count_capabilities(path))


def summary_rows(
    *,
    seeds_path: Path,
    reviewed_path: Path,
    eval_paths: list[Path],
    reviewed_targets: dict[str, int] = REVIEWED_TARGETS,
) -> list[dict[str, str]]:
    seed_counts = count_capabilities(seeds_path)
    reviewed_counts = count_capabilities(reviewed_path)
    eval_counts: Counter[str] = Counter()
    for eval_path in eval_paths:
        add_counts(eval_counts, eval_path)

    rows: list[dict[str, str]] = []
    for capability in sorted(ALLOWED_CAPABILITIES):
        reviewed_count = reviewed_counts[capability]
        target = reviewed_targets[capability]
        deficit = max(target - reviewed_count, 0)
        rows.append(
            {
                "capability": capability,
                "seed_count": str(seed_counts[capability]),
                "reviewed_count": str(reviewed_count),
                "eval_count": str(eval_counts[capability]),
                "reviewed_target": str(target),
                "reviewed_deficit": str(deficit),
            }
        )
    return rows


def review_source(row: dict[str, Any]) -> str:
    review = row.get("review")
    if not isinstance(review, dict):
        return DEFAULT_REVIEW_SOURCE
    source = review.get("source")
    return source if isinstance(source, str) and source else DEFAULT_REVIEW_SOURCE


def provenance_rows(reviewed_path: Path) -> list[dict[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for line_number, row in load_jsonl(reviewed_path):
        capability = row.get("capability")
        if not isinstance(capability, str) or not capability:
            raise ValueError(f"{reviewed_path}:{line_number}: capability must be a non-empty string")
        if capability not in ALLOWED_CAPABILITIES:
            raise ValueError(f"{reviewed_path}:{line_number}: unknown capability: {capability}")
        counts[(capability, review_source(row))] += 1
    return [
        {"capability": capability, "source": source, "count": str(count)}
        for (capability, source), count in sorted(counts.items())
    ]


def task_shape_rows(reviewed_path: Path) -> list[dict[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for line_number, row in load_jsonl(reviewed_path):
        capability = row.get("capability")
        if not isinstance(capability, str) or not capability:
            raise ValueError(f"{reviewed_path}:{line_number}: capability must be a non-empty string")
        if capability not in ALLOWED_CAPABILITIES:
            raise ValueError(f"{reviewed_path}:{line_number}: unknown capability: {capability}")
        task_shape = row.get("task_shape")
        if not isinstance(task_shape, str) or not task_shape:
            raise ValueError(f"{reviewed_path}:{line_number}: task_shape must be a non-empty string")
        if task_shape not in TASK_SHAPES:
            raise ValueError(f"{reviewed_path}:{line_number}: unknown task_shape: {task_shape}")
        counts[(capability, task_shape)] += 1
    return [
        {"capability": capability, "task_shape": task_shape, "count": str(count)}
        for (capability, task_shape), count in sorted(counts.items())
    ]


def user_prompt(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return ""
    message = messages[1]
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    return content if isinstance(content, str) else ""


def prompt_tokens(prompt: str) -> set[str]:
    return set(TOKEN_PATTERN.findall(prompt.lower()))


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def near_duplicate_rows(reviewed_path: Path, *, limit: int = 50) -> list[dict[str, str]]:
    prompts: list[dict[str, Any]] = []
    for line_number, row in load_jsonl(reviewed_path):
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError(f"{reviewed_path}:{line_number}: id must be a non-empty string")
        capability = row.get("capability")
        if not isinstance(capability, str) or not capability:
            raise ValueError(f"{reviewed_path}:{line_number}: capability must be a non-empty string")
        task_shape = row.get("task_shape")
        if not isinstance(task_shape, str) or not task_shape:
            raise ValueError(f"{reviewed_path}:{line_number}: task_shape must be a non-empty string")
        prompt = user_prompt(row)
        prompts.append(
            {
                "id": row_id,
                "capability": capability,
                "task_shape": task_shape,
                "prompt": prompt,
                "tokens": prompt_tokens(prompt),
            }
        )

    pairs: list[dict[str, str]] = []
    for left, right in combinations(prompts, 2):
        similarity = jaccard_similarity(left["tokens"], right["tokens"])
        pairs.append(
            {
                "left_id": left["id"],
                "right_id": right["id"],
                "similarity": f"{similarity:.3f}",
                "left_capability": left["capability"],
                "right_capability": right["capability"],
                "left_task_shape": left["task_shape"],
                "right_task_shape": right["task_shape"],
                "left_prompt": left["prompt"],
                "right_prompt": right["prompt"],
            }
        )
    pairs.sort(key=lambda row: (-float(row["similarity"]), row["left_id"], row["right_id"]))
    return pairs[:limit]


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "capability",
        "seed_count",
        "reviewed_count",
        "eval_count",
        "reviewed_target",
        "reviewed_deficit",
    ]
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_provenance_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["capability", "source", "count"])
        writer.writeheader()
        writer.writerows(rows)


def write_task_shape_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["capability", "task_shape", "count"])
        writer.writeheader()
        writer.writerows(rows)


def write_near_duplicate_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "left_id",
                "right_id",
                "similarity",
                "left_capability",
                "right_capability",
                "left_task_shape",
                "right_task_shape",
                "left_prompt",
                "right_prompt",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=Path, default=DEFAULT_SEEDS)
    parser.add_argument("--reviewed", type=Path, default=DEFAULT_REVIEWED)
    parser.add_argument("--eval", type=Path, action="append", dest="evals", default=DEFAULT_EVALS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--provenance-output", type=Path, default=DEFAULT_PROVENANCE_OUTPUT)
    parser.add_argument("--task-shape-output", type=Path, default=DEFAULT_TASK_SHAPE_OUTPUT)
    parser.add_argument("--near-duplicate-output", type=Path, default=DEFAULT_NEAR_DUPLICATE_OUTPUT)
    parser.add_argument("--near-duplicate-limit", type=int, default=50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = summary_rows(seeds_path=args.seeds, reviewed_path=args.reviewed, eval_paths=args.evals)
    write_csv(args.output, rows)
    write_provenance_csv(args.provenance_output, provenance_rows(args.reviewed))
    write_task_shape_csv(args.task_shape_output, task_shape_rows(args.reviewed))
    write_near_duplicate_csv(args.near_duplicate_output, near_duplicate_rows(args.reviewed, limit=args.near_duplicate_limit))
    print(f"wrote capability distribution: {args.output}")
    print(f"wrote reviewed provenance distribution: {args.provenance_output}")
    print(f"wrote reviewed task-shape distribution: {args.task_shape_output}")
    print(f"wrote reviewed near-duplicate signal: {args.near_duplicate_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
