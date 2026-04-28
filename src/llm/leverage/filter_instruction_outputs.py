import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm.leverage.collect_instructions import InstructionSeed, load_jsonl, load_seeds
from llm.leverage.validate_reviewed_instructions import secret_markers_in_text


REQUIRED_ROLES = ["system", "user", "assistant"]
DEFAULT_MAX_RESPONSE_CHARS = 2200


@dataclass(frozen=True)
class FilterResult:
    source_prompt_id: str
    category: str
    model: str
    decision: str
    issues: list[str]
    response_chars: int


def message_content(row: dict[str, Any], role: str) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return ""
    for message in messages:
        if isinstance(message, dict) and message.get("role") == role:
            content = message.get("content")
            if isinstance(content, str):
                return content
    return ""


def row_text(row: dict[str, Any]) -> str:
    parts: list[str] = []
    for value in row.values():
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    content = item.get("content")
                    if isinstance(content, str):
                        parts.append(content)
    return "\n".join(parts)


def filter_row(
    row: dict[str, Any],
    *,
    seeds: dict[str, InstructionSeed],
    max_response_chars: int,
) -> FilterResult:
    issues: list[str] = []
    source_prompt_id = row.get("source_prompt_id")
    category = row.get("category")
    model = row.get("model")
    if not isinstance(source_prompt_id, str) or not source_prompt_id:
        issues.append("missing_source_prompt_id")
        source_prompt_id = ""
    if not isinstance(category, str) or not category:
        issues.append("missing_category")
        category = ""
    if not isinstance(model, str) or not model:
        issues.append("missing_model")
        model = ""
    seed = seeds.get(source_prompt_id)
    if source_prompt_id and seed is None:
        issues.append("unknown_source_prompt_id")
    elif seed is not None and category and seed.category != category:
        issues.append("category_mismatch")

    messages = row.get("messages")
    if not isinstance(messages, list):
        issues.append("messages_not_list")
    else:
        roles = [message.get("role") for message in messages if isinstance(message, dict)]
        if roles != REQUIRED_ROLES:
            issues.append("bad_message_roles")
    review = row.get("review")
    if not isinstance(review, dict) or review.get("status") != "raw":
        issues.append("review_status_not_raw")

    response = row.get("raw_response")
    assistant_content = message_content(row, "assistant")
    if not isinstance(response, str) or not response.strip():
        issues.append("empty_raw_response")
        response = ""
    if assistant_content != response:
        issues.append("assistant_raw_response_mismatch")
    if len(response) > max_response_chars:
        issues.append("response_too_long")

    text = row_text(row)
    for marker in secret_markers_in_text(text):
        issues.append(f"secret_marker:{marker}")

    decision = "needs_judge" if not issues else "reject"
    if issues == ["response_too_long"]:
        decision = "needs_judge"
    return FilterResult(
        source_prompt_id=source_prompt_id,
        category=category,
        model=model,
        decision=decision,
        issues=issues,
        response_chars=len(response),
    )


def filter_rows(
    rows: list[tuple[int, dict[str, Any]]],
    *,
    seeds: dict[str, InstructionSeed],
    max_response_chars: int,
) -> list[tuple[int, dict[str, Any], FilterResult]]:
    return [
        (line_number, row, filter_row(row, seeds=seeds, max_response_chars=max_response_chars))
        for line_number, row in rows
    ]


def write_csv(path: Path, results: list[tuple[int, dict[str, Any], FilterResult]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "line_number",
                "source_prompt_id",
                "category",
                "model",
                "decision",
                "issue_count",
                "issues",
                "response_chars",
            ],
        )
        writer.writeheader()
        for line_number, _row, result in results:
            writer.writerow(
                {
                    "line_number": line_number,
                    "source_prompt_id": result.source_prompt_id,
                    "category": result.category,
                    "model": result.model,
                    "decision": result.decision,
                    "issue_count": len(result.issues),
                    "issues": ";".join(result.issues),
                    "response_chars": result.response_chars,
                }
            )


def write_candidate_jsonl(
    path: Path,
    results: list[tuple[int, dict[str, Any], FilterResult]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for _line_number, row, result in results:
            if result.decision == "needs_judge":
                output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--seeds", type=Path, default=Path("prompts/leverage-training-seed-v0.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates-output", type=Path)
    parser.add_argument("--max-response-chars", type=int, default=DEFAULT_MAX_RESPONSE_CHARS)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.max_response_chars <= 0:
        raise ValueError("--max-response-chars must be positive")


def main() -> int:
    args = parse_args()
    validate_args(args)
    seeds = load_seeds(args.seeds)
    rows = load_jsonl(args.input)
    results = filter_rows(rows, seeds=seeds, max_response_chars=args.max_response_chars)
    write_csv(args.output, results)
    if args.candidates_output is not None:
        write_candidate_jsonl(args.candidates_output, results)
    print(f"filtered {len(results)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
