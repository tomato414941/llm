import argparse
import csv
import json
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm.leverage.collect_instructions import InstructionSeed, load_jsonl, load_seeds
from llm.leverage.validate_reviewed_instructions import secret_markers_in_text


REQUIRED_ROLES = ["system", "user", "assistant"]
DEFAULT_MAX_RESPONSE_CHARS = 2200
WORD_COUNT_PATTERN = re.compile(r"\b(exactly|at most)\s+(\w+|\d+)\s+words?\b", re.IGNORECASE)
NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


@dataclass(frozen=True)
class FilterResult:
    source_prompt_id: str
    capability: str
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


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def _parse_word_count_constraint(constraint: str) -> tuple[str, int] | None:
    match = WORD_COUNT_PATTERN.search(constraint)
    if match is None:
        return None
    limit_text = match.group(2).lower()
    limit = int(limit_text) if limit_text.isdigit() else NUMBER_WORDS.get(limit_text)
    if limit is None:
        return None
    return match.group(1).lower(), limit


def deterministic_response_issues(seed: InstructionSeed, response: str) -> list[str]:
    issues: list[str] = []
    constraints = [constraint.lower() for constraint in seed.constraints]

    if seed.output_format == "json_object":
        stripped = response.strip()
        if stripped.startswith("```") or stripped.endswith("```"):
            issues.append("json_markdown_fence")
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            issues.append("invalid_json")
        else:
            if not isinstance(parsed, dict):
                issues.append("json_not_object")

    if any("no punctuation" in constraint for constraint in constraints):
        punctuation = set(string.punctuation)
        if any(char in punctuation for char in response):
            issues.append("punctuation_forbidden")

    count = _word_count(response)
    for constraint in seed.constraints:
        parsed_count = _parse_word_count_constraint(constraint)
        if parsed_count is None:
            continue
        kind, limit = parsed_count
        if kind == "exactly" and count != limit:
            issues.append(f"word_count_not_{limit}")
        elif kind == "at most" and count > limit:
            issues.append(f"word_count_over_{limit}")

    return issues


def filter_row(
    row: dict[str, Any],
    *,
    seeds: dict[str, InstructionSeed],
    max_response_chars: int,
) -> FilterResult:
    issues: list[str] = []
    source_prompt_id = row.get("source_prompt_id")
    capability = row.get("capability")
    model = row.get("model")
    if not isinstance(source_prompt_id, str) or not source_prompt_id:
        issues.append("missing_source_prompt_id")
        source_prompt_id = ""
    if not isinstance(capability, str) or not capability:
        issues.append("missing_capability")
        capability = ""
    if not isinstance(model, str) or not model:
        issues.append("missing_model")
        model = ""
    seed = seeds.get(source_prompt_id)
    if source_prompt_id and seed is None:
        issues.append("unknown_source_prompt_id")
    elif seed is not None and capability and seed.capability != capability:
        issues.append("capability_mismatch")

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
    if seed is not None and response:
        issues.extend(deterministic_response_issues(seed, response))

    text = row_text(row)
    for marker in secret_markers_in_text(text):
        issues.append(f"secret_marker:{marker}")

    decision = "needs_judge" if not issues else "reject"
    if issues == ["response_too_long"]:
        decision = "needs_judge"
    return FilterResult(
        source_prompt_id=source_prompt_id,
        capability=capability,
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
                "capability",
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
                    "capability": result.capability,
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


def summary_rows(results: list[tuple[int, dict[str, Any], FilterResult]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = len(results)
    rows.append({"scope": "overall", "name": "total", "count": total, "rate": "1.000" if total else "0.000"})

    decision_counts: dict[str, int] = {}
    capability_counts: dict[str, int] = {}
    issue_counts: dict[str, int] = {}
    for _line_number, _row, result in results:
        decision_counts[result.decision] = decision_counts.get(result.decision, 0) + 1
        capability_counts[result.capability] = capability_counts.get(result.capability, 0) + 1
        for issue in result.issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1

    for scope, counts in (
        ("decision", decision_counts),
        ("capability", capability_counts),
        ("issue", issue_counts),
    ):
        for name, count in sorted(counts.items()):
            rate = count / total if total else 0.0
            rows.append({"scope": scope, "name": name, "count": count, "rate": f"{rate:.3f}"})
    return rows


def write_summary_csv(path: Path, results: list[tuple[int, dict[str, Any], FilterResult]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["scope", "name", "count", "rate"])
        writer.writeheader()
        writer.writerows(summary_rows(results))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--seeds", type=Path, default=Path("tracks/leverage/prompts/instruction-seeds.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates-output", type=Path)
    parser.add_argument("--summary-output", type=Path)
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
    if args.summary_output is not None:
        write_summary_csv(args.summary_output, results)
    print(f"filtered {len(results)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
