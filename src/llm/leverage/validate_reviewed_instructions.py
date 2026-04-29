import argparse
import json
from pathlib import Path
import re
from typing import Any

from llm.leverage.capabilities import ALLOWED_CAPABILITIES


REQUIRED_ROW_FIELDS = {"id", "source_prompt_id", "capability", "messages", "review"}
REQUIRED_ROLES = ["system", "user", "assistant"]
ACCEPTED_STATUS = "accepted_instruction"
REVIEW_SOURCES = {"judge_accepted_candidate", "edited_candidate", "manual", "historical_reviewed"}
JUDGE_DECISIONS = {"accept", "needs_edit", "reject", "parse_error"}
SECRET_MARKERS = [
    "OPENAI_API_KEY",
    "RUNPOD_API_KEY",
    "BEGIN PRIVATE KEY",
    "ghp_",
]
SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
]


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


def eval_prompts(eval_dir: Path) -> set[str]:
    prompts: set[str] = set()
    for path in sorted(eval_dir.glob("*.jsonl")):
        for _line_number, row in load_jsonl(path):
            prompt = row.get("prompt")
            if isinstance(prompt, str):
                prompts.add(prompt)
    return prompts


def row_text(row: dict[str, Any]) -> str:
    parts: list[str] = []
    for value in row.values():
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    content = item.get("content")
                    if isinstance(content, str):
                        parts.append(content)
    return "\n".join(parts)


def secret_markers_in_text(text: str) -> list[str]:
    markers = [marker for marker in SECRET_MARKERS if marker in text]
    markers.extend(pattern.pattern for pattern in SECRET_PATTERNS if pattern.search(text))
    return markers


def validate_row(
    row: dict[str, Any],
    *,
    seen_ids: set[str],
    held_out_prompts: set[str],
) -> list[str]:
    errors: list[str] = []
    missing = REQUIRED_ROW_FIELDS - row.keys()
    if missing:
        errors.append(f"missing required fields: {sorted(missing)}")

    row_id = row.get("id")
    if not isinstance(row_id, str) or not row_id:
        errors.append("id must be a non-empty string")
    elif row_id in seen_ids:
        errors.append(f"duplicate id: {row_id}")
    else:
        seen_ids.add(row_id)

    source_prompt_id = row.get("source_prompt_id")
    if not isinstance(source_prompt_id, str) or not source_prompt_id:
        errors.append("source_prompt_id must be a non-empty string")

    capability = row.get("capability")
    if not isinstance(capability, str) or not capability:
        errors.append("capability must be a non-empty string")
    elif capability not in ALLOWED_CAPABILITIES:
        errors.append(f"capability must be one of {sorted(ALLOWED_CAPABILITIES)}")

    messages = row.get("messages")
    if not isinstance(messages, list):
        errors.append("messages must be a list")
    else:
        roles = [message.get("role") for message in messages if isinstance(message, dict)]
        if roles != REQUIRED_ROLES:
            errors.append(f"messages must have roles {REQUIRED_ROLES}")
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                errors.append(f"messages[{index}] must be an object")
                continue
            content = message.get("content")
            if not isinstance(content, str) or not content:
                errors.append(f"messages[{index}].content must be a non-empty string")
        if len(messages) >= 2 and isinstance(messages[1], dict):
            user_prompt = messages[1].get("content")
            if isinstance(user_prompt, str) and user_prompt in held_out_prompts:
                errors.append("user prompt duplicates a held-out eval prompt")

    review = row.get("review")
    if not isinstance(review, dict):
        errors.append("review must be an object")
    else:
        if review.get("status") != ACCEPTED_STATUS:
            errors.append(f"review.status must be {ACCEPTED_STATUS!r}")
        if not isinstance(review.get("author"), str) or not review["author"]:
            errors.append("review.author must be a non-empty string")
        if not isinstance(review.get("notes"), str) or not review["notes"]:
            errors.append("review.notes must be a non-empty string")
        source = review.get("source")
        if source is not None and source not in REVIEW_SOURCES:
            errors.append(f"review.source must be one of {sorted(REVIEW_SOURCES)}")
        for field_name in ("generator_model", "judge_model"):
            value = review.get(field_name)
            if value is not None and (not isinstance(value, str) or not value):
                errors.append(f"review.{field_name} must be a non-empty string when present")
        judge_decision = review.get("judge_decision")
        if judge_decision is not None and judge_decision not in JUDGE_DECISIONS:
            errors.append(f"review.judge_decision must be one of {sorted(JUDGE_DECISIONS)}")
        if source in {"judge_accepted_candidate", "edited_candidate"}:
            for field_name in ("generator_model", "judge_model", "judge_decision"):
                if field_name not in review:
                    errors.append(f"review.{field_name} is required for source {source!r}")

    text = row_text(row)
    for marker in secret_markers_in_text(text):
        errors.append(f"secret marker appears in row text: {marker}")

    return errors


def validate_file(path: Path, *, eval_dir: Path) -> list[str]:
    errors: list[str] = []
    seen_ids: set[str] = set()
    held_out_prompts = eval_prompts(eval_dir)
    rows = load_jsonl(path)
    if not rows:
        errors.append(f"{path}: file must contain at least one row")
    for line_number, row in rows:
        for error in validate_row(row, seen_ids=seen_ids, held_out_prompts=held_out_prompts):
            errors.append(f"{path}:{line_number}: {error}")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--eval-dir", type=Path, default=Path("tracks/leverage/evals"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    errors = validate_file(args.path, eval_dir=args.eval_dir)
    if errors:
        for error in errors:
            print(error)
        return 1
    print(f"validated {args.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
