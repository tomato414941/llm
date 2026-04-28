import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_ROW_FIELDS = {"id", "source_prompt_id", "category", "messages", "review"}
REQUIRED_ROLES = ["system", "user", "assistant"]
ACCEPTED_STATUS = "accepted_candidate"
SECRET_MARKERS = [
    "OPENAI_API_KEY",
    "RUNPOD_API_KEY",
    "BEGIN PRIVATE KEY",
    "ghp_",
    "sk-",
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

    category = row.get("category")
    if not isinstance(category, str) or not category:
        errors.append("category must be a non-empty string")

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

    text = row_text(row)
    for marker in SECRET_MARKERS:
        if marker in text:
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
    parser.add_argument("--eval-dir", type=Path, default=Path("evals"))
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
