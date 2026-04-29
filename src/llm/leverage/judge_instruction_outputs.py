import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Any, Callable

from llm.leverage.collect_instructions import load_jsonl
from llm.leverage.collect_openai import ChatResult, chat_completions_client


DEFAULT_JUDGE_MODEL = "anthropic/claude-sonnet-4.6"
DEFAULT_JUDGE_LABEL = "claude-sonnet-4-6-openrouter"
ChatClient = Callable[[dict[str, Any]], ChatResult | str]
JudgeCandidate = tuple[str, str, float]


def answer_id(row: dict[str, Any]) -> str:
    source_prompt_id = row.get("source_prompt_id")
    generator_model = generator_model_label(row)
    if isinstance(source_prompt_id, str) and source_prompt_id and generator_model:
        return f"{source_prompt_id}:{generator_model}"
    return ""


def generator_model_label(row: dict[str, Any]) -> str:
    model = row.get("generator_model", row.get("model"))
    return model if isinstance(model, str) else ""


def assistant_content(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "assistant":
                content = message.get("content")
                if isinstance(content, str):
                    return content
    raw_response = row.get("raw_response")
    return raw_response if isinstance(raw_response, str) else ""


def user_prompt(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, str):
                    return content
    return ""


def build_judge_prompt(row: dict[str, Any]) -> str:
    constraints = row.get("constraints")
    constraints_text = "\n".join(f"- {item}" for item in constraints if isinstance(item, str))
    return (
        "Evaluate this candidate instruction answer for possible training-data use.\n"
        "Return only JSON with keys: scores, decision, reason.\n"
        "scores must contain integer values from 0 to 2 for correctness, "
        "instruction_following, conciseness, and safety.\n"
        "decision must be one of accept, needs_edit, reject.\n\n"
        f"source_prompt_id: {row.get('source_prompt_id', '')}\n"
        f"output_format: {row.get('output_format', '')}\n"
        f"constraints:\n{constraints_text}\n\n"
        f"user_prompt:\n{user_prompt(row)}\n\n"
        f"candidate_answer:\n{assistant_content(row)}"
    )


def build_payload(
    row: dict[str, Any],
    *,
    judge_model: str,
    max_tokens: int,
    temperature: float,
    reasoning_effort: str,
    exclude_reasoning: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": judge_model,
        "messages": [
            {
                "role": "system",
                "content": "You are a strict data-quality judge. Return JSON only.",
            },
            {"role": "user", "content": build_judge_prompt(row)},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if reasoning_effort != "provider_default" or exclude_reasoning:
        reasoning: dict[str, Any] = {"exclude": exclude_reasoning}
        if reasoning_effort != "provider_default":
            reasoning["effort"] = reasoning_effort
        payload["reasoning"] = reasoning
    return payload


def parse_judge_response(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    payload = json.loads(cleaned)
    if not isinstance(payload, dict):
        raise ValueError("judge response must be a JSON object")
    return payload


def normalize_judgment(payload: dict[str, Any]) -> tuple[dict[str, int], str, str]:
    scores_payload = payload.get("scores")
    if not isinstance(scores_payload, dict):
        raise ValueError("judge response must contain scores object")
    scores: dict[str, int] = {}
    for name in ("correctness", "instruction_following", "conciseness", "safety"):
        value = scores_payload.get(name)
        if not isinstance(value, int) or value < 0 or value > 2:
            raise ValueError(f"judge score must be an integer from 0 to 2: {name}")
        scores[name] = value
    decision = payload.get("decision")
    if decision not in {"accept", "needs_edit", "reject"}:
        raise ValueError("judge decision must be accept, needs_edit, or reject")
    reason = payload.get("reason")
    if not isinstance(reason, str) or not reason:
        raise ValueError("judge reason must be a non-empty string")
    return scores, decision, reason


def judgment_record(
    row: dict[str, Any],
    *,
    judge_model: str,
    judge_label: str,
    judge_response: str,
) -> dict[str, Any]:
    payload = parse_judge_response(judge_response)
    scores, decision, reason = normalize_judgment(payload)
    return {
        "source_prompt_id": row.get("source_prompt_id", ""),
        "answer_id": answer_id(row),
        "generator_model": generator_model_label(row),
        "judge_model": judge_label,
        "judge_api_model": judge_model,
        "scores": scores,
        "decision": decision,
        "reason": reason,
        "raw_judge_response": judge_response,
    }


def failed_judgment_record(
    row: dict[str, Any],
    *,
    judge_model: str,
    judge_label: str,
    judge_response: str,
    error: Exception,
) -> dict[str, Any]:
    return {
        "source_prompt_id": row.get("source_prompt_id", ""),
        "answer_id": answer_id(row),
        "generator_model": generator_model_label(row),
        "judge_model": judge_label,
        "judge_api_model": judge_model,
        "scores": {
            "correctness": 0,
            "instruction_following": 0,
            "conciseness": 0,
            "safety": 0,
        },
        "decision": "parse_error",
        "reason": f"Judge response could not be parsed: {error}",
        "raw_judge_response": judge_response,
    }


def client_response_text(response: ChatResult | str) -> str:
    if isinstance(response, ChatResult):
        return response.text
    return response


def parse_judge_candidate(value: str) -> JudgeCandidate:
    label, separator, model_and_weight = value.partition("=")
    if not separator or not label or not model_and_weight:
        raise ValueError("--judge-candidate must use label=model[:weight]")
    model, weight_separator, weight_text = model_and_weight.rpartition(":")
    if not weight_separator:
        model = model_and_weight
        weight = 1.0
    else:
        try:
            weight = float(weight_text)
        except ValueError as exc:
            raise ValueError("--judge-candidate weight must be a positive number") from exc
    if not model:
        raise ValueError("--judge-candidate must include a model")
    if weight <= 0:
        raise ValueError("--judge-candidate weight must be positive")
    return label, model, weight


def choose_judge(
    row: dict[str, Any],
    *,
    judge_model: str,
    judge_label: str,
    judge_candidates: list[JudgeCandidate],
    rng: random.Random,
) -> JudgeCandidate:
    if not judge_candidates:
        return judge_label, judge_model
    generator_model = generator_model_label(row)
    eligible = [(label, model) for label, model, _weight in judge_candidates if label != generator_model]
    if not eligible:
        raise ValueError(f"no eligible judge candidate for generator model: {generator_model}")
    weights = [weight for label, _model, weight in judge_candidates if label != generator_model]
    return rng.choices(eligible, weights=weights, k=1)[0]


def judge_rows(
    rows: list[tuple[int, dict[str, Any]]],
    *,
    client: ChatClient,
    judge_model: str,
    judge_label: str,
    max_tokens: int,
    temperature: float,
    reasoning_effort: str,
    exclude_reasoning: bool,
    limit: int | None,
    judge_candidates: list[JudgeCandidate] | None = None,
    random_seed: int = 0,
    existing_records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = list(existing_records or [])
    completed_answer_ids = {
        row["answer_id"] for row in records if isinstance(row.get("answer_id"), str) and row["answer_id"]
    }
    selected_rows = rows[:limit] if limit is not None else rows
    rng = random.Random(random_seed)
    candidates = judge_candidates or []
    for _line_number, row in selected_rows:
        if answer_id(row) in completed_answer_ids:
            continue
        selected_judge_label, selected_judge_model = choose_judge(
            row,
            judge_model=judge_model,
            judge_label=judge_label,
            judge_candidates=candidates,
            rng=rng,
        )
        response = client_response_text(
            client(
                build_payload(
                    row,
                    judge_model=selected_judge_model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    reasoning_effort=reasoning_effort,
                    exclude_reasoning=exclude_reasoning,
                )
            )
        )
        try:
            record = judgment_record(
                row,
                judge_model=selected_judge_model,
                judge_label=selected_judge_label,
                judge_response=response,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            record = failed_judgment_record(
                row,
                judge_model=selected_judge_model,
                judge_label=selected_judge_label,
                judge_response=response,
                error=exc,
            )
        records.append(record)
    return records


def load_existing_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    seen_answer_ids: set[str] = set()
    for line_number, row in load_jsonl(path):
        answer_id_value = row.get("answer_id")
        if not isinstance(answer_id_value, str) or not answer_id_value:
            raise ValueError(f"{path}:{line_number}: missing answer_id for resume")
        if answer_id_value in seen_answer_ids:
            raise ValueError(f"{path}:{line_number}: duplicate answer_id for resume")
        seen_answer_ids.add(answer_id_value)
        records.append(row)
    return records


def write_jsonl(path: Path, rows: list[dict[str, Any]], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "source_prompt_id",
                "answer_id",
                "generator_model",
                "judge_model",
                "decision",
                "correctness",
                "instruction_following",
                "conciseness",
                "safety",
                "reason",
            ],
        )
        writer.writeheader()
        for row in rows:
            scores = row["scores"]
            writer.writerow(
                {
                    "source_prompt_id": row["source_prompt_id"],
                    "answer_id": row["answer_id"],
                    "generator_model": row["generator_model"],
                    "judge_model": row["judge_model"],
                    "decision": row["decision"],
                    "correctness": scores["correctness"],
                    "instruction_following": scores["instruction_following"],
                    "conciseness": scores["conciseness"],
                    "safety": scores["safety"],
                    "reason": row["reason"],
                }
            )


def summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    total = len(rows)
    summary.append({"scope": "overall", "name": "total", "value": total, "rate": "1.000" if total else "0.000"})

    decision_counts: dict[str, int] = {}
    generator_counts: dict[str, int] = {}
    judge_counts: dict[str, int] = {}
    score_totals = {
        "correctness": 0,
        "instruction_following": 0,
        "conciseness": 0,
        "safety": 0,
    }
    for row in rows:
        decision = row.get("decision")
        if isinstance(decision, str):
            decision_counts[decision] = decision_counts.get(decision, 0) + 1
        generator_model = row.get("generator_model")
        if isinstance(generator_model, str):
            generator_counts[generator_model] = generator_counts.get(generator_model, 0) + 1
        judge_model = row.get("judge_model")
        if isinstance(judge_model, str):
            judge_counts[judge_model] = judge_counts.get(judge_model, 0) + 1
        scores = row.get("scores")
        if isinstance(scores, dict):
            for name in score_totals:
                value = scores.get(name)
                if isinstance(value, int):
                    score_totals[name] += value

    for scope, counts in (
        ("decision", decision_counts),
        ("generator_model", generator_counts),
        ("judge_model", judge_counts),
    ):
        for name, count in sorted(counts.items()):
            rate = count / total if total else 0.0
            summary.append({"scope": scope, "name": name, "value": count, "rate": f"{rate:.3f}"})
    for name, total_score in score_totals.items():
        average = total_score / total if total else 0.0
        summary.append({"scope": "avg_score", "name": name, "value": f"{average:.3f}", "rate": ""})
    return summary


def write_summary_csv(path: Path, rows: list[dict[str, Any]], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["scope", "name", "value", "rate"])
        writer.writeheader()
        writer.writerows(summary_rows(rows))


def environment_value(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"{name} must be set")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--judge-label", default=DEFAULT_JUDGE_LABEL)
    parser.add_argument(
        "--judge-candidate",
        action="append",
        default=[],
        help="Eligible random judge in label=model form. May be repeated.",
    )
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--reasoning-effort",
        choices=("provider_default", "none", "minimal", "low", "medium", "high", "xhigh"),
        default="none",
    )
    parser.add_argument("--exclude-reasoning", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    if args.temperature < 0:
        raise ValueError("--temperature must be non-negative")
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if not args.judge_label:
        raise ValueError("--judge-label must not be empty")
    args.judge_candidates = [parse_judge_candidate(value) for value in args.judge_candidate]


def main() -> int:
    args = parse_args()
    validate_args(args)
    rows = load_jsonl(args.input)
    existing_records = load_existing_records(args.output) if args.resume and not args.overwrite else None
    base_url = environment_value("OPENAI_BASE_URL")
    api_key = environment_value("OPENAI_API_KEY")
    client = chat_completions_client(base_url, api_key, args.timeout_seconds)
    judgments = judge_rows(
        rows,
        client=client,
        judge_model=args.judge_model,
        judge_label=args.judge_label,
        judge_candidates=args.judge_candidates,
        random_seed=args.random_seed,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
        exclude_reasoning=args.exclude_reasoning,
        limit=args.limit,
        existing_records=existing_records,
    )
    write_jsonl(args.output, judgments, overwrite=args.overwrite)
    if args.csv_output is not None:
        write_csv(args.csv_output, judgments, overwrite=args.overwrite)
    if args.summary_output is not None:
        write_summary_csv(args.summary_output, judgments, overwrite=args.overwrite)
    print(f"judged {len(judgments)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
