import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from llm.leverage.collect_openai import chat_completions_client


DEFAULT_MODEL = "qwen/qwen3.5-flash-02-23"
DEFAULT_MODEL_LABEL = "qwen3_5_flash_openrouter"
REQUIRED_FIELDS = {
    "id",
    "category",
    "purpose",
    "system_prompt",
    "prompt",
    "output_format",
    "constraints",
}

ChatClient = Callable[[dict[str, Any]], str]


@dataclass(frozen=True)
class InstructionSeed:
    id: str
    category: str
    purpose: str
    system_prompt: str
    prompt: str
    output_format: str
    constraints: list[str]


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


def seed_from_payload(path: Path, line_number: int, payload: dict[str, Any]) -> InstructionSeed:
    missing = REQUIRED_FIELDS - payload.keys()
    if missing:
        raise ValueError(f"{path}:{line_number}: missing required fields: {sorted(missing)}")
    constraints = payload["constraints"]
    if not isinstance(constraints, list) or not constraints:
        raise ValueError(f"{path}:{line_number}: constraints must be a non-empty list")
    if not all(isinstance(item, str) and item for item in constraints):
        raise ValueError(f"{path}:{line_number}: constraints must contain non-empty strings")
    values: dict[str, str] = {}
    for name in REQUIRED_FIELDS - {"constraints"}:
        value = payload[name]
        if not isinstance(value, str) or not value:
            raise ValueError(f"{path}:{line_number}: {name} must be a non-empty string")
        values[name] = value
    return InstructionSeed(
        id=values["id"],
        category=values["category"],
        purpose=values["purpose"],
        system_prompt=values["system_prompt"],
        prompt=values["prompt"],
        output_format=values["output_format"],
        constraints=constraints,
    )


def load_seeds(path: Path) -> dict[str, InstructionSeed]:
    seeds: dict[str, InstructionSeed] = {}
    for line_number, payload in load_jsonl(path):
        seed = seed_from_payload(path, line_number, payload)
        if seed.id in seeds:
            raise ValueError(f"{path}:{line_number}: duplicate seed id: {seed.id}")
        seeds[seed.id] = seed
    if not seeds:
        raise ValueError(f"{path}: file must contain at least one seed")
    return seeds


def build_payload(
    seed: InstructionSeed,
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    thinking_mode: str,
    thinking_param: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": seed.system_prompt},
            {"role": "user", "content": seed.prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if thinking_mode == "disabled":
        if thinking_param == "chat_template_kwargs":
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        elif thinking_param == "enable_thinking":
            payload["enable_thinking"] = False
    return payload


def collect_outputs(
    seeds: dict[str, InstructionSeed],
    *,
    client: ChatClient,
    output_path: Path,
    api_model: str,
    model_label: str,
    max_tokens: int,
    temperature: float,
    thinking_mode: str,
    thinking_param: str,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for seed in seeds.values():
            payload = build_payload(
                seed,
                model=api_model,
                max_tokens=max_tokens,
                temperature=temperature,
                thinking_mode=thinking_mode,
                thinking_param=thinking_param,
            )
            response = client(payload)
            record = {
                "source_prompt_id": seed.id,
                "category": seed.category,
                "purpose": seed.purpose,
                "model": model_label,
                "messages": [
                    {"role": "system", "content": seed.system_prompt},
                    {"role": "user", "content": seed.prompt},
                    {"role": "assistant", "content": response},
                ],
                "raw_response": response,
                "output_format": seed.output_format,
                "constraints": seed.constraints,
                "review": {"status": "raw"},
            }
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_file.flush()


def environment_value(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"{name} must be set")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-label", default=DEFAULT_MODEL_LABEL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--thinking-mode",
        choices=("disabled", "none"),
        default="none",
        help="Use 'none' to avoid sending provider-specific thinking controls.",
    )
    parser.add_argument(
        "--thinking-param",
        choices=("chat_template_kwargs", "enable_thinking"),
        default="chat_template_kwargs",
    )
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    if args.temperature < 0:
        raise ValueError("--temperature must be non-negative")
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive")
    if not args.model_label:
        raise ValueError("--model-label must not be empty")


def main() -> None:
    args = parse_args()
    validate_args(args)
    seeds = load_seeds(args.seeds)
    base_url = environment_value("OPENAI_BASE_URL")
    api_key = environment_value("OPENAI_API_KEY")
    client = chat_completions_client(base_url, api_key, args.timeout_seconds)
    collect_outputs(
        seeds,
        client=client,
        output_path=args.output,
        api_model=args.model,
        model_label=args.model_label,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        thinking_mode=args.thinking_mode,
        thinking_param=args.thinking_param,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
