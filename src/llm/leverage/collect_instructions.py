import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from llm.leverage.capabilities import ALLOWED_CAPABILITIES
from llm.leverage.collect_openai import ChatResult, chat_completions_client


DEFAULT_MODEL = "qwen/qwen3.6-plus"
DEFAULT_MODEL_LABEL = "qwen3-6-plus-openrouter"
REQUIRED_FIELDS = {
    "id",
    "capability",
    "purpose",
    "system_prompt",
    "prompt",
    "output_format",
    "constraints",
}

ChatClient = Callable[[dict[str, Any]], ChatResult]
ModelCandidate = tuple[str, str, float]


@dataclass(frozen=True)
class InstructionSeed:
    id: str
    capability: str
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
    if values["capability"] not in ALLOWED_CAPABILITIES:
        raise ValueError(
            f"{path}:{line_number}: capability must be one of {sorted(ALLOWED_CAPABILITIES)}"
        )
    return InstructionSeed(
        id=values["id"],
        capability=values["capability"],
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


def select_seeds(seeds: dict[str, InstructionSeed], seed_ids: list[str]) -> dict[str, InstructionSeed]:
    if not seed_ids:
        return seeds
    selected: dict[str, InstructionSeed] = {}
    for seed_id in seed_ids:
        try:
            selected[seed_id] = seeds[seed_id]
        except KeyError as exc:
            raise ValueError(f"unknown seed id: {seed_id}") from exc
    return selected


def build_payload(
    seed: InstructionSeed,
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    thinking_mode: str,
    thinking_param: str,
    reasoning_effort: str,
    exclude_reasoning: bool,
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
    if reasoning_effort != "provider_default" or exclude_reasoning:
        reasoning: dict[str, Any] = {"exclude": exclude_reasoning}
        if reasoning_effort != "provider_default":
            reasoning["effort"] = reasoning_effort
        payload["reasoning"] = reasoning
    return payload


def parse_model_candidate(value: str, *, option_name: str) -> ModelCandidate:
    label, separator, model_and_weight = value.partition("=")
    if not separator or not label or not model_and_weight:
        raise ValueError(f"{option_name} must use label=model[:weight]")
    model, weight_separator, weight_text = model_and_weight.rpartition(":")
    if not weight_separator:
        model = model_and_weight
        weight = 1.0
    else:
        try:
            weight = float(weight_text)
        except ValueError as exc:
            raise ValueError(f"{option_name} weight must be a positive number") from exc
    if not model:
        raise ValueError(f"{option_name} must include a model")
    if weight <= 0:
        raise ValueError(f"{option_name} weight must be positive")
    return label, model, weight


def choose_model(
    *,
    api_model: str,
    model_label: str,
    generator_candidates: list[ModelCandidate],
    rng: random.Random,
) -> tuple[str, str]:
    if not generator_candidates:
        return model_label, api_model
    labels = [(label, model) for label, model, _weight in generator_candidates]
    weights = [weight for _label, _model, weight in generator_candidates]
    return rng.choices(labels, weights=weights, k=1)[0]


def collect_outputs(
    seeds: dict[str, InstructionSeed],
    *,
    client: ChatClient,
    output_path: Path,
    api_model: str,
    model_label: str,
    generator_candidates: list[ModelCandidate] | None = None,
    random_seed: int = 0,
    max_tokens: int,
    temperature: float,
    thinking_mode: str,
    thinking_param: str,
    reasoning_effort: str,
    exclude_reasoning: bool,
    overwrite: bool,
    resume: bool = False,
) -> None:
    if output_path.exists() and not overwrite and not resume:
        raise FileExistsError(f"output already exists: {output_path}")
    completed_seed_ids: set[str] = set()
    if output_path.exists() and resume and not overwrite:
        for line_number, row in load_jsonl(output_path):
            source_prompt_id = row.get("source_prompt_id")
            if not isinstance(source_prompt_id, str) or not source_prompt_id:
                raise ValueError(f"{output_path}:{line_number}: missing source_prompt_id for resume")
            if source_prompt_id in completed_seed_ids:
                raise ValueError(f"{output_path}:{line_number}: duplicate source_prompt_id for resume")
            completed_seed_ids.add(source_prompt_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if resume and output_path.exists() and not overwrite else "w"
    rng = random.Random(random_seed)
    candidates = generator_candidates or []
    with output_path.open(mode, encoding="utf-8") as output_file:
        for seed in seeds.values():
            if seed.id in completed_seed_ids:
                continue
            selected_label, selected_model = choose_model(
                api_model=api_model,
                model_label=model_label,
                generator_candidates=candidates,
                rng=rng,
            )
            payload = build_payload(
                seed,
                model=selected_model,
                max_tokens=max_tokens,
                temperature=temperature,
                thinking_mode=thinking_mode,
                thinking_param=thinking_param,
                reasoning_effort=reasoning_effort,
                exclude_reasoning=exclude_reasoning,
            )
            response = client(payload)
            record = {
                "source_prompt_id": seed.id,
                "capability": seed.capability,
                "purpose": seed.purpose,
                "model": selected_label,
                "messages": [
                    {"role": "system", "content": seed.system_prompt},
                    {"role": "user", "content": seed.prompt},
                    {"role": "assistant", "content": response.text},
                ],
                "raw_response": response.text,
                "output_format": seed.output_format,
                "constraints": seed.constraints,
                "generation": {
                    "api_model": selected_model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "finish_reason": response.finish_reason,
                    "usage": response.usage,
                },
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
    parser.add_argument("--seed-id", action="append", default=[])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-label", default=DEFAULT_MODEL_LABEL)
    parser.add_argument(
        "--generator-candidate",
        action="append",
        default=[],
        help="Eligible random generator in label=model[:weight] form. May be repeated.",
    )
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=16384)
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
    parser.add_argument(
        "--reasoning-effort",
        choices=("provider_default", "none", "minimal", "low", "medium", "high", "xhigh"),
        default="none",
    )
    parser.add_argument("--exclude-reasoning", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
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
    if not args.model_label:
        raise ValueError("--model-label must not be empty")
    args.generator_candidates = [
        parse_model_candidate(value, option_name="--generator-candidate") for value in args.generator_candidate
    ]


def main() -> None:
    args = parse_args()
    validate_args(args)
    seeds = select_seeds(load_seeds(args.seeds), args.seed_id)
    base_url = environment_value("OPENAI_BASE_URL")
    api_key = environment_value("OPENAI_API_KEY")
    client = chat_completions_client(base_url, api_key, args.timeout_seconds)
    collect_outputs(
        seeds,
        client=client,
        output_path=args.output,
        api_model=args.model,
        model_label=args.model_label,
        generator_candidates=args.generator_candidates,
        random_seed=args.random_seed,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        thinking_mode=args.thinking_mode,
        thinking_param=args.thinking_param,
        reasoning_effort=args.reasoning_effort,
        exclude_reasoning=args.exclude_reasoning,
        overwrite=args.overwrite,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
