import json
from pathlib import Path

import pytest

from llm.leverage import collect_instructions


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def seed_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": "lt_seed_test",
        "category": "resource_judgment",
        "purpose": "Generate a resource judgment example.",
        "system_prompt": "Answer as a project lead.",
        "prompt": "When should we use hosted inference?",
        "output_format": "short_answer",
        "constraints": ["mention cost", "mention training"],
    }
    payload.update(overrides)
    return payload


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_load_seeds_reads_prompt_seed_schema(tmp_path: Path) -> None:
    path = tmp_path / "seeds.jsonl"
    write_jsonl(path, [seed_payload()])

    seeds = collect_instructions.load_seeds(path)

    seed = seeds["lt_seed_test"]
    assert seed.category == "resource_judgment"
    assert seed.system_prompt == "Answer as a project lead."
    assert seed.constraints == ["mention cost", "mention training"]


def test_load_seeds_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "seeds.jsonl"
    write_jsonl(path, [seed_payload(), seed_payload()])

    with pytest.raises(ValueError, match="duplicate seed id"):
        collect_instructions.load_seeds(path)


def test_build_payload_uses_seed_system_prompt_and_user_prompt() -> None:
    seed = collect_instructions.InstructionSeed(
        id="lt_seed_test",
        category="resource_judgment",
        purpose="Generate a resource judgment example.",
        system_prompt="Answer as a project lead.",
        prompt="When should we use hosted inference?",
        output_format="short_answer",
        constraints=["mention cost"],
    )

    payload = collect_instructions.build_payload(
        seed,
        model="provider/model",
        max_tokens=256,
        temperature=0.1,
        thinking_mode="none",
        thinking_param="chat_template_kwargs",
        reasoning_effort="none",
        exclude_reasoning=True,
    )

    assert payload == {
        "model": "provider/model",
        "messages": [
            {"role": "system", "content": "Answer as a project lead."},
            {"role": "user", "content": "When should we use hosted inference?"},
        ],
        "max_tokens": 256,
        "temperature": 0.1,
        "reasoning": {"exclude": True, "effort": "none"},
    }


def test_collect_outputs_writes_raw_review_schema(tmp_path: Path) -> None:
    output = tmp_path / "instruction_outputs.jsonl"
    seeds = {
        "lt_seed_test": collect_instructions.InstructionSeed(
            id="lt_seed_test",
            category="resource_judgment",
            purpose="Generate a resource judgment example.",
            system_prompt="Answer as a project lead.",
            prompt="When should we use hosted inference?",
            output_format="short_answer",
            constraints=["mention cost"],
        )
    }

    collect_instructions.collect_outputs(
        seeds,
        client=lambda _payload: "Use hosted inference before spending GPU time.",
        output_path=output,
        api_model="provider/model",
        model_label="provider_model",
        max_tokens=256,
        temperature=0.1,
        thinking_mode="none",
        thinking_param="chat_template_kwargs",
        reasoning_effort="none",
        exclude_reasoning=True,
        overwrite=False,
    )

    assert read_jsonl(output) == [
        {
            "source_prompt_id": "lt_seed_test",
            "category": "resource_judgment",
            "purpose": "Generate a resource judgment example.",
            "model": "provider_model",
            "messages": [
                {"role": "system", "content": "Answer as a project lead."},
                {"role": "user", "content": "When should we use hosted inference?"},
                {
                    "role": "assistant",
                    "content": "Use hosted inference before spending GPU time.",
                },
            ],
            "raw_response": "Use hosted inference before spending GPU time.",
            "output_format": "short_answer",
            "constraints": ["mention cost"],
            "review": {"status": "raw"},
        }
    ]


def test_real_training_seed_file_loads() -> None:
    seeds = collect_instructions.load_seeds(Path("prompts/leverage-training-seed-v0.jsonl"))

    assert len(seeds) == 12
    assert "lt_seed_001" in seeds
