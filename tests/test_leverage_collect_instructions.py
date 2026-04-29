import json
from pathlib import Path

import pytest

from llm.leverage.collect_openai import ChatResult
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
        client=lambda _payload: ChatResult(
            "Use hosted inference before spending GPU time.",
            "stop",
            {"completion_tokens": 8, "prompt_tokens": 12, "total_tokens": 20},
        ),
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
            "generation": {
                "api_model": "provider/model",
                "max_tokens": 256,
                "temperature": 0.1,
                "finish_reason": "stop",
                "usage": {"completion_tokens": 8, "prompt_tokens": 12, "total_tokens": 20},
            },
            "review": {"status": "raw"},
        }
    ]


def test_real_training_seed_file_loads() -> None:
    seeds = collect_instructions.load_seeds(Path("tracks/leverage/prompts/leverage-training-seed-v0.jsonl"))

    assert len(seeds) == 63
    assert "lt_seed_001" in seeds
    assert "lt_seed_063" in seeds


def test_collect_outputs_resume_skips_existing_seed_rows(tmp_path: Path) -> None:
    output = tmp_path / "instruction_outputs.jsonl"
    existing = {
        "source_prompt_id": "lt_seed_001",
        "category": "resource_judgment",
        "purpose": "Existing row.",
        "model": "provider_model",
        "messages": [
            {"role": "system", "content": "Answer as a project lead."},
            {"role": "user", "content": "First prompt"},
            {"role": "assistant", "content": "Existing answer."},
        ],
        "raw_response": "Existing answer.",
        "output_format": "short_answer",
        "constraints": ["mention cost"],
        "review": {"status": "raw"},
    }
    write_jsonl(output, [existing])
    seeds = {
        "lt_seed_001": collect_instructions.InstructionSeed(
            id="lt_seed_001",
            category="resource_judgment",
            purpose="Existing row.",
            system_prompt="Answer as a project lead.",
            prompt="First prompt",
            output_format="short_answer",
            constraints=["mention cost"],
        ),
        "lt_seed_002": collect_instructions.InstructionSeed(
            id="lt_seed_002",
            category="resource_judgment",
            purpose="New row.",
            system_prompt="Answer as a project lead.",
            prompt="Second prompt",
            output_format="short_answer",
            constraints=["mention cost"],
        ),
    }

    collect_instructions.collect_outputs(
        seeds,
        client=lambda payload: ChatResult(
            f"Generated for {payload['messages'][1]['content']}",
            "stop",
            {"completion_tokens": 6},
        ),
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
        resume=True,
    )

    rows = read_jsonl(output)
    assert [row["source_prompt_id"] for row in rows] == ["lt_seed_001", "lt_seed_002"]
    assert rows[0]["raw_response"] == "Existing answer."
    assert rows[1]["raw_response"] == "Generated for Second prompt"
    assert rows[1]["generation"]["finish_reason"] == "stop"
