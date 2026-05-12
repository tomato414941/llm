from argparse import Namespace
import pytest

from llm import observe
from llm.config import compact_defaults, load_toml, observe_config_defaults
from llm.observe import build_parser, load_prompt_file, validate_args


def valid_args(**overrides) -> Namespace:
    values = {
        "batch_size": 1,
        "eval_iters": 1,
        "max_new_tokens": 1,
        "temperature": 1.0,
        "top_k": None,
        "sampling": True,
        "samples": 1,
        "prompt": "",
        "prompt_file": None,
    }
    values.update(overrides)
    return Namespace(**values)


def test_validate_args_rejects_non_positive_values() -> None:
    with pytest.raises(ValueError, match="--batch-size"):
        validate_args(valid_args(batch_size=0))

    with pytest.raises(ValueError, match="--eval-iters"):
        validate_args(valid_args(eval_iters=0))

    with pytest.raises(ValueError, match="--max-new-tokens"):
        validate_args(valid_args(max_new_tokens=0))

    with pytest.raises(ValueError, match="--temperature"):
        validate_args(valid_args(temperature=0))

    with pytest.raises(ValueError, match="--top-k"):
        validate_args(valid_args(top_k=0))

    with pytest.raises(ValueError, match="--samples"):
        validate_args(valid_args(samples=0))


def test_validate_args_rejects_prompt_and_prompt_file() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        validate_args(valid_args(prompt="KING:", prompt_file="prompts.jsonl"))


def test_load_prompt_file_reads_jsonl(tmp_path) -> None:
    path = tmp_path / "prompts.jsonl"
    path.write_text(
        '{"id":"king","prompt":"KING:"}\n{"id":"scene","prompt":"SCENE"}\n',
        encoding="utf-8",
    )

    assert load_prompt_file(path) == [
        {"id": "king", "prompt": "KING:"},
        {"id": "scene", "prompt": "SCENE"},
    ]


def test_observe_config_defaults_reads_config(tmp_path) -> None:
    path = tmp_path / "run.toml"
    path.write_text(
        """
[data]
tokens = "tracks/from-scratch/data/processed/tokens.pt"

[outputs]
checkpoint = "tracks/from-scratch/runs/model/checkpoint.pt"
observation = "tracks/from-scratch/runs/model/observation.md"

[evaluation]
prompt_file = "tracks/from-scratch/data/eval-prompts/example.jsonl"
eval_iters = 2
batch_size = 4

[generation]
generate_tokens = 5
temperature = 0.8
top_k = 10
seed = 7
samples = 3
sampling = false
""",
        encoding="utf-8",
    )

    defaults = compact_defaults(observe_config_defaults(load_toml(path)))

    assert defaults["tokens"] == "tracks/from-scratch/data/processed/tokens.pt"
    assert defaults["checkpoint"] == "tracks/from-scratch/runs/model/checkpoint.pt"
    assert defaults["output"] == "tracks/from-scratch/runs/model/observation.md"
    assert "summary_output" not in defaults
    assert defaults["prompt_file"] == "tracks/from-scratch/data/eval-prompts/example.jsonl"
    assert defaults["eval_iters"] == 2
    assert defaults["max_new_tokens"] == 5
    assert defaults["samples"] == 3
    assert defaults["sampling"] is False


def test_observe_cli_overrides_config_defaults() -> None:
    parser = build_parser({"eval_iters": 2, "checkpoint": "from-config.pt"})

    args = parser.parse_args(["--eval-iters", "5", "--checkpoint", "from-cli.pt"])

    assert args.eval_iters == 5
    assert args.checkpoint.name == "from-cli.pt"


def test_build_observations_loads_context_once_for_prompt_file(monkeypatch, tmp_path) -> None:
    prompt_file = tmp_path / "prompts.jsonl"
    prompt_file.write_text(
        '{"id":"king","prompt":"KING:"}\n{"id":"scene","prompt":"SCENE"}\n',
        encoding="utf-8",
    )
    calls = {"context": 0, "observation": 0}
    args = Namespace(prompt_file=prompt_file)

    def fake_load_context(_args):
        calls["context"] += 1
        return object()

    def fake_build_observation(_context, _args, prompt=None, prompt_id=""):
        calls["observation"] += 1
        return {"prompt": prompt, "prompt_id": prompt_id}

    monkeypatch.setattr(observe, "load_observation_context", fake_load_context)
    monkeypatch.setattr(observe, "build_observation", fake_build_observation)

    observations = observe.build_observations(args)

    assert calls == {"context": 1, "observation": 2}
    assert observations == [
        {"prompt": "KING:", "prompt_id": "king"},
        {"prompt": "SCENE", "prompt_id": "scene"},
    ]
