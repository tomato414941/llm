from pathlib import Path
from typing import Any
import tomllib


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as config_file:
        data = tomllib.load(config_file)
    if not isinstance(data, dict):
        raise ValueError("config must be a TOML table")
    return data


def nested_get(data: dict[str, Any], section: str, key: str, default: Any = None) -> Any:
    section_data = data.get(section, {})
    if not isinstance(section_data, dict):
        raise ValueError(f"config section [{section}] must be a table")
    return section_data.get(key, default)


def config_defaults(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": nested_get(config, "run", "run_id"),
        "input": nested_get(config, "data", "input"),
        "tokens": nested_get(config, "data", "tokens"),
        "manifest": nested_get(config, "data", "manifest"),
        "checkpoint": nested_get(config, "outputs", "checkpoint"),
        "metrics_output": nested_get(config, "outputs", "metrics"),
        "tokenizer": nested_get(config, "data", "tokenizer"),
        "tokenizer_path": nested_get(config, "data", "tokenizer_path"),
        "max_iters": nested_get(config, "train", "max_iters"),
        "eval_interval": nested_get(config, "train", "eval_interval"),
        "eval_iters": nested_get(config, "train", "eval_iters"),
        "batch_size": nested_get(config, "train", "batch_size"),
        "learning_rate": nested_get(config, "train", "learning_rate"),
        "weight_decay": nested_get(config, "train", "weight_decay"),
        "min_learning_rate": nested_get(config, "train", "min_learning_rate"),
        "warmup_iters": nested_get(config, "train", "warmup_iters"),
        "lr_decay_iters": nested_get(config, "train", "lr_decay_iters"),
        "seed": nested_get(config, "train", "seed"),
        "block_size": nested_get(config, "model", "block_size"),
        "embedding_dim": nested_get(config, "model", "embedding_dim"),
        "num_heads": nested_get(config, "model", "num_heads"),
        "num_layers": nested_get(config, "model", "num_layers"),
        "dropout": nested_get(config, "model", "dropout"),
        "generate_tokens": nested_get(config, "generation", "generate_tokens"),
        "temperature": nested_get(config, "generation", "temperature"),
        "top_k": nested_get(config, "generation", "top_k"),
        "sampling": nested_get(config, "generation", "sampling"),
    }


def observe_config_defaults(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "checkpoint": nested_get(config, "outputs", "checkpoint"),
        "tokens": nested_get(config, "data", "tokens"),
        "prompt_file": nested_get(config, "evaluation", "prompt_file"),
        "output": nested_get(config, "outputs", "observation"),
        "summary_output": nested_get(config, "outputs", "summary"),
        "eval_iters": nested_get(config, "evaluation", "eval_iters"),
        "batch_size": nested_get(config, "evaluation", "batch_size"),
        "max_new_tokens": nested_get(config, "generation", "generate_tokens"),
        "temperature": nested_get(config, "generation", "temperature"),
        "top_k": nested_get(config, "generation", "top_k"),
        "sampling": nested_get(config, "generation", "sampling"),
        "seed": nested_get(config, "generation", "seed"),
        "samples": nested_get(config, "generation", "samples"),
    }


def compact_defaults(defaults: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in defaults.items() if value is not None}
