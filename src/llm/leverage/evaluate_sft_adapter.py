import argparse
import json
import re
from pathlib import Path
from typing import Any

from llm.config import load_toml
from llm.leverage.evaluate import (
    Prediction,
    evaluate_predictions,
    load_predictions,
    load_task_suites,
    write_results,
    write_summary,
)
from llm.leverage.train_sft_smoke import REQUIRED_PACKAGES


DEFAULT_CONFIG = Path("tracks/leverage/configs/leverage-sft-smoke.toml")
DEFAULT_SYSTEM_PROMPT = "Return only the requested answer. Do not include hidden reasoning."


def section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"config section [{name}] must be a table")
    return value


def path_value(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty path string")
    return Path(value)


def string_value(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def int_value(value: Any, label: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def require_inference_packages() -> dict[str, Any]:
    modules: dict[str, Any] = {}
    missing: list[str] = []
    for package in REQUIRED_PACKAGES:
        try:
            modules[package] = __import__(package)
        except ImportError:
            missing.append(package)
    if missing:
        raise RuntimeError(
            "missing optional SFT inference packages: "
            + ", ".join(missing)
            + ". Install them in the RunPod image before evaluating the adapter."
        )
    return modules


def config_defaults(config_path: Path) -> dict[str, Any]:
    config = load_toml(config_path)
    data = section(config, "data")
    model = section(config, "model")
    outputs = section(config, "outputs")
    method = section(config, "method")
    eval_tasks = data.get("eval_tasks")
    if not isinstance(eval_tasks, list) or not eval_tasks:
        raise ValueError("data.eval_tasks must be a non-empty list")
    return {
        "tasks": [path_value(item, "data.eval_tasks[]") for item in eval_tasks],
        "base_model": string_value(model.get("student"), "model.student"),
        "adapter_dir": path_value(outputs.get("adapter_dir"), "outputs.adapter_dir"),
        "output_root": path_value(outputs.get("root"), "outputs.root"),
        "max_new_tokens": int_value(method.get("eval_max_new_tokens", 128), "method.eval_max_new_tokens"),
    }


def prediction_paths(output_root: Path) -> tuple[Path, Path, Path]:
    return (
        output_root / "post-training-predictions.jsonl",
        output_root / "post-training-scores.csv",
        output_root / "post-training-summary.csv",
    )


def task_messages(prompt: str, system_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]


def render_qwen_messages(messages: list[dict[str, str]], tokenizer: Any) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return "\n".join(f"{message['role']}: {message['content']}" for message in messages) + "\nassistant:"


def extract_qwen_final_response(response: str) -> str:
    final = response
    if "</think>" in final:
        final = final.rsplit("</think>", 1)[1]
    final = re.sub(r"<think>.*?</think>", "", final, flags=re.DOTALL)
    final = final.replace("<think>", "")
    final = final.strip()
    final = re.sub(r"^(assistant|user)\s*\n+", "", final)
    return final.strip()


def load_model_pair(
    *,
    modules: dict[str, Any],
    base_model: str,
    adapter_dir: Path,
    device: str,
) -> tuple[Any, Any, Any]:
    transformers = modules["transformers"]
    peft = modules["peft"]

    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = transformers.AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True)
    adapter_base = transformers.AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True)
    adapter = peft.PeftModel.from_pretrained(adapter_base, adapter_dir)
    base.to(device)
    adapter.to(device)
    base.eval()
    adapter.eval()
    return tokenizer, base, adapter


def generate_text(
    *,
    modules: dict[str, Any],
    tokenizer: Any,
    model: Any,
    prompt: str,
    system_prompt: str,
    max_new_tokens: int,
    device: str,
) -> str:
    torch = modules["torch"]
    text = render_qwen_messages(task_messages(prompt, system_prompt), tokenizer)
    encoded = tokenizer(text, return_tensors="pt").to(device)
    input_length = encoded["input_ids"].shape[-1]
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = generated[0][input_length:]
    return extract_qwen_final_response(tokenizer.decode(new_tokens, skip_special_tokens=True))


def write_predictions(path: Path, predictions: list[Prediction]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(
                json.dumps(
                    {
                        "task_id": prediction.task_id,
                        "model": prediction.model,
                        "response": prediction.response,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def parse_qwen_final_predictions(predictions: list[Prediction]) -> list[Prediction]:
    return [
        Prediction(
            task_id=prediction.task_id,
            model=prediction.model,
            response=extract_qwen_final_response(prediction.response),
        )
        for prediction in predictions
    ]


def parse_predictions(
    *,
    tasks_paths: list[Path],
    predictions_path: Path,
    output_root: Path,
) -> list[str]:
    tasks = load_task_suites(tasks_paths)
    predictions = parse_qwen_final_predictions(load_predictions(predictions_path, set(tasks)))
    parsed_predictions_path, scores_path, summary_path = prediction_paths(output_root)
    parsed_predictions_path = output_root / "post-training-predictions.qwen-final.jsonl"
    scores_path = output_root / "post-training-scores.qwen-final.csv"
    summary_path = output_root / "post-training-summary.qwen-final.csv"
    write_predictions(parsed_predictions_path, predictions)
    results = evaluate_predictions(tasks, predictions)
    write_results(scores_path, results)
    write_summary(summary_path, results)
    return [
        f"parsed and scored {len(predictions)} Qwen final responses from {predictions_path}",
        f"wrote parsed predictions: {parsed_predictions_path}",
        f"wrote scores: {scores_path}",
        f"wrote summary: {summary_path}",
    ]


def run_eval(
    *,
    tasks_paths: list[Path],
    base_model: str,
    adapter_dir: Path,
    output_root: Path,
    base_label: str,
    adapter_label: str,
    max_new_tokens: int,
    system_prompt: str,
    device: str,
    dry_run: bool,
) -> list[str]:
    tasks = load_task_suites(tasks_paths)
    predictions_path, scores_path, summary_path = prediction_paths(output_root)
    if dry_run:
        return [
            f"would evaluate {len(tasks)} tasks",
            f"base model: {base_model}",
            f"adapter: {adapter_dir}",
            f"predictions output: {predictions_path}",
            f"scores output: {scores_path}",
            f"summary output: {summary_path}",
        ]
    if not adapter_dir.exists():
        raise FileNotFoundError(f"adapter directory does not exist: {adapter_dir}")

    modules = require_inference_packages()
    torch = modules["torch"]
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; run adapter eval on a CUDA RunPod image or pass --device cpu")

    tokenizer, base, adapter = load_model_pair(
        modules=modules,
        base_model=base_model,
        adapter_dir=adapter_dir,
        device=device,
    )
    predictions: list[Prediction] = []
    for task in tasks.values():
        predictions.append(
            Prediction(
                task_id=task.id,
                model=base_label,
                response=generate_text(
                    modules=modules,
                    tokenizer=tokenizer,
                    model=base,
                    prompt=task.prompt,
                    system_prompt=system_prompt,
                    max_new_tokens=max_new_tokens,
                    device=device,
                ),
            )
        )
        predictions.append(
            Prediction(
                task_id=task.id,
                model=adapter_label,
                response=generate_text(
                    modules=modules,
                    tokenizer=tokenizer,
                    model=adapter,
                    prompt=task.prompt,
                    system_prompt=system_prompt,
                    max_new_tokens=max_new_tokens,
                    device=device,
                ),
            )
        )
    write_predictions(predictions_path, predictions)
    results = evaluate_predictions(tasks, predictions)
    write_results(scores_path, results)
    write_summary(summary_path, results)
    return [
        f"evaluated {len(tasks)} tasks",
        f"wrote predictions: {predictions_path}",
        f"wrote scores: {scores_path}",
        f"wrote summary: {summary_path}",
    ]


def parse_args() -> argparse.Namespace:
    defaults = config_defaults(DEFAULT_CONFIG)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--tasks", type=Path, action="append", default=None)
    parser.add_argument("--base-model", default=defaults["base_model"])
    parser.add_argument("--adapter-dir", type=Path, default=defaults["adapter_dir"])
    parser.add_argument("--output-root", type=Path, default=defaults["output_root"])
    parser.add_argument("--base-label", default="qwen3.5-0.8b-base")
    parser.add_argument("--adapter-label", default="qwen3.5-0.8b-lora-smoke")
    parser.add_argument("--max-new-tokens", type=int, default=defaults["max_new_tokens"])
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--parse-predictions", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    defaults = config_defaults(args.config)
    tasks = args.tasks if args.tasks is not None else defaults["tasks"]
    if args.parse_predictions is not None:
        for line in parse_predictions(
            tasks_paths=tasks,
            predictions_path=args.parse_predictions,
            output_root=args.output_root,
        ):
            print(line)
        return 0
    for line in run_eval(
        tasks_paths=tasks,
        base_model=args.base_model,
        adapter_dir=args.adapter_dir,
        output_root=args.output_root,
        base_label=args.base_label,
        adapter_label=args.adapter_label,
        max_new_tokens=args.max_new_tokens,
        system_prompt=args.system_prompt,
        device=args.device,
        dry_run=args.dry_run,
    ):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
