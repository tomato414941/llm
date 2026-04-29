import argparse
import csv
from pathlib import Path
from typing import Any

from llm.config import load_toml
from llm.leverage.validate_reviewed_instructions import load_jsonl


REQUIRED_PACKAGES = ("torch", "transformers", "peft", "trl")
DEFAULT_CONFIG = Path("tracks/leverage/configs/leverage-sft-smoke.toml")


def require_training_packages() -> dict[str, Any]:
    modules: dict[str, Any] = {}
    missing: list[str] = []
    for package in REQUIRED_PACKAGES:
        try:
            modules[package] = __import__(package)
        except ImportError:
            missing.append(package)
    if missing:
        raise RuntimeError(
            "missing optional SFT training packages: "
            + ", ".join(missing)
            + ". Install them in the RunPod training image before running this smoke."
        )
    return modules


def section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"config section [{name}] must be a table")
    return value


def path_value(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty path string")
    return Path(value)


def int_value(value: Any, label: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def string_value(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def load_training_rows(path: Path, max_examples: int) -> list[dict[str, Any]]:
    rows = [row for _line_number, row in load_jsonl(path)]
    if len(rows) > max_examples:
        raise ValueError(f"{path} has {len(rows)} rows, exceeding max_examples={max_examples}")
    for index, row in enumerate(rows):
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"{path}: row {index + 1} must contain messages")
    return rows


def write_metrics(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def render_messages(row: dict[str, Any], tokenizer: Any) -> str:
    messages = row["messages"]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return "\n".join(f"{message['role']}: {message['content']}" for message in messages)


def train_lora_smoke(
    *,
    modules: dict[str, Any],
    rows: list[dict[str, Any]],
    student_model: str,
    adapter_dir: Path,
    max_epochs: int,
) -> list[float]:
    torch = modules["torch"]
    transformers = modules["transformers"]
    peft = modules["peft"]

    tokenizer = transformers.AutoTokenizer.from_pretrained(student_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(student_model, trust_remote_code=True)
    lora_config = peft.LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = peft.get_peft_model(model, lora_config)
    model.to("cuda")
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    losses: list[float] = []
    for _epoch in range(max_epochs):
        for row in rows:
            text = render_messages(row, tokenizer)
            encoded = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                padding=False,
            )
            encoded = {key: value.to("cuda") for key, value in encoded.items()}
            labels = encoded["input_ids"].clone()
            outputs = model(**encoded, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))
            if not torch.isfinite(loss):
                raise RuntimeError("training loss became NaN or inf")
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    return losses


def run_smoke(config_path: Path, *, dry_run: bool) -> list[str]:
    config = load_toml(config_path)
    data = section(config, "data")
    method = section(config, "method")
    model = section(config, "model")
    outputs = section(config, "outputs")

    train_export = path_value(data.get("train_export"), "data.train_export")
    adapter_dir = path_value(outputs.get("adapter_dir"), "outputs.adapter_dir")
    logs_dir = path_value(outputs.get("logs"), "outputs.logs")
    metrics_path = path_value(outputs.get("metrics"), "outputs.metrics")
    notes_path = path_value(outputs.get("notes"), "outputs.notes")
    max_examples = int_value(method.get("max_train_examples"), "method.max_train_examples")
    max_epochs = int_value(method.get("max_epochs"), "method.max_epochs")
    student_model = string_value(model.get("student"), "model.student")

    rows = load_training_rows(train_export, max_examples)
    if dry_run:
        return [
            f"would train {len(rows)} rows from {train_export}",
            f"student model: {student_model}",
            f"adapter output: {adapter_dir}",
            f"metrics output: {metrics_path}",
        ]

    modules = require_training_packages()
    torch = modules["torch"]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; run this smoke on a CUDA RunPod image")

    logs_dir.mkdir(parents=True, exist_ok=True)
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    losses = train_lora_smoke(
        modules=modules,
        rows=rows,
        student_model=student_model,
        adapter_dir=adapter_dir,
        max_epochs=max_epochs,
    )
    write_metrics(
        metrics_path,
        [
            {"metric": "rows", "value": str(len(rows))},
            {"metric": "student_model", "value": student_model},
            {"metric": "cuda_device", "value": torch.cuda.get_device_name(0)},
            {"metric": "steps", "value": str(len(losses))},
            {"metric": "final_loss", "value": f"{losses[-1]:.6f}" if losses else ""},
            {"metric": "status", "value": "completed"},
        ],
    )
    notes_path.write_text(
        "\n".join(
            [
                "# Leverage SFT Smoke",
                "",
                f"Rows: {len(rows)}",
                f"Epochs: {max_epochs}",
                f"Student model: `{student_model}`",
                f"CUDA device: `{torch.cuda.get_device_name(0)}`",
                f"Final loss: `{losses[-1]:.6f}`" if losses else "Final loss: unavailable",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return [
        f"loaded {len(rows)} training rows",
        f"cuda device: {torch.cuda.get_device_name(0)}",
        f"saved adapter: {adapter_dir}",
        f"wrote metrics: {metrics_path}",
        f"wrote notes: {notes_path}",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for line in run_smoke(args.config, dry_run=args.dry_run):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
