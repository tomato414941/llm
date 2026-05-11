import argparse
import csv
import json
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm.config import load_toml


REQUIRED_PACKAGES = ("torch", "transformers", "peft", "trl")
DEFAULT_CONFIG = Path("tracks/leverage/configs/leverage-sft-smoke.toml")


@dataclass(frozen=True)
class EarlyStoppingConfig:
    enabled: bool = False
    validation_examples: int = 0
    eval_every_steps: int = 0
    patience: int = 0
    min_delta: float = 0.0


@dataclass
class EarlyStoppingState:
    best_loss: float | None = None
    best_step: int = 0
    checks_without_improvement: int = 0
    stopped: bool = False
    stop_step: int = 0
    stop_reason: str = ""

    def update(self, *, validation_loss: float, step: int, config: EarlyStoppingConfig) -> None:
        if self.best_loss is None or validation_loss < self.best_loss - config.min_delta:
            self.best_loss = validation_loss
            self.best_step = step
            self.checks_without_improvement = 0
            return
        self.checks_without_improvement += 1
        if self.checks_without_improvement >= config.patience:
            self.stopped = True
            self.stop_step = step
            self.stop_reason = "validation_loss_patience_exhausted"


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


def positive_int_value(value: Any, label: str) -> int:
    value = int_value(value, label)
    if value <= 0:
        raise ValueError(f"{label} must be positive")
    return value


def string_value(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def optional_string_value(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return string_value(value, label)


def bool_value(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def non_negative_int_value(value: Any, label: str) -> int:
    value = int_value(value, label)
    if value < 0:
        raise ValueError(f"{label} must be non-negative")
    return value


def non_negative_float_value(value: Any, label: str) -> float:
    if not isinstance(value, int | float):
        raise ValueError(f"{label} must be a number")
    value = float(value)
    if value < 0:
        raise ValueError(f"{label} must be non-negative")
    return value


def early_stopping_config(config: dict[str, Any]) -> EarlyStoppingConfig:
    value = config.get("early_stopping")
    if value is None:
        return EarlyStoppingConfig()
    if not isinstance(value, dict):
        raise ValueError("config section [early_stopping] must be a table")
    enabled = bool_value(value.get("enabled", False), "early_stopping.enabled")
    validation_examples = non_negative_int_value(
        value.get("validation_examples", 0),
        "early_stopping.validation_examples",
    )
    eval_every_steps = non_negative_int_value(
        value.get("eval_every_steps", 0),
        "early_stopping.eval_every_steps",
    )
    patience = non_negative_int_value(value.get("patience", 0), "early_stopping.patience")
    min_delta = non_negative_float_value(value.get("min_delta", 0.0), "early_stopping.min_delta")
    if enabled:
        if validation_examples <= 0:
            raise ValueError("early_stopping.validation_examples must be positive when enabled")
        if eval_every_steps <= 0:
            raise ValueError("early_stopping.eval_every_steps must be positive when enabled")
        if patience <= 0:
            raise ValueError("early_stopping.patience must be positive when enabled")
    return EarlyStoppingConfig(
        enabled=enabled,
        validation_examples=validation_examples,
        eval_every_steps=eval_every_steps,
        patience=patience,
        min_delta=min_delta,
    )


def load_training_rows(path: Path, max_examples: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if len(rows) >= max_examples:
                break
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank lines are not allowed")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be a JSON object")
            rows.append(row)
    for index, row in enumerate(rows):
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"{path}: row {index + 1} must contain messages")
    return rows


def split_training_rows(
    rows: list[dict[str, Any]],
    config: EarlyStoppingConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not config.enabled:
        return rows, []
    if len(rows) <= config.validation_examples:
        raise ValueError("training rows must exceed early_stopping.validation_examples")
    return rows[: -config.validation_examples], rows[-config.validation_examples :]


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


def nvidia_smi_sample() -> dict[str, str]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return {
            "gpu_utilization_percent": "",
            "gpu_memory_used_mb": "",
            "gpu_memory_total_mb": "",
        }
    first_line = completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else ""
    values = [value.strip() for value in first_line.split(",")]
    if len(values) != 3:
        return {
            "gpu_utilization_percent": "",
            "gpu_memory_used_mb": "",
            "gpu_memory_total_mb": "",
        }
    return {
        "gpu_utilization_percent": values[0],
        "gpu_memory_used_mb": values[1],
        "gpu_memory_total_mb": values[2],
    }


def numeric_sample_values(samples: list[dict[str, str]], key: str) -> list[float]:
    values: list[float] = []
    for sample in samples:
        try:
            values.append(float(sample[key]))
        except (KeyError, TypeError, ValueError):
            pass
    return values


def gpu_sample_summary(samples: list[dict[str, str]]) -> dict[str, float]:
    utilization_values = numeric_sample_values(samples, "gpu_utilization_percent")
    memory_used_values = numeric_sample_values(samples, "gpu_memory_used_mb")
    memory_total_values = numeric_sample_values(samples, "gpu_memory_total_mb")
    return {
        "gpu_sample_count": float(len(utilization_values)),
        "gpu_utilization_avg_percent": sum(utilization_values) / len(utilization_values)
        if utilization_values
        else 0.0,
        "gpu_utilization_max_percent": max(utilization_values) if utilization_values else 0.0,
        "gpu_memory_used_max_mb": max(memory_used_values) if memory_used_values else 0.0,
        "gpu_memory_total_mb": max(memory_total_values) if memory_total_values else 0.0,
    }


def start_gpu_sampler(path: Path, interval_seconds: float = 1.0) -> tuple[threading.Event, threading.Thread]:
    path.parent.mkdir(parents=True, exist_ok=True)
    stop_event = threading.Event()

    def sample_loop() -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "elapsed_seconds",
                    "gpu_utilization_percent",
                    "gpu_memory_used_mb",
                    "gpu_memory_total_mb",
                ],
            )
            writer.writeheader()
            started = time.monotonic()
            while not stop_event.is_set():
                writer.writerow(
                    {
                        "elapsed_seconds": f"{time.monotonic() - started:.3f}",
                        **nvidia_smi_sample(),
                    }
                )
                handle.flush()
                stop_event.wait(interval_seconds)

    thread = threading.Thread(target=sample_loop, daemon=True)
    thread.start()
    return stop_event, thread


def load_gpu_samples(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def train_lora_smoke(
    *,
    modules: dict[str, Any],
    rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    student_model: str,
    adapter_dir: Path,
    progress_path: Path,
    max_epochs: int,
    batch_size: int,
    max_length: int,
    torch_dtype: str | None,
    gradient_checkpointing: bool,
    gradient_accumulation_steps: int,
    log_every_steps: int,
    early_stopping: EarlyStoppingConfig,
) -> dict[str, float]:
    torch = modules["torch"]
    transformers = modules["transformers"]
    peft = modules["peft"]
    total_started = time.monotonic()

    tokenizer_started = time.monotonic()
    tokenizer = transformers.AutoTokenizer.from_pretrained(student_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer_load_seconds = time.monotonic() - tokenizer_started

    model_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if torch_dtype:
        model_kwargs["torch_dtype"] = getattr(torch, torch_dtype)
    model_started = time.monotonic()
    model = transformers.AutoModelForCausalLM.from_pretrained(student_model, **model_kwargs)
    model_load_seconds = time.monotonic() - model_started

    adapter_started = time.monotonic()
    if gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    lora_config = peft.LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = peft.get_peft_model(model, lora_config)
    adapter_setup_seconds = time.monotonic() - adapter_started

    cuda_started = time.monotonic()
    model.to("cuda")
    model.train()
    cuda_transfer_seconds = time.monotonic() - cuda_started

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    losses: list[float] = []
    validation_losses: list[float] = []
    token_count = 0
    render_started = time.monotonic()
    texts = [render_messages(row, tokenizer) for row in rows]
    validation_texts = [render_messages(row, tokenizer) for row in validation_rows]
    render_seconds = time.monotonic() - render_started

    def collate_texts(batch: list[str]) -> dict[str, Any]:
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        labels = encoded["input_ids"].clone()
        labels[encoded["attention_mask"] == 0] = -100
        encoded["labels"] = labels
        return encoded

    dataloader = torch.utils.data.DataLoader(
        texts,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_texts,
    )
    validation_dataloader = torch.utils.data.DataLoader(
        validation_texts,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_texts,
    )

    def validation_loss() -> float:
        model.eval()
        total_loss = 0.0
        total_batches = 0
        with torch.no_grad():
            for encoded in validation_dataloader:
                encoded = {key: value.to("cuda") for key, value in encoded.items()}
                loss = model(**encoded).loss
                if not torch.isfinite(loss):
                    raise RuntimeError("validation loss became NaN or inf")
                total_loss += float(loss.detach().cpu())
                total_batches += 1
        model.train()
        return total_loss / total_batches if total_batches else 0.0

    gpu_samples_path = progress_path.with_name("gpu-samples.csv")
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_file = progress_path.open("w", encoding="utf-8", newline="")
    progress_writer = csv.DictWriter(
        progress_file,
        fieldnames=[
            "step",
            "optimizer_steps",
            "tokens",
            "loss",
            "tokens_per_second",
            "peak_vram_gb",
            "gpu_utilization_percent",
            "gpu_memory_used_mb",
            "gpu_memory_total_mb",
            "validation_loss",
            "early_stopping_best_loss",
            "early_stopping_checks_without_improvement",
        ],
    )
    progress_writer.writeheader()
    optimizer_steps = 0
    early_stopping_state = EarlyStoppingState()
    gpu_stop_event, gpu_sampler_thread = start_gpu_sampler(gpu_samples_path)
    try:
        for _epoch in range(max_epochs):
            for step, encoded in enumerate(dataloader, start=len(losses) + 1):
                encoded = {key: value.to("cuda") for key, value in encoded.items()}
                token_count += int(encoded["attention_mask"].sum().detach().cpu())
                outputs = model(**encoded)
                loss = outputs.loss
                if not torch.isfinite(loss):
                    raise RuntimeError("training loss became NaN or inf")
                (loss / gradient_accumulation_steps).backward()
                losses.append(float(loss.detach().cpu()))
                if step % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1
                current_validation_loss = ""
                if early_stopping.enabled and step % early_stopping.eval_every_steps == 0:
                    measured_validation_loss = validation_loss()
                    validation_losses.append(measured_validation_loss)
                    early_stopping_state.update(
                        validation_loss=measured_validation_loss,
                        step=step,
                        config=early_stopping,
                    )
                    current_validation_loss = f"{measured_validation_loss:.6f}"
                if step % log_every_steps == 0:
                    elapsed = time.monotonic() - started
                    gpu_sample = nvidia_smi_sample()
                    progress_row = {
                        "step": str(step),
                        "optimizer_steps": str(optimizer_steps),
                        "tokens": str(token_count),
                        "loss": f"{losses[-1]:.6f}",
                        "tokens_per_second": f"{token_count / elapsed if elapsed else 0.0:.3f}",
                        "peak_vram_gb": f"{torch.cuda.max_memory_allocated() / 1024**3:.3f}",
                        **gpu_sample,
                        "validation_loss": current_validation_loss,
                        "early_stopping_best_loss": ""
                        if early_stopping_state.best_loss is None
                        else f"{early_stopping_state.best_loss:.6f}",
                        "early_stopping_checks_without_improvement": str(
                            early_stopping_state.checks_without_improvement
                        ),
                    }
                    progress_writer.writerow(progress_row)
                    progress_file.flush()
                    print(
                        " ".join(f"{key}={value}" for key, value in progress_row.items()),
                        flush=True,
                    )
                if early_stopping_state.stopped:
                    break
            if early_stopping_state.stopped:
                break
        if losses and len(losses) % gradient_accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
    finally:
        gpu_stop_event.set()
        gpu_sampler_thread.join(timeout=5)
        progress_file.close()
    gpu_summary = gpu_sample_summary(load_gpu_samples(gpu_samples_path))
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    train_seconds = time.monotonic() - started
    total_seconds = time.monotonic() - total_started
    return {
        "steps": float(len(losses)),
        "optimizer_steps": float(optimizer_steps),
        "final_loss": losses[-1] if losses else 0.0,
        "validation_examples": float(len(validation_rows)),
        "validation_checks": float(len(validation_losses)),
        "final_validation_loss": validation_losses[-1] if validation_losses else 0.0,
        "best_validation_loss": early_stopping_state.best_loss
        if early_stopping_state.best_loss is not None
        else 0.0,
        "best_validation_step": float(early_stopping_state.best_step),
        "early_stopped": 1.0 if early_stopping_state.stopped else 0.0,
        "early_stopping_stop_step": float(early_stopping_state.stop_step),
        "total_seconds": total_seconds,
        "tokenizer_load_seconds": tokenizer_load_seconds,
        "model_load_seconds": model_load_seconds,
        "adapter_setup_seconds": adapter_setup_seconds,
        "cuda_transfer_seconds": cuda_transfer_seconds,
        "render_seconds": render_seconds,
        "pre_train_seconds": total_seconds - train_seconds,
        "train_seconds": train_seconds,
        "tokens": float(token_count),
        "tokens_per_second": token_count / train_seconds if train_seconds else 0.0,
        "max_memory_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
        **gpu_summary,
    }


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
    max_epochs = positive_int_value(method.get("max_epochs"), "method.max_epochs")
    batch_size = positive_int_value(method.get("batch_size", 1), "method.batch_size")
    max_length = positive_int_value(method.get("max_length", 1024), "method.max_length")
    gradient_accumulation_steps = positive_int_value(
        method.get("gradient_accumulation_steps", 1),
        "method.gradient_accumulation_steps",
    )
    log_every_steps = positive_int_value(method.get("log_every_steps", 50), "method.log_every_steps")
    early_stopping = early_stopping_config(config)
    gradient_checkpointing = bool_value(
        method.get("gradient_checkpointing", False),
        "method.gradient_checkpointing",
    )
    student_model = string_value(model.get("student"), "model.student")
    torch_dtype = optional_string_value(model.get("torch_dtype"), "model.torch_dtype")

    rows = load_training_rows(train_export, max_examples)
    training_rows, validation_rows = split_training_rows(rows, early_stopping)
    if dry_run:
        return [
            f"would train {len(training_rows)} rows from {train_export}",
            f"validation rows: {len(validation_rows)}",
            f"student model: {student_model}",
            f"batch size: {batch_size}",
            f"max length: {max_length}",
            f"gradient checkpointing: {gradient_checkpointing}",
            f"gradient accumulation steps: {gradient_accumulation_steps}",
            f"log every steps: {log_every_steps}",
            f"early stopping: {early_stopping.enabled}",
            f"adapter output: {adapter_dir}",
            f"metrics output: {metrics_path}",
        ]

    modules = require_training_packages()
    torch = modules["torch"]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; run this smoke on a CUDA RunPod image")

    logs_dir.mkdir(parents=True, exist_ok=True)
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = train_lora_smoke(
        modules=modules,
        rows=training_rows,
        validation_rows=validation_rows,
        student_model=student_model,
        adapter_dir=adapter_dir,
        progress_path=logs_dir / "progress.csv",
        max_epochs=max_epochs,
        batch_size=batch_size,
        max_length=max_length,
        torch_dtype=torch_dtype,
        gradient_checkpointing=gradient_checkpointing,
        gradient_accumulation_steps=gradient_accumulation_steps,
        log_every_steps=log_every_steps,
        early_stopping=early_stopping,
    )
    write_metrics(
        metrics_path,
        [
            {"metric": "rows", "value": str(len(rows))},
            {"metric": "training_rows", "value": str(len(training_rows))},
            {"metric": "validation_rows", "value": str(len(validation_rows))},
            {"metric": "student_model", "value": student_model},
            {"metric": "cuda_device", "value": torch.cuda.get_device_name(0)},
            {"metric": "epochs", "value": str(max_epochs)},
            {"metric": "batch_size", "value": str(batch_size)},
            {"metric": "max_length", "value": str(max_length)},
            {"metric": "gradient_checkpointing", "value": str(gradient_checkpointing)},
            {"metric": "gradient_accumulation_steps", "value": str(gradient_accumulation_steps)},
            {"metric": "log_every_steps", "value": str(log_every_steps)},
            {"metric": "early_stopping_enabled", "value": str(early_stopping.enabled)},
            {"metric": "early_stopping_eval_every_steps", "value": str(early_stopping.eval_every_steps)},
            {"metric": "early_stopping_patience", "value": str(early_stopping.patience)},
            {"metric": "early_stopping_min_delta", "value": f"{early_stopping.min_delta:.6f}"},
            {"metric": "steps", "value": str(int(metrics["steps"]))},
            {"metric": "optimizer_steps", "value": str(int(metrics["optimizer_steps"]))},
            {"metric": "tokens", "value": str(int(metrics["tokens"]))},
            {"metric": "total_seconds", "value": f"{metrics['total_seconds']:.3f}"},
            {"metric": "pre_train_seconds", "value": f"{metrics['pre_train_seconds']:.3f}"},
            {"metric": "tokenizer_load_seconds", "value": f"{metrics['tokenizer_load_seconds']:.3f}"},
            {"metric": "model_load_seconds", "value": f"{metrics['model_load_seconds']:.3f}"},
            {"metric": "adapter_setup_seconds", "value": f"{metrics['adapter_setup_seconds']:.3f}"},
            {"metric": "cuda_transfer_seconds", "value": f"{metrics['cuda_transfer_seconds']:.3f}"},
            {"metric": "render_seconds", "value": f"{metrics['render_seconds']:.3f}"},
            {"metric": "train_seconds", "value": f"{metrics['train_seconds']:.3f}"},
            {"metric": "tokens_per_second", "value": f"{metrics['tokens_per_second']:.3f}"},
            {"metric": "max_memory_allocated_gb", "value": f"{metrics['max_memory_allocated_gb']:.3f}"},
            {"metric": "gpu_sample_count", "value": str(int(metrics["gpu_sample_count"]))},
            {
                "metric": "gpu_utilization_avg_percent",
                "value": f"{metrics['gpu_utilization_avg_percent']:.3f}",
            },
            {
                "metric": "gpu_utilization_max_percent",
                "value": f"{metrics['gpu_utilization_max_percent']:.3f}",
            },
            {"metric": "gpu_memory_used_max_mb", "value": f"{metrics['gpu_memory_used_max_mb']:.0f}"},
            {"metric": "gpu_memory_total_mb", "value": f"{metrics['gpu_memory_total_mb']:.0f}"},
            {"metric": "final_loss", "value": f"{metrics['final_loss']:.6f}"},
            {"metric": "validation_checks", "value": str(int(metrics["validation_checks"]))},
            {"metric": "final_validation_loss", "value": f"{metrics['final_validation_loss']:.6f}"},
            {"metric": "best_validation_loss", "value": f"{metrics['best_validation_loss']:.6f}"},
            {"metric": "best_validation_step", "value": str(int(metrics["best_validation_step"]))},
            {"metric": "early_stopped", "value": str(bool(metrics["early_stopped"]))},
            {
                "metric": "early_stopping_stop_step",
                "value": str(int(metrics["early_stopping_stop_step"])),
            },
            {"metric": "status", "value": "completed"},
        ],
    )
    notes_path.write_text(
        "\n".join(
            [
                "# Leverage SFT Smoke",
                "",
                f"Rows: {len(rows)}",
                f"Training rows: {len(training_rows)}",
                f"Validation rows: {len(validation_rows)}",
                f"Epochs: {max_epochs}",
                f"Batch size: {batch_size}",
                f"Max length: `{max_length}`",
                f"Gradient checkpointing: `{gradient_checkpointing}`",
                f"Gradient accumulation steps: `{gradient_accumulation_steps}`",
                f"Log every steps: `{log_every_steps}`",
                f"Early stopping enabled: `{early_stopping.enabled}`",
                f"Early stopping eval every steps: `{early_stopping.eval_every_steps}`",
                f"Early stopping patience: `{early_stopping.patience}`",
                f"Early stopping min delta: `{early_stopping.min_delta:.6f}`",
                f"Student model: `{student_model}`",
                f"CUDA device: `{torch.cuda.get_device_name(0)}`",
                f"Total seconds: `{metrics['total_seconds']:.3f}`",
                f"Pre-train seconds: `{metrics['pre_train_seconds']:.3f}`",
                f"Tokenizer load seconds: `{metrics['tokenizer_load_seconds']:.3f}`",
                f"Model load seconds: `{metrics['model_load_seconds']:.3f}`",
                f"CUDA transfer seconds: `{metrics['cuda_transfer_seconds']:.3f}`",
                f"Train seconds: `{metrics['train_seconds']:.3f}`",
                f"Tokens/sec: `{metrics['tokens_per_second']:.3f}`",
                f"Peak VRAM GB: `{metrics['max_memory_allocated_gb']:.3f}`",
                f"GPU utilization avg percent: `{metrics['gpu_utilization_avg_percent']:.3f}`",
                f"GPU utilization max percent: `{metrics['gpu_utilization_max_percent']:.3f}`",
                f"Final loss: `{metrics['final_loss']:.6f}`",
                f"Validation checks: `{metrics['validation_checks']:.0f}`",
                f"Final validation loss: `{metrics['final_validation_loss']:.6f}`",
                f"Best validation loss: `{metrics['best_validation_loss']:.6f}`",
                f"Best validation step: `{metrics['best_validation_step']:.0f}`",
                f"Early stopped: `{bool(metrics['early_stopped'])}`",
                f"Early stopping stop step: `{metrics['early_stopping_stop_step']:.0f}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return [
        f"loaded {len(rows)} rows",
        f"training rows: {len(training_rows)}",
        f"validation rows: {len(validation_rows)}",
        f"cuda device: {torch.cuda.get_device_name(0)}",
        f"batch size: {batch_size}",
        f"gradient accumulation steps: {gradient_accumulation_steps}",
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
