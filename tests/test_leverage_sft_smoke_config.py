from pathlib import Path
import tomllib


CONFIG_PATH = Path("tracks/leverage/configs/leverage-sft-smoke.toml")
QWEN35_9B_CONFIG_PATH = Path("tracks/leverage/configs/leverage-sft-qwen35-9b.toml")
QWEN35_9B_BATCH2_200_CONFIG_PATH = Path("tracks/leverage/configs/leverage-sft-qwen35-9b-batch2-200.toml")
QWEN35_9B_BATCH4_200_CONFIG_PATH = Path("tracks/leverage/configs/leverage-sft-qwen35-9b-batch4-200.toml")


def load_config() -> dict[str, object]:
    return tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_leverage_sft_smoke_config_references_existing_inputs() -> None:
    config = load_config()
    model = config["model"]
    data = config["data"]
    assert isinstance(model, dict)
    assert isinstance(data, dict)

    assert model["student"] == "Qwen/Qwen3.5-0.8B"
    assert model["target_baseline"] == "Qwen/Qwen3.5-9B"
    assert Path(data["reviewed-instructions"]).exists()
    for task in data["eval_tasks"]:
        assert Path(task).exists()


def test_leverage_sft_smoke_config_keeps_paid_run_bounded() -> None:
    config = load_config()
    method = config["method"]
    runpod = config["runpod"]
    stop = config["stop"]
    success = config["success"]
    assert isinstance(method, dict)
    assert isinstance(runpod, dict)
    assert isinstance(stop, dict)
    assert isinstance(success, dict)

    assert method["first_choice"] == "lora"
    assert method["max_train_examples"] == 1200
    assert method["max_epochs"] == 1
    assert method["batch_size"] == 4
    assert method["max_runtime_minutes"] <= 60
    assert runpod["required"] is False
    assert runpod["max_cost_usd"] <= 1.0
    assert runpod["cleanup_required"] is True
    assert any("cost cap" in condition for condition in stop["conditions"])
    assert any("training export" in criterion for criterion in success["criteria"])


def test_qwen35_9b_sft_config_is_bounded_baseline() -> None:
    config = tomllib.loads(QWEN35_9B_CONFIG_PATH.read_text(encoding="utf-8"))
    model = config["model"]
    data = config["data"]
    method = config["method"]
    runpod = config["runpod"]
    outputs = config["outputs"]
    assert isinstance(model, dict)
    assert isinstance(data, dict)
    assert isinstance(method, dict)
    assert isinstance(runpod, dict)
    assert isinstance(outputs, dict)

    assert model["student"] == "Qwen/Qwen3.5-9B"
    assert model["torch_dtype"] == "bfloat16"
    assert Path(data["reviewed-instructions"]).exists()
    train_export = Path(data["train_export"])
    assert train_export.parent.exists()
    assert train_export.suffix == ".jsonl"
    assert method["max_train_examples"] == 1200
    assert method["max_epochs"] == 1
    assert method["batch_size"] == 2
    assert method["max_length"] == 512
    assert method["gradient_checkpointing"] is True
    assert method["gradient_accumulation_steps"] == 4
    assert method["log_every_steps"] == 10
    assert method["max_runtime_minutes"] <= 60
    assert runpod["required"] is False
    assert runpod["cleanup_required"] is True
    assert outputs["root"] == "outputs/leverage-sft-qwen35-9b"


def test_qwen35_9b_batch2_200_config_is_bounded_measurement() -> None:
    config = tomllib.loads(QWEN35_9B_BATCH2_200_CONFIG_PATH.read_text(encoding="utf-8"))
    model = config["model"]
    data = config["data"]
    method = config["method"]
    runpod = config["runpod"]
    outputs = config["outputs"]
    assert isinstance(model, dict)
    assert isinstance(data, dict)
    assert isinstance(method, dict)
    assert isinstance(runpod, dict)
    assert isinstance(outputs, dict)

    assert model["student"] == "Qwen/Qwen3.5-9B"
    assert data["train_export"] == "tracks/leverage/sft/bootstrap-200.train.jsonl"
    assert method["max_train_examples"] == 200
    assert method["batch_size"] == 2
    assert method["max_length"] == 512
    assert method["gradient_checkpointing"] is True
    assert method["gradient_accumulation_steps"] == 4
    assert method["log_every_steps"] == 10
    assert method["max_runtime_minutes"] <= 30
    assert runpod["max_cost_usd"] <= 0.5
    assert outputs["root"] == "outputs/leverage-sft-qwen35-9b-batch2-200"


def test_qwen35_9b_batch4_200_config_is_bounded_measurement() -> None:
    config = tomllib.loads(QWEN35_9B_BATCH4_200_CONFIG_PATH.read_text(encoding="utf-8"))
    model = config["model"]
    data = config["data"]
    method = config["method"]
    runpod = config["runpod"]
    outputs = config["outputs"]
    assert isinstance(model, dict)
    assert isinstance(data, dict)
    assert isinstance(method, dict)
    assert isinstance(runpod, dict)
    assert isinstance(outputs, dict)

    assert model["student"] == "Qwen/Qwen3.5-9B"
    assert data["train_export"] == "tracks/leverage/sft/bootstrap-200.train.jsonl"
    assert method["max_train_examples"] == 200
    assert method["batch_size"] == 4
    assert method["max_length"] == 512
    assert method["gradient_checkpointing"] is True
    assert method["gradient_accumulation_steps"] == 4
    assert method["log_every_steps"] == 10
    assert method["max_runtime_minutes"] <= 30
    assert runpod["max_cost_usd"] <= 0.5
    assert outputs["root"] == "outputs/leverage-sft-qwen35-9b-batch4-200"


def test_leverage_sft_smoke_doc_exists() -> None:
    doc = Path("tracks/leverage/docs/sft-smoke.md")

    assert doc.exists()
    text = doc.read_text(encoding="utf-8")
    assert "not a claim that the model improves" in text
    assert "Qwen/Qwen3.5-9B" in text
    assert "Do not launch a paid GPU job" in text
