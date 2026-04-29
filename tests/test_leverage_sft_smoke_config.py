from pathlib import Path
import tomllib


CONFIG_PATH = Path("tracks/leverage/configs/leverage-sft-smoke.toml")


def load_config() -> dict[str, object]:
    return tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_leverage_sft_smoke_config_references_existing_inputs() -> None:
    config = load_config()
    data = config["data"]
    assert isinstance(data, dict)

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
    assert method["max_train_examples"] == 16
    assert method["max_runtime_minutes"] <= 30
    assert runpod["required"] is False
    assert runpod["max_cost_usd"] <= 1.0
    assert runpod["cleanup_required"] is True
    assert any("cost cap" in condition for condition in stop["conditions"])
    assert any("training export" in criterion for criterion in success["criteria"])


def test_leverage_sft_smoke_doc_exists() -> None:
    doc = Path("tracks/leverage/docs/sft-smoke.md")

    assert doc.exists()
    text = doc.read_text(encoding="utf-8")
    assert "not a claim that the model improves" in text
    assert "Do not launch a paid GPU job" in text
