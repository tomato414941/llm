import csv
import json
from pathlib import Path

import pytest

from llm.leverage import evaluate_sft_adapter
from llm.leverage.evaluate_sft_adapter import (
    config_defaults,
    extract_qwen_final_response,
    parse_predictions,
    parse_qwen_final_predictions,
    prediction_paths,
    require_inference_packages,
    render_qwen_messages,
    run_eval,
    write_predictions,
)
from llm.leverage.evaluate import Prediction, load_predictions


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def write_config(tmp_path: Path) -> Path:
    task_path = tmp_path / "tracks" / "leverage" / "evals" / "smoke.jsonl"
    write_jsonl(
        task_path,
        [
            {
                "id": "task-1",
                "capability": "knowledge_qa",
                "prompt": "Return ok.",
                "scoring": {"type": "exact", "expected": "ok"},
            }
        ],
    )
    config_path = tmp_path / "tracks" / "leverage" / "configs" / "leverage-sft-smoke.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        f"""
[model]
student = "base/model"

[data]
eval_tasks = ["{task_path}"]

[method]
max_train_examples = 10
max_epochs = 3
eval_max_new_tokens = 32

[outputs]
root = "{tmp_path / "outputs" / "leverage-sft-smoke"}"
adapter_dir = "{tmp_path / "outputs" / "leverage-sft-smoke" / "lora-adapter"}"
logs = "{tmp_path / "outputs" / "leverage-sft-smoke" / "logs"}"
metrics = "{tmp_path / "outputs" / "leverage-sft-smoke" / "metrics.csv"}"
notes = "{tmp_path / "outputs" / "leverage-sft-smoke" / "notes.md"}"
""",
        encoding="utf-8",
    )
    return config_path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_config_defaults_reads_sft_smoke_eval_inputs(tmp_path: Path) -> None:
    config_path = write_config(tmp_path)

    defaults = config_defaults(config_path)

    assert defaults["base_model"] == "base/model"
    assert defaults["adapter_dir"] == tmp_path / "outputs" / "leverage-sft-smoke" / "lora-adapter"
    assert defaults["max_new_tokens"] == 32
    assert len(defaults["tasks"]) == 1


def test_prediction_paths_stay_under_output_root() -> None:
    predictions, scores, summary = prediction_paths(Path("outputs/leverage-sft-smoke"))

    assert predictions == Path("outputs/leverage-sft-smoke/post-training-predictions.jsonl")
    assert scores == Path("outputs/leverage-sft-smoke/post-training-scores.csv")
    assert summary == Path("outputs/leverage-sft-smoke/post-training-summary.csv")


def test_run_eval_dry_run_reports_comparison_plan(tmp_path: Path) -> None:
    config_path = write_config(tmp_path)
    defaults = config_defaults(config_path)

    lines = run_eval(
        tasks_paths=defaults["tasks"],
        base_model="base/model",
        adapter_dir=Path("outputs/adapter"),
        output_root=Path("outputs/eval"),
        base_label="base",
        adapter_label="adapter",
        max_new_tokens=32,
        system_prompt="system",
        device="cuda",
        dry_run=True,
    )

    assert any("would evaluate 1 tasks" in line for line in lines)
    assert any("base/model" in line for line in lines)
    assert any("post-training-summary.csv" in line for line in lines)


def test_run_eval_requires_adapter_for_real_run(tmp_path: Path) -> None:
    config_path = write_config(tmp_path)
    defaults = config_defaults(config_path)

    with pytest.raises(FileNotFoundError, match="adapter directory"):
        run_eval(
            tasks_paths=defaults["tasks"],
            base_model="base/model",
            adapter_dir=tmp_path / "missing-adapter",
            output_root=tmp_path / "outputs",
            base_label="base",
            adapter_label="adapter",
            max_new_tokens=32,
            system_prompt="system",
            device="cpu",
            dry_run=False,
        )


def test_write_predictions_uses_existing_evaluate_contract(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.jsonl"

    write_predictions(
        predictions_path,
        [
            Prediction(task_id="task-1", model="base", response="ok"),
            Prediction(task_id="task-1", model="adapter", response="ok"),
        ],
    )

    rows = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {"task_id": "task-1", "model": "base", "response": "ok"},
        {"task_id": "task-1", "model": "adapter", "response": "ok"},
    ]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("</think>\n\nParis", "Paris"),
        ("assistant\n<think>\n\n</think>\n\nWilliam Shakespeare", "William Shakespeare"),
        ("user\n</think>\n\nyes", "yes"),
        ("<think>scratch work</think>\n\n{\"status\":\"ok\"}", "{\"status\":\"ok\"}"),
    ],
)
def test_extract_qwen_final_response_returns_content_after_thinking_block(raw: str, expected: str) -> None:
    assert extract_qwen_final_response(raw) == expected


def test_parse_qwen_final_predictions_preserves_model_and_task_id() -> None:
    parsed = parse_qwen_final_predictions(
        [Prediction(task_id="task-1", model="adapter", response="assistant\n</think>\n\nok")]
    )

    assert parsed == [Prediction(task_id="task-1", model="adapter", response="ok")]


def test_render_qwen_messages_disables_thinking_when_tokenizer_supports_it() -> None:
    class Tokenizer:
        chat_template = "template"

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
            assert messages == [{"role": "user", "content": "Return ok."}]
            assert tokenize is False
            assert add_generation_prompt is True
            assert enable_thinking is False
            return "rendered"

    assert render_qwen_messages([{"role": "user", "content": "Return ok."}], Tokenizer()) == "rendered"


def test_render_qwen_messages_falls_back_when_enable_thinking_is_not_supported() -> None:
    class Tokenizer:
        chat_template = "template"

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert tokenize is False
            assert add_generation_prompt is True
            return "fallback-rendered"

    assert render_qwen_messages([{"role": "user", "content": "Return ok."}], Tokenizer()) == "fallback-rendered"


def test_written_predictions_can_be_scored_by_existing_evaluator(tmp_path: Path) -> None:
    task_path = tmp_path / "tasks.jsonl"
    write_jsonl(
        task_path,
        [
            {
                "id": "task-1",
                "capability": "knowledge_qa",
                "prompt": "Return ok.",
                "scoring": {"type": "exact", "expected": "ok"},
            }
        ],
    )
    predictions_path = tmp_path / "predictions.jsonl"
    scores_path = tmp_path / "scores.csv"

    write_predictions(
        predictions_path,
        [
            Prediction(task_id="task-1", model="base", response="wrong"),
            Prediction(task_id="task-1", model="adapter", response="ok"),
        ],
    )
    tasks = evaluate_sft_adapter.load_task_suites([task_path])
    predictions = load_predictions(predictions_path, set(tasks))
    results = evaluate_sft_adapter.evaluate_predictions(tasks, predictions)
    evaluate_sft_adapter.write_results(scores_path, results)

    rows = read_csv(scores_path)
    assert rows[0]["model"] == "base"
    assert rows[0]["passed"] == "false"
    assert rows[1]["model"] == "adapter"
    assert rows[1]["passed"] == "true"


def test_parse_predictions_writes_qwen_final_predictions_scores_and_summary(tmp_path: Path) -> None:
    task_path = tmp_path / "tasks.jsonl"
    write_jsonl(
        task_path,
        [
            {
                "id": "task-1",
                "capability": "knowledge_qa",
                "prompt": "Return ok.",
                "scoring": {"type": "exact", "expected": "ok"},
            }
        ],
    )
    predictions_path = tmp_path / "predictions.jsonl"
    write_predictions(predictions_path, [Prediction(task_id="task-1", model="adapter", response="</think>\n\nok")])
    output_root = tmp_path / "outputs"

    lines = parse_predictions(
        tasks_paths=[task_path],
        predictions_path=predictions_path,
        output_root=output_root,
    )

    assert any("parsed and scored 1 Qwen final responses" in line for line in lines)
    assert (output_root / "post-training-predictions.qwen-final.jsonl").exists()
    rows = read_csv(output_root / "post-training-scores.qwen-final.csv")
    assert rows[0]["passed"] == "true"
    summary = read_csv(output_root / "post-training-summary.qwen-final.csv")
    assert summary[0]["passed_count"] == "1"


def test_require_inference_packages_reports_missing_optional_dependencies(monkeypatch) -> None:
    monkeypatch.setattr(evaluate_sft_adapter, "REQUIRED_PACKAGES", ("definitely_missing_inference_package",))

    with pytest.raises(RuntimeError, match="missing optional SFT inference packages"):
        require_inference_packages()
