import csv
import json
from pathlib import Path

import pytest

from llm.leverage import evaluate_openrouter
from llm.leverage.collect_openai import ChatResult


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def task_path(tmp_path: Path) -> Path:
    path = tmp_path / "evals" / "tasks.jsonl"
    write_jsonl(
        path,
        [
            {
                "id": "task-1",
                "capability": "knowledge_qa",
                "prompt": "Return ok.",
                "scoring": {"type": "exact", "expected": "ok"},
            }
        ],
    )
    return path


def test_parse_model_requires_label_and_api_model() -> None:
    assert evaluate_openrouter.parse_model("label=provider/model") == ("label", "provider/model")

    with pytest.raises(ValueError, match="label=api_model"):
        evaluate_openrouter.parse_model("provider/model")


def test_dry_run_reports_models_and_outputs(tmp_path: Path) -> None:
    lines = evaluate_openrouter.run_eval(
        tasks_paths=[task_path(tmp_path)],
        models=[("label", "provider/model")],
        output_root=tmp_path / "outputs",
        max_tokens=16,
        temperature=0.0,
        system_prompt="system",
        timeout_seconds=1.0,
        thinking_mode="none",
        thinking_param="chat_template_kwargs",
        reasoning_effort="provider_default",
        exclude_reasoning=True,
        overwrite=False,
        resume=False,
        dry_run=True,
    )

    assert any("would evaluate 1 tasks" in line for line in lines)
    assert any("label=provider/model" in line for line in lines)
    assert any("openrouter-summary.csv" in line for line in lines)
    assert any("openrouter-run.json" in line for line in lines)


def test_run_eval_collects_scores_for_multiple_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payloads: list[dict[str, object]] = []

    def fake_client(_base_url: str, _api_key: str, _timeout_seconds: float):
        def complete(payload):
            payloads.append(payload)
            content = "ok" if payload["model"] == "provider/good" else "wrong"
            return ChatResult(content, "stop", {"completion_tokens": 1})

        return complete

    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setattr(evaluate_openrouter, "chat_completions_client", fake_client)
    output_root = tmp_path / "outputs"

    lines = evaluate_openrouter.run_eval(
        tasks_paths=[task_path(tmp_path)],
        models=[("good", "provider/good"), ("bad", "provider/bad")],
        output_root=output_root,
        max_tokens=16,
        temperature=0.0,
        system_prompt="system",
        timeout_seconds=1.0,
        thinking_mode="none",
        thinking_param="chat_template_kwargs",
        reasoning_effort="provider_default",
        exclude_reasoning=True,
        overwrite=False,
        dry_run=False,
        resume=False,
    )

    assert len(payloads) == 2
    assert any("evaluated 1 tasks with 2 models" in line for line in lines)
    predictions = [
        json.loads(line)
        for line in (output_root / "openrouter-predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["model"] for row in predictions] == ["good", "bad"]
    scores = read_csv(output_root / "openrouter-scores.csv")
    assert [(row["model"], row["passed"]) for row in scores] == [("good", "true"), ("bad", "false")]
    metadata = json.loads((output_root / "openrouter-run.json").read_text(encoding="utf-8"))
    assert metadata["reasoning_effort"] == "provider_default"
    assert metadata["exclude_reasoning"] is True
    assert metadata["models"] == [
        {"label": "good", "api_model": "provider/good"},
        {"label": "bad", "api_model": "provider/bad"},
    ]


def test_run_eval_resume_only_requests_missing_predictions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tasks_path = tmp_path / "evals" / "tasks.jsonl"
    write_jsonl(
        tasks_path,
        [
            {
                "id": "task-1",
                "capability": "knowledge_qa",
                "prompt": "Return ok.",
                "scoring": {"type": "exact", "expected": "ok"},
            },
            {
                "id": "task-2",
                "capability": "knowledge_qa",
                "prompt": "Return ok.",
                "scoring": {"type": "exact", "expected": "ok"},
            },
        ],
    )
    output_root = tmp_path / "outputs"
    predictions = output_root / "openrouter-predictions.jsonl"
    write_jsonl(predictions, [{"task_id": "task-1", "model": "model", "response": "ok"}])
    payloads: list[dict[str, object]] = []

    def fake_client(_base_url: str, _api_key: str, _timeout_seconds: float):
        def complete(payload):
            payloads.append(payload)
            return ChatResult("ok", "stop", {})

        return complete

    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setattr(evaluate_openrouter, "chat_completions_client", fake_client)

    evaluate_openrouter.run_eval(
        tasks_paths=[tasks_path],
        models=[("model", "provider/model")],
        output_root=output_root,
        max_tokens=16,
        temperature=0.0,
        system_prompt="system",
        timeout_seconds=1.0,
        thinking_mode="none",
        thinking_param="chat_template_kwargs",
        reasoning_effort="provider_default",
        exclude_reasoning=True,
        overwrite=False,
        resume=True,
        dry_run=False,
    )

    assert [payload["messages"][1]["content"] for payload in payloads] == ["Return ok."]
    assert len(predictions.read_text(encoding="utf-8").splitlines()) == 2
