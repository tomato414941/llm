import json
from pathlib import Path

import pytest

from llm.leverage import collect_openai
from llm.leverage.evaluate import Task


def task(task_id: str, prompt: str = "Return ok.") -> Task:
    return Task(
        id=task_id,
        category="qa",
        prompt=prompt,
        scoring={"type": "exact", "expected": "ok"},
        suite="suite",
    )


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_build_payload_disables_qwen_thinking_with_chat_template_kwargs() -> None:
    payload = collect_openai.build_payload(
        task("first", "Say ok."),
        model="Qwen/Qwen3.5-35B-A3B",
        system_prompt="System.",
        max_tokens=64,
        temperature=0.0,
        thinking_mode="disabled",
        thinking_param="chat_template_kwargs",
        reasoning_effort="none",
        exclude_reasoning=True,
    )

    assert payload["model"] == "Qwen/Qwen3.5-35B-A3B"
    assert payload["messages"] == [
        {"role": "system", "content": "System."},
        {"role": "user", "content": "Say ok."},
    ]
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["reasoning"] == {"exclude": True, "effort": "none"}


def test_build_payload_can_omit_thinking_controls() -> None:
    payload = collect_openai.build_payload(
        task("first"),
        model="model",
        system_prompt="System.",
        max_tokens=64,
        temperature=0.0,
        thinking_mode="none",
        thinking_param="chat_template_kwargs",
        reasoning_effort="provider_default",
        exclude_reasoning=False,
    )

    assert "chat_template_kwargs" not in payload
    assert "enable_thinking" not in payload
    assert "reasoning" not in payload


def test_response_text_reads_openai_chat_completion_text() -> None:
    assert (
        collect_openai.response_text(
            {"choices": [{"message": {"content": "answer"}}]},
        )
        == "answer"
    )


def test_response_result_reads_finish_reason_and_usage() -> None:
    result = collect_openai.response_result(
        {
            "choices": [{"message": {"content": "answer"}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 32, "total_tokens": 42},
        }
    )

    assert result.text == "answer"
    assert result.finish_reason == "length"
    assert result.usage == {"prompt_tokens": 10, "completion_tokens": 32, "total_tokens": 42}


def test_response_text_reads_openai_content_parts() -> None:
    assert (
        collect_openai.response_text(
            {
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"type": "text", "text": "first"},
                                {"type": "output_text", "content": " second"},
                                " third",
                            ]
                        }
                    }
                ]
            },
        )
        == "first second third"
    )


def test_chat_client_sends_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_headers: list[str] = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"ok"}}]}'

    def fake_urlopen(http_request, timeout):
        seen_headers.append(http_request.get_header("User-agent"))
        assert timeout == 3
        return Response()

    monkeypatch.setattr(collect_openai.request, "urlopen", fake_urlopen)

    client = collect_openai.chat_completions_client("https://example.test/v1", "secret", 3)

    assert client({"messages": []}) == collect_openai.ChatResult("ok", None, {})
    assert seen_headers == [collect_openai.DEFAULT_USER_AGENT]


def test_collect_predictions_writes_prediction_jsonl(tmp_path: Path) -> None:
    output = tmp_path / "predictions.jsonl"
    payloads: list[dict[str, object]] = []

    def client(payload):
        payloads.append(payload)
        return collect_openai.ChatResult(
            f"response for {payload['messages'][1]['content']}",
            "stop",
            {"completion_tokens": 4},
        )

    collect_openai.collect_predictions(
        {
            "first": task("first", "First?"),
            "second": task("second", "Second?"),
        },
        client=client,
        output_path=output,
        api_model="Qwen/Qwen3.5-35B-A3B",
        model_label="qwen3_5_35b_a3b",
        system_prompt="System.",
        max_tokens=32,
        temperature=0.0,
        thinking_mode="disabled",
        thinking_param="chat_template_kwargs",
        reasoning_effort="none",
        exclude_reasoning=True,
        overwrite=False,
    )

    assert [payload["messages"][1]["content"] for payload in payloads] == ["First?", "Second?"]
    assert read_jsonl(output) == [
        {
            "task_id": "first",
            "model": "qwen3_5_35b_a3b",
            "response": "response for First?",
            "generation": {
                "api_model": "Qwen/Qwen3.5-35B-A3B",
                "max_tokens": 32,
                "temperature": 0.0,
                "finish_reason": "stop",
                "usage": {"completion_tokens": 4},
            },
        },
        {
            "task_id": "second",
            "model": "qwen3_5_35b_a3b",
            "response": "response for Second?",
            "generation": {
                "api_model": "Qwen/Qwen3.5-35B-A3B",
                "max_tokens": 32,
                "temperature": 0.0,
                "finish_reason": "stop",
                "usage": {"completion_tokens": 4},
            },
        },
    ]


def test_collect_predictions_refuses_existing_output_without_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "predictions.jsonl"
    output.write_text("", encoding="utf-8")

    with pytest.raises(FileExistsError, match="output already exists"):
        collect_openai.collect_predictions(
            {"first": task("first")},
            client=lambda _payload: "ok",
            output_path=output,
            api_model="model",
            model_label="label",
            system_prompt="System.",
            max_tokens=32,
            temperature=0.0,
            thinking_mode="disabled",
            thinking_param="chat_template_kwargs",
            reasoning_effort="none",
            exclude_reasoning=True,
            overwrite=False,
        )


def test_environment_value_rejects_missing_secret(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        collect_openai.environment_value("OPENAI_API_KEY")


def test_redact_hides_api_key() -> None:
    assert collect_openai.redact("failed with secret-key", ["secret-key"]) == (
        "failed with [REDACTED]"
    )
