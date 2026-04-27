import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable
from urllib import error, request

from llm.leverage.evaluate import Task, load_task_suites


DEFAULT_MODEL = "Qwen/Qwen3.5-35B-A3B"
DEFAULT_MODEL_LABEL = "qwen3_5_35b_a3b"
DEFAULT_SYSTEM_PROMPT = "Return only the requested answer. Do not include hidden reasoning."


ChatClient = Callable[[dict[str, Any]], str]


def build_payload(
    task: Task,
    *,
    model: str,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
    thinking_mode: str,
    thinking_param: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task.prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if thinking_mode == "disabled":
        if thinking_param == "chat_template_kwargs":
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        elif thinking_param == "enable_thinking":
            payload["enable_thinking"] = False
    return payload


def chat_completions_client(base_url: str, api_key: str, timeout_seconds: float) -> ChatClient:
    endpoint = base_url.rstrip("/") + "/chat/completions"

    def complete(payload: dict[str, Any]) -> str:
        body = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            endpoint,
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=timeout_seconds) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"OpenAI-compatible API request failed with HTTP {exc.code}: "
                f"{redact(detail, [api_key])}"
            ) from exc
        except error.URLError as exc:
            reason = redact(str(exc.reason), [api_key])
            raise RuntimeError(f"OpenAI-compatible API request failed: {reason}") from exc
        return response_text(response_payload)

    return complete


def redact(text: str, secrets: list[str]) -> str:
    redacted = text
    for secret in secrets:
        if secret:
            redacted = redacted.replace(secret, "[REDACTED]")
    return redacted


def response_text(response_payload: dict[str, Any]) -> str:
    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("API response must contain at least one choice")
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise ValueError("API response choice must be an object")
    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("API response choice must contain a message object")
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        if parts:
            return "".join(parts)
    raise ValueError("API response message content must be text")


def collect_predictions(
    tasks: dict[str, Task],
    *,
    client: ChatClient,
    output_path: Path,
    api_model: str,
    model_label: str,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
    thinking_mode: str,
    thinking_param: str,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for task in tasks.values():
            payload = build_payload(
                task,
                model=api_model,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                thinking_mode=thinking_mode,
                thinking_param=thinking_param,
            )
            response = client(payload)
            record = {
                "task_id": task.id,
                "model": model_label,
                "response": response,
            }
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_file.flush()


def environment_value(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"{name} must be set")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True, action="append")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-label", default=DEFAULT_MODEL_LABEL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--thinking-mode",
        choices=("disabled", "none"),
        default="disabled",
        help="Use 'none' to avoid sending provider-specific thinking controls.",
    )
    parser.add_argument(
        "--thinking-param",
        choices=("chat_template_kwargs", "enable_thinking"),
        default="chat_template_kwargs",
        help="Use enable_thinking for providers such as DashScope that expect a top-level flag.",
    )
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    if args.temperature < 0:
        raise ValueError("--temperature must be non-negative")
    if not args.model_label:
        raise ValueError("--model-label must not be empty")
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive")


def main() -> None:
    args = parse_args()
    validate_args(args)
    tasks = load_task_suites(args.tasks)
    base_url = environment_value("OPENAI_BASE_URL")
    api_key = environment_value("OPENAI_API_KEY")
    client = chat_completions_client(base_url, api_key, args.timeout_seconds)
    collect_predictions(
        tasks,
        client=client,
        output_path=args.output,
        api_model=args.model,
        model_label=args.model_label,
        system_prompt=args.system_prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        thinking_mode=args.thinking_mode,
        thinking_param=args.thinking_param,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
