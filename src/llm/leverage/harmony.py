import re
from typing import Any


FINAL_MARKER_RE = re.compile(r"(?:assistant\s*)?final", re.IGNORECASE)


def _string_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, str):
            return content
    return str(value)


def extract_harmony_final_response(response: Any) -> str:
    """Extract the visible final answer from gpt-oss Harmony text."""
    text = _string_content(response).strip()
    matches = list(FINAL_MARKER_RE.finditer(text))
    if matches:
        text = text[matches[-1].end() :]
    text = re.sub(r"^\s*(assistant\s*)?(analysis|commentary)\b", "", text, flags=re.IGNORECASE)
    return text.strip()
