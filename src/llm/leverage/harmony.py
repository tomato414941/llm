import re
from dataclasses import dataclass
from typing import Any


FINAL_MARKER_RE = re.compile(r"(?:assistant\s*)?final", re.IGNORECASE)
NON_FINAL_CHANNEL_RE = re.compile(r"\b(?:assistant\s*)?(?:analysis|commentary)\b", re.IGNORECASE)
LEADING_NON_FINAL_CHANNEL_RE = re.compile(r"^\s*(?:assistant\s*)?(?:analysis|commentary)\b", re.IGNORECASE)


@dataclass(frozen=True)
class HarmonyExtraction:
    final_response: str
    final_marker_found: bool
    final_response_empty: bool
    non_final_channel_in_final: bool


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
    return analyze_harmony_response(response).final_response


def analyze_harmony_response(response: Any) -> HarmonyExtraction:
    """Extract the final answer and report common Harmony contract issues."""
    text = _string_content(response).strip()
    matches = list(FINAL_MARKER_RE.finditer(text))
    final_marker_found = bool(matches)
    if matches:
        text = text[matches[-1].end() :]
    elif LEADING_NON_FINAL_CHANNEL_RE.search(text):
        text = LEADING_NON_FINAL_CHANNEL_RE.sub("", text, count=1)
    final_response = text.strip()
    return HarmonyExtraction(
        final_response=final_response,
        final_marker_found=final_marker_found,
        final_response_empty=not final_response,
        non_final_channel_in_final=bool(NON_FINAL_CHANNEL_RE.search(final_response)),
    )
