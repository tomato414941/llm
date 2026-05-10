import pytest

from llm.leverage.harmony import extract_harmony_final_response


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("analysisThe answer is 42.assistantfinal42", "42"),
        ("analysis scratch assistant final The answer is 42.", "The answer is 42."),
        ("assistantfinal\n\n42", "42"),
        ("final {\"answer\":\"ok\"}", "{\"answer\":\"ok\"}"),
        ("analysis first assistantfinal wrong assistantfinal right", "right"),
        ("42", "42"),
    ],
)
def test_extract_harmony_final_response(raw: str, expected: str) -> None:
    assert extract_harmony_final_response(raw) == expected


def test_extract_harmony_final_response_accepts_pipeline_message_dict() -> None:
    raw = {
        "role": "assistant",
        "content": "analysisWe need 6 * 7.assistantfinal42",
    }

    assert extract_harmony_final_response(raw) == "42"
