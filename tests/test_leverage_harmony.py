import pytest

from llm.leverage.harmony import analyze_harmony_response, extract_harmony_final_response


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


def test_analyze_harmony_response_reports_missing_final_marker() -> None:
    extraction = analyze_harmony_response("analysis 42")

    assert extraction.final_response == "42"
    assert extraction.final_marker_found is False
    assert extraction.final_response_empty is False
    assert extraction.non_final_channel_in_final is False


def test_analyze_harmony_response_reports_empty_final_response() -> None:
    extraction = analyze_harmony_response("analysis scratch assistantfinal")

    assert extraction.final_response == ""
    assert extraction.final_marker_found is True
    assert extraction.final_response_empty is True


def test_analyze_harmony_response_reports_non_final_channel_leakage() -> None:
    extraction = analyze_harmony_response("analysis scratch assistantfinalanalysis leaked")

    assert extraction.final_response == "analysis leaked"
    assert extraction.final_marker_found is True
    assert extraction.non_final_channel_in_final is True
