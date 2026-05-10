from llm.leverage.gpt_oss_tiny_eval import build_messages, generated_message, summarize_extractions
from llm.leverage.harmony import HarmonyExtraction


def test_build_messages_uses_system_and_user_prompt() -> None:
    assert build_messages("Return ok.", "Reasoning: low") == [
        {"role": "system", "content": "Reasoning: low"},
        {"role": "user", "content": "Return ok."},
    ]


def test_generated_message_reads_transformers_pipeline_chat_output() -> None:
    output = [
        {
            "generated_text": [
                {"role": "system", "content": "Reasoning: low"},
                {"role": "user", "content": "Return ok."},
                {"role": "assistant", "content": "analysis short assistantfinalok"},
            ]
        }
    ]

    assert generated_message(output) == {"role": "assistant", "content": "analysis short assistantfinalok"}


def test_summarize_extractions_counts_harmony_contract_issues() -> None:
    extractions = [
        HarmonyExtraction(
            final_response="ok",
            final_marker_found=True,
            final_response_empty=False,
            non_final_channel_in_final=False,
        ),
        HarmonyExtraction(
            final_response="",
            final_marker_found=True,
            final_response_empty=True,
            non_final_channel_in_final=False,
        ),
        HarmonyExtraction(
            final_response="analysis leaked",
            final_marker_found=False,
            final_response_empty=False,
            non_final_channel_in_final=True,
        ),
    ]

    assert summarize_extractions(extractions) == {
        "task_count": 3,
        "missing_final_marker_count": 1,
        "empty_final_response_count": 1,
        "non_final_channel_in_final_count": 1,
    }
