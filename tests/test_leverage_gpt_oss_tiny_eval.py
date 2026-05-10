from llm.leverage.gpt_oss_tiny_eval import build_messages, generated_message


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
