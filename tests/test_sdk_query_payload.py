"""Tests for Claude Agent SDK prompt payload construction."""

from dialeng.services.llm.utils import build_sdk_query_payload


def test_build_sdk_query_payload_keeps_notebook_images_as_blocks():
    image_block = {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": "abc123",
        },
    }
    context_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Captured screenshot for analysis."},
                image_block,
            ],
        },
        {
            "role": "assistant",
            "content": "I can inspect the screenshot if you rerun the prompt.",
        },
    ]

    system_prompt, prompt_payload = build_sdk_query_payload(
        "Describe what you see in the screenshot.",
        context_messages,
        "You are a helpful assistant.",
    )

    assert "Authoritative current notebook context:" in system_prompt
    assert "User: Captured screenshot for analysis. [Notebook image 1]" in system_prompt
    assert "Assistant: I can inspect the screenshot if you rerun the prompt." in system_prompt
    assert isinstance(prompt_payload, list)
    assert prompt_payload[0]["type"] == "image"
    assert prompt_payload[0]["source"]["data"] == "abc123"
    assert prompt_payload[-1]["type"] == "text"
    assert "[Notebook image 1]" in prompt_payload[-1]["text"]
    assert "Describe what you see in the screenshot." in prompt_payload[-1]["text"]


def test_build_sdk_query_payload_falls_back_to_plain_text_without_images():
    context_messages = [
        {"role": "user", "content": "My name is Joe Doe"},
        {"role": "assistant", "content": "Nice to meet you Joe Doe."},
    ]

    system_prompt, prompt_payload = build_sdk_query_payload(
        "What is my name?",
        context_messages,
        "You are a helpful assistant.",
    )

    assert "User: My name is Joe Doe" in system_prompt
    assert "Assistant: Nice to meet you Joe Doe." in system_prompt
    assert prompt_payload == "What is my name?"
