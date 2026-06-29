from claimbench.model import format_user_prompt


class ThinkingTokenizer:
    chat_template = "{{ messages }}"

    def apply_chat_template(self, conversation, *, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize is False
        assert add_generation_prompt is True
        return f"{conversation[0]['content']}|thinking={enable_thinking}"


class PlainTokenizer:
    chat_template = "{{ messages }}"

    def apply_chat_template(self, conversation, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        return conversation[0]["content"]


class SystemTokenizer:
    chat_template = "{{ messages }}"

    def apply_chat_template(self, conversation, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        return "|".join(f"{item['role']}={item['content']}" for item in conversation)


def test_format_user_prompt_passes_thinking_flag_when_supported():
    rendered = format_user_prompt(
        ThinkingTokenizer(),
        "solve",
        use_chat_template=True,
        chat_enable_thinking=False,
    )

    assert rendered == "solve|thinking=False"


def test_format_user_prompt_falls_back_for_non_qwen_templates():
    rendered = format_user_prompt(
        PlainTokenizer(),
        "solve",
        use_chat_template=True,
        chat_enable_thinking=True,
    )

    assert rendered == "solve"


def test_format_user_prompt_can_include_system_message():
    rendered = format_user_prompt(
        SystemTokenizer(),
        "question",
        use_chat_template=True,
        system_prompt="system instruction",
    )

    assert rendered == "system=system instruction|user=question"
