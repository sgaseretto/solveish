"""Prompt cell separator utilities for splitting/joining user prompts and AI responses."""
import re
import uuid


SEPARATOR_PREFIX = "##### \U0001f916Reply\U0001f916<!-- SOLVEIT_SEPARATOR_"
SEPARATOR_SUFFIX = " -->"
SEPARATOR_PATTERN = re.compile(r'##### \U0001f916Reply\U0001f916<!-- SOLVEIT_SEPARATOR_([a-f0-9]+) -->')


def make_separator() -> str:
    """Generate a new separator with random ID"""
    sep_id = uuid.uuid4().hex[:8]
    return f"{SEPARATOR_PREFIX}{sep_id}{SEPARATOR_SUFFIX}"


def split_prompt_content(content: str) -> tuple[str, str]:
    """Split prompt cell content into (user_prompt, ai_response)"""
    match = SEPARATOR_PATTERN.search(content)
    if match:
        user_prompt = content[:match.start()].strip()
        ai_response = content[match.end():].strip()
        return user_prompt, ai_response
    else:
        return content.strip(), ""


def join_prompt_content(user_prompt: str, ai_response: str) -> str:
    """Join user prompt and AI response with separator"""
    if not ai_response:
        return user_prompt
    return f"{user_prompt}\n\n{make_separator()}\n\n{ai_response}"
