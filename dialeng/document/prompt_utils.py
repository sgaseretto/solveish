"""Prompt cell separator utilities for splitting/joining user prompts and AI responses."""
import re
import uuid
from typing import Pattern, Tuple


SEPARATOR_PREFIX = "##### \U0001f916Reply\U0001f916<!-- SOLVEIT_SEPARATOR_"
SEPARATOR_SUFFIX = " -->"
SEPARATOR_PATTERN = re.compile(r'##### \U0001f916Reply\U0001f916<!-- SOLVEIT_SEPARATOR_([a-f0-9]+) -->')

# Legacy prompt separator used by older load/save paths. Keep parsing support so
# notebooks saved before the prompt serialization cleanup still round-trip.
LEGACY_SEPARATOR_PREFIX = "##### Reply<!-- SOLVEIT_SEPARATOR_"
LEGACY_SEPARATOR_PATTERN = re.compile(r'##### Reply<!-- SOLVEIT_SEPARATOR_([a-f0-9]+) -->')

SUPPORTED_SEPARATOR_PATTERNS: tuple[Pattern[str], ...] = (
    SEPARATOR_PATTERN,
    LEGACY_SEPARATOR_PATTERN,
)


def make_separator() -> str:
    """Generate a new separator with random ID"""
    sep_id = uuid.uuid4().hex[:8]
    return f"{SEPARATOR_PREFIX}{sep_id}{SEPARATOR_SUFFIX}"


def split_prompt_content(content: str) -> Tuple[str, str]:
    """Split prompt cell content into (user_prompt, ai_response).

    Accepts both the current prompt separator and the legacy separator so
    notebooks saved by older Dialeng versions still load correctly.
    """
    for pattern in SUPPORTED_SEPARATOR_PATTERNS:
        match = pattern.search(content)
        if match:
            user_prompt = content[:match.start()].strip()
            ai_response = content[match.end():].strip()
            return user_prompt, ai_response
    return content.strip(), ""


def join_prompt_content(user_prompt: str, ai_response: str) -> str:
    """Join user prompt and AI response with separator"""
    if not ai_response:
        return user_prompt
    return f"{user_prompt}\n\n{make_separator()}\n\n{ai_response}"
