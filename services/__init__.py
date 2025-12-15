"""Services layer - Business logic for kernel, LLM, and storage."""

from .dialoghelper_service import (
    get_msg_idx,
    find_msgs,
    read_msg,
    get_cells_before,
    cell_to_dict,
    build_context_messages,
    cell_to_messages,
    MAX_CONTEXT_CELLS,
)

from .llm_service import LLMService, llm_service, SYSTEM_PROMPTS

__all__ = [
    # dialoghelper_service
    "get_msg_idx",
    "find_msgs",
    "read_msg",
    "get_cells_before",
    "cell_to_dict",
    "build_context_messages",
    "cell_to_messages",
    "MAX_CONTEXT_CELLS",
    # llm_service
    "LLMService",
    "llm_service",
    "SYSTEM_PROMPTS",
]
