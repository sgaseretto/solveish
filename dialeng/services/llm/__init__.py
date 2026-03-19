"""LLM service package - Provider-based architecture for multi-LLM support.

Re-exports the main public API so consumers can do:
    from dialeng.services.llm import LLMService, llm_service, SYSTEM_PROMPTS
"""
from .llm_service import LLMService, llm_service
from .constants import SYSTEM_PROMPTS

# Import provider implementations so they self-register with the registry
from .providers import claudette_provider, claude_agent_sdk_provider  # noqa: F401

__all__ = ['LLMService', 'llm_service', 'SYSTEM_PROMPTS']
