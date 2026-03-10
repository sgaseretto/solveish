"""LLM service package - Provider-based architecture for multi-LLM support.

Re-exports the main public API so consumers can do:
    from services.llm import LLMService, llm_service, SYSTEM_PROMPTS
"""
from .llm_service import LLMService, llm_service
from .constants import SYSTEM_PROMPTS

__all__ = ['LLMService', 'llm_service', 'SYSTEM_PROMPTS']
