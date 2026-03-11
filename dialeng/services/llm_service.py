"""Compatibility shim - imports from the new services.llm package.

All logic has moved to services/llm/. This file exists so that
existing imports like `from dialeng.services.llm_service import LLMService`
continue to work.
"""
from .llm import LLMService, llm_service, SYSTEM_PROMPTS

__all__ = ['LLMService', 'llm_service', 'SYSTEM_PROMPTS']
