"""Prompt builders for OCR/LLM geometry analysis."""

from .base import BASE_PROMPT_TEMPLATE
from .problem_rules import constraints_for, example_for, prompt_context_for, rules_for

__all__ = [
    "BASE_PROMPT_TEMPLATE",
    "constraints_for",
    "example_for",
    "prompt_context_for",
    "rules_for",
]
