"""Constants for OpenAI provider.

This module defines constants used across the OpenAI provider implementation,
following the principle of single source of truth.
"""

# Default configuration values
DEFAULT_MODEL = "gpt-5.2-codex"
DEFAULT_MAX_TOKENS = 64_000
DEFAULT_REASONING_EFFORT = "medium"  # minimal|low|medium|high|xhigh

# Debug logging configuration
DEFAULT_DEBUG_TRUNCATE_LENGTH = 2000  # Characters before truncation in debug logs

# Deep research model identifiers
DEEP_RESEARCH_MODELS = frozenset(
    {
        "o3-deep-research",
        "o3-deep-research-2025-06-26",
        "o4-mini-deep-research",
        "o4-mini-deep-research-2025-06-26",
    }
)
