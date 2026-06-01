"""
Provider integrations for hallucination-sentinel.

Each provider extracts token logprob data from API responses and normalizes
it into the format expected by entropy.py.
"""

from .base import (
    BaseProvider,
    CompletionLogprobs,
    ProviderCapabilityError,
    ProviderConfig,
    TokenLogprob,
    TokenLogprobResult,
)
from .openai_compatible import (
    OpenAICompatibleProvider,
    PROVIDER_SPECS,
    ProviderSpec,
    TokenLogprobProvider,
    smoke_provider,
)

__all__ = [
    "BaseProvider",
    "CompletionLogprobs",
    "OpenAICompatibleProvider",
    "PROVIDER_SPECS",
    "ProviderCapabilityError",
    "ProviderConfig",
    "ProviderSpec",
    "TokenLogprob",
    "TokenLogprobProvider",
    "TokenLogprobResult",
    "smoke_provider",
]
