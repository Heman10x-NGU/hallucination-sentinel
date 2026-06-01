"""
Abstract base provider interface for logprob extraction.

All providers must implement this interface to integrate with
Hallucination Sentinel's entropy pipeline.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ProviderCapabilityError(Exception):
    """Raised when a provider lacks the capability for a requested operation.

    Attributes:
        capability: The missing capability (e.g. "top_k_logprobs", "logprobs").
        provider: The provider name that raised this error.
    """

    def __init__(self, message: str, *, capability: str = "", provider: str = ""):
        self.capability = capability
        self.provider = provider
        super().__init__(message)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderConfig:
    """Configuration for an LLM provider.

    All fields fall back to environment variables when not set explicitly.
    No API keys are ever hardcoded; they must come from args or env.
    """

    api_key: str
    base_url: str
    model: str

    @classmethod
    def from_env(
        cls,
        *,
        api_key_env: str = "OPENAI_API_KEY",
        base_url_env: str = "OPENAI_BASE_URL",
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> ProviderConfig:
        """Build config from environment variables with optional overrides.

        Args:
            api_key_env: Name of the env var for the API key.
            base_url_env: Name of the env var for the base URL.
            model: Model name (no env var default).
            api_key: Explicit API key (overrides env var).
            base_url: Explicit base URL (overrides env var).

        Returns:
            A frozen ProviderConfig.

        Raises:
            ValueError: If api_key is not provided and the env var is unset/empty.
        """
        resolved_key = api_key or os.environ.get(api_key_env, "")
        if not resolved_key:
            raise ValueError(
                f"API key not provided. Set {api_key_env} or pass api_key explicitly."
            )
        resolved_url = base_url or os.environ.get(base_url_env, "https://api.openai.com/v1")
        return cls(api_key=resolved_key, base_url=resolved_url, model=model)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class TokenLogprob:
    """Normalized logprob data for a single token position."""

    token: str
    logprob: float
    token_id: Optional[int] = None
    top_logprobs: dict[str, float] = field(default_factory=dict)


@dataclass
class CompletionLogprobs:
    """Normalized logprob data for an entire completion."""

    tokens: list[TokenLogprob]
    provider: str
    model: str
    top_k: int
    echo_used: bool = False

    @property
    def selected_logprobs(self) -> list[float]:
        """Logprobs of the tokens the model actually chose."""
        return [t.logprob for t in self.tokens]

    @property
    def topk_logprobs(self) -> list[dict[str, float]]:
        """Top-k logprob dicts per position (includes selected token)."""
        return [t.top_logprobs for t in self.tokens]

    def has_top_k(self) -> bool:
        return self.top_k > 0 and any(t.top_logprobs for t in self.tokens)


@dataclass
class TokenLogprobResult:
    """Result of score_generation(): token logprobs with computed entropy.

    Attributes:
        token_logprobs: Per-token logprob data from the provider.
        entropies: Per-token entropy values (nats). None if top-k unavailable.
        perplexity: Per-token perplexity. None if no logprobs at all.
        provider: Provider name.
        model: Model name.
        top_k: Number of top-k alternatives available per position (0 if none).
        warnings: Diagnostic messages about data quality or limitations.
    """

    token_logprobs: list[TokenLogprob]
    entropies: Optional[np.ndarray]
    perplexity: Optional[float]
    provider: str
    model: str
    top_k: int
    warnings: list[str] = field(default_factory=list)

    def has_top_k(self) -> bool:
        """True if top-k alternatives are available (enables CES)."""
        return self.top_k > 0 and self.entropies is not None

    def has_selected_only(self) -> bool:
        """True if only selected-token logprobs are available (no top-k)."""
        return self.top_k == 0 and self.perplexity is not None


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseProvider(ABC):
    """Abstract base class for all LLM logprob providers."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 100,
        top_k: Optional[int] = None,
        echo: bool = False,
    ) -> CompletionLogprobs:
        """Generate text and return normalized logprob data."""
        ...

    @abstractmethod
    def generate_with_logprobs(
        self,
        prompt: str,
        *,
        max_tokens: int = 100,
        top_k: Optional[int] = None,
    ) -> CompletionLogprobs:
        """Generate text from prompt and return logprobs for generated tokens.

        This is a focused variant of generate() without echo/teacher-forcing.
        Use score_output() to score a specific output against a prompt.
        """
        ...

    @abstractmethod
    def score_output(
        self,
        prompt: str,
        output: str,
        *,
        top_k: Optional[int] = None,
    ) -> CompletionLogprobs:
        """Score an output given its prompt context.

        Uses echo/teacher-forcing: the provider receives the full
        (prompt, output) pair and returns logprobs for the output tokens
        conditioned on the prompt.

        Raises:
            ProviderCapabilityError: If the provider lacks echo support.
        """
        ...

    @abstractmethod
    def score_text(
        self,
        text: str,
        *,
        top_k: Optional[int] = None,
    ) -> CompletionLogprobs:
        """Score arbitrary text using echo-based prompt scoring.

        No prompt context is used -- the text is scored in isolation.
        """
        ...

    @abstractmethod
    def check_health(self) -> dict:
        """Quick health check: can we reach the API and get logprobs?"""
        ...
