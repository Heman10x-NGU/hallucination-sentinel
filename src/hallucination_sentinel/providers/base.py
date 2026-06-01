"""
Abstract base provider interface for logprob extraction.

All providers must implement this interface to integrate with
Hallucination Sentinel's entropy pipeline.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


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
    def score_text(
        self,
        text: str,
        *,
        top_k: Optional[int] = None,
    ) -> CompletionLogprobs:
        """Score arbitrary text using echo-based prompt scoring."""
        ...

    @abstractmethod
    def check_health(self) -> dict:
        """Quick health check: can we reach the API and get logprobs?"""
        ...
