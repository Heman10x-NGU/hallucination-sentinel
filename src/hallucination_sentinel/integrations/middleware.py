"""
Middleware for wrapping LLM calls with hallucination checks.

Provides callable wrappers that intercept LLM outputs, run the
sentinel pipeline, and optionally block or annotate results.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional

from ..schemas import SentinelResult


@dataclass(frozen=True)
class SentinelMiddleware:
    """
    Wraps an LLM callable to automatically check outputs for hallucinations.

    Usage:
        middleware = SentinelMiddleware(
            llm_call=my_llm_function,
            provider="together",
            block_on_critical=True,
        )
        result = middleware("What is the capital of France?")
    """

    llm_call: Callable[..., str]
    provider: str = "together"
    block_on_critical: bool = True
    on_warning: Optional[Callable[[SentinelResult], None]] = None

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        """
        Call the LLM and check the result for hallucinations.

        Args:
            prompt: The prompt to send to the LLM.
            **kwargs: Additional arguments passed to the LLM call.

        Returns:
            The LLM response text (unchanged if not blocked).

        Raises:
            HallucinationBlockedError: If block_on_critical=True and risk is CRITICAL.
        """
        response = self.llm_call(prompt, **kwargs)

        # TODO: Run the full sentinel pipeline on the response
        # For now, pass through without checking
        return response


class HallucinationBlockedError(Exception):
    """Raised when a generation is blocked due to critical hallucination risk."""

    def __init__(self, result: SentinelResult):
        self.result = result
        super().__init__(
            f"Hallucination blocked: risk={result.risk_level.value}, "
            f"probability={result.calibrated_probability:.2f}"
        )
