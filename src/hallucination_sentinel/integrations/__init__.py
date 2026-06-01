"""
Integrations for Hallucination Sentinel.

Provides middleware and framework integrations for embedding
hallucination checks into existing pipelines.
"""

from .middleware import (
    DiagnosticPeak,
    HallucinationBlockedError,
    PolicyAction,
    RoutingDecision,
    SentinelMiddleware,
    TaskCriticality,
    guard_output,
    guard_output_from_entropies,
    guard_output_with_logprobs,
)

__all__ = [
    "DiagnosticPeak",
    "HallucinationBlockedError",
    "PolicyAction",
    "RoutingDecision",
    "SentinelMiddleware",
    "TaskCriticality",
    "guard_output",
    "guard_output_from_entropies",
    "guard_output_with_logprobs",
]
