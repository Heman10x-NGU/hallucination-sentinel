"""
Data schemas for Hallucination Sentinel.

Defines the structured request/response types used across the package.
All types are frozen dataclasses for immutability.
"""

from dataclasses import dataclass, field
from typing import Optional

from .thresholds import RiskLevel


@dataclass(frozen=True)
class TokenScore:
    """Per-token risk assessment."""

    index: int
    token: str
    entropy: float
    is_flagged: bool


@dataclass(frozen=True)
class SentinelResult:
    """Complete result of a hallucination assessment."""

    ces_score: float
    calibrated_probability: float
    risk_level: RiskLevel
    token_count: int
    token_entropies: tuple[float, ...]
    flagged_tokens: tuple[TokenScore, ...] = ()
    warnings: tuple[str, ...] = ()
    provider: Optional[str] = None
    model: Optional[str] = None

    @property
    def is_hallucination(self) -> bool:
        """True if risk level is HIGH or CRITICAL."""
        return self.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)

    @property
    def should_block(self) -> bool:
        """True if risk level is CRITICAL (action should be blocked)."""
        return self.risk_level == RiskLevel.CRITICAL
