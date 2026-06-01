"""
Threshold policies for classifying calibrated risk scores into severity bands.

Maps calibrated probabilities to human-readable risk levels:
- LOW: likely faithful, safe to act on
- MEDIUM: uncertain, recommend verification
- HIGH: likely hallucinated, do not act without review
- CRITICAL: near-certain hallucination, block action
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class RiskLevel(str, Enum):
    """Risk severity bands."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class ThresholdPolicy:
    """Defines the boundaries between risk levels."""

    medium: float = 0.30
    high: float = 0.60
    critical: float = 0.85

    def classify(self, probability: float) -> RiskLevel:
        """
        Classify a calibrated probability into a RiskLevel.

        Args:
            probability: Calibrated hallucination probability in [0, 1].

        Returns:
            RiskLevel enum value.
        """
        if probability < self.medium:
            return RiskLevel.LOW
        elif probability < self.high:
            return RiskLevel.MEDIUM
        elif probability < self.critical:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL


def assign_thresholds(
    probabilities: list[float],
    policy: Optional[ThresholdPolicy] = None,
) -> list[RiskLevel]:
    """
    Assign risk levels to a list of calibrated probabilities.

    Args:
        probabilities: List of calibrated hallucination probabilities.
        policy: Threshold policy to use. Defaults to ThresholdPolicy().

    Returns:
        List of RiskLevel values, one per input probability.
    """
    if policy is None:
        policy = ThresholdPolicy()
    return [policy.classify(p) for p in probabilities]
