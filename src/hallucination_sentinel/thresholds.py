"""
Threshold policies for classifying CES scores into risk bands.

Two policies are supported:

1. **REFERENCE_QUANTILE** (default, works with or without labels):
   Uses quantiles of the reference ECDF to define boundaries.

2. **MAX_TPR_AT_FPR** (supervised only):
   Finds the threshold that maximises true-positive rate subject to a
   maximum false-positive-rate constraint, using Youden's J statistic.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Policy enum
# ---------------------------------------------------------------------------

class ThresholdPolicy(str, Enum):
    """Supported threshold-selection policies."""

    REFERENCE_QUANTILE = "reference_quantile"
    MAX_TPR_AT_FPR = "max_tpr_at_fpr"


# ---------------------------------------------------------------------------
# Risk level (re-exported for convenience)
# ---------------------------------------------------------------------------

class RiskLevel(str, Enum):
    """Risk severity bands."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Quantile-based thresholds (default)
# ---------------------------------------------------------------------------

def _quantile_thresholds(
    ecdf_values: list[float],
    low_q: float = 75.0,
    medium_q: float = 90.0,
    high_q: float = 97.0,
) -> dict[str, float]:
    """Compute quantile boundaries from the reference ECDF values.

    Default bands:
        LOW     : CES < p75
        MEDIUM  : p75 <= CES < p90
        HIGH    : p90 <= CES < p97
        CRITICAL: CES >= p97
    """
    if not ecdf_values:
        return {"low": 0.75, "medium": 0.90, "high": 0.97}
    arr = np.asarray(ecdf_values, dtype=np.float64)
    return {
        "low": float(np.percentile(arr, low_q)),
        "medium": float(np.percentile(arr, medium_q)),
        "high": float(np.percentile(arr, high_q)),
    }


# ---------------------------------------------------------------------------
# Supervised thresholds (max-TPR-at-FPR / Youden's J)
# ---------------------------------------------------------------------------

def _youden_thresholds(
    ces_scores_faithful: np.ndarray,
    ces_scores_hallucinated: np.ndarray,
    max_fpr: float = 0.05,
) -> dict[str, float]:
    """Find thresholds that maximise Youden's J (TPR - FPR).

    The *critical* threshold targets the requested FPR bound.
    *high* and *medium* are set at the midpoints between the critical
    threshold and the median faithful score.
    """
    faithful = np.asarray(ces_scores_faithful, dtype=np.float64)
    halluc = np.asarray(ces_scores_hallucinated, dtype=np.float64)

    if faithful.size == 0 or halluc.size == 0:
        return {"low": 0.75, "medium": 0.90, "high": 0.97}

    # Candidate thresholds: union of all unique scores
    candidates = np.unique(np.concatenate([faithful, halluc]))
    best_j = -1.0
    best_t = candidates[0]

    for t in candidates:
        tpr = float(np.mean(halluc >= t))
        fpr = float(np.mean(faithful >= t))
        if fpr > max_fpr:
            continue
        j = tpr - fpr
        if j > best_j:
            best_j = j
            best_t = t

    critical = float(best_t)
    median_faithful = float(np.median(faithful))

    # Spread high and medium between median faithful and critical
    high = median_faithful + 0.667 * (critical - median_faithful)
    medium = median_faithful + 0.333 * (critical - median_faithful)

    return {
        "low": round(medium, 6),
        "medium": round(high, 6),
        "high": round(critical, 6),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def assign_thresholds(
    calibration_artifact,
    policy: str | ThresholdPolicy = ThresholdPolicy.REFERENCE_QUANTILE,
    *,
    low_q: float = 75.0,
    medium_q: float = 90.0,
    high_q: float = 97.0,
    ces_scores_faithful: Optional[np.ndarray] = None,
    ces_scores_hallucinated: Optional[np.ndarray] = None,
    max_fpr: float = 0.05,
) -> dict[str, float]:
    """Compute and store risk-level thresholds in the calibration artifact.

    Args:
        calibration_artifact: A :class:`CalibrationArtifact` whose
            ``ecdf_values`` (and optionally ``thresholds``) will be updated
            **in place**.
        policy: ``"reference_quantile"`` (default) or ``"max_tpr_at_fpr"``.
        low_q, medium_q, high_q: Percentile positions for the quantile
            policy (0-100 scale).
        ces_scores_faithful: CES scores for faithful examples (supervised).
        ces_scores_hallucinated: CES scores for hallucinated examples
            (supervised).
        max_fpr: Maximum false-positive rate for the supervised policy.

    Returns:
        The thresholds dict that was written into the artifact.

    Raises:
        ValueError: If the supervised policy is requested but CES scores
            are not provided.
    """
    policy = ThresholdPolicy(policy)

    if policy is ThresholdPolicy.REFERENCE_QUANTILE:
        thresholds = _quantile_thresholds(
            calibration_artifact.ecdf_values, low_q, medium_q, high_q
        )
    elif policy is ThresholdPolicy.MAX_TPR_AT_FPR:
        if ces_scores_faithful is None or ces_scores_hallucinated is None:
            raise ValueError(
                "MAX_TPR_AT_FPR policy requires both ces_scores_faithful "
                "and ces_scores_hallucinated."
            )
        thresholds = _youden_thresholds(
            ces_scores_faithful, ces_scores_hallucinated, max_fpr
        )
    else:
        raise ValueError(f"Unknown threshold policy: {policy}")

    calibration_artifact.thresholds = thresholds
    return thresholds
