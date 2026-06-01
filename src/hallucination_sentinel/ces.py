"""
Calibrated Entropy Score (CES) algorithm for hallucination risk assessment.

CES measures how extreme a generation's entropy profile is relative to a
calibrated reference distribution.  It is a ranking score, not a probability.

    CES = sqrt( F0(mean_entropy) * F0(max_entropy) )

where F0 is the empirical CDF built from reference (faithful) entropy values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from .calibration import CalibrationArtifact

# ---------------------------------------------------------------------------
# Warnings helper
# ---------------------------------------------------------------------------
SHORT_TOKEN_THRESHOLD = 10
LOW_POWER_TOKEN_THRESHOLD = 100


def _collect_warnings(
    token_count: int,
    entropy_mode: str,
    calibration_artifact: Optional[CalibrationArtifact],
) -> list[str]:
    """Emit contextual warnings based on input metadata."""
    warnings: list[str] = []
    if token_count < SHORT_TOKEN_THRESHOLD:
        warnings.append(
            f"short_text: only {token_count} tokens.  "
            "CES reliability degrades for very short generations."
        )
    elif token_count < LOW_POWER_TOKEN_THRESHOLD:
        warnings.append(
            f"low_power: {token_count} tokens.  "
            "Statistical power is reduced for short outputs."
        )
    if entropy_mode in ("top_k", "top_k_with_residual"):
        warnings.append(
            "top_k_entropy: entropy computed from top-k logprobs is an "
            "approximation of full-vocabulary entropy."
        )
    if calibration_artifact is not None:
        if calibration_artifact.entropy_mode and entropy_mode != calibration_artifact.entropy_mode:
            warnings.append(
                f"entropy_mode_mismatch: scoring uses '{entropy_mode}' but "
                f"calibration was built with '{calibration_artifact.entropy_mode}'."
            )
        if calibration_artifact.entropy_base and calibration_artifact.entropy_base != "e":
            warnings.append(
                f"entropy_base_mismatch: calibration uses base "
                f"'{calibration_artifact.entropy_base}'."
            )
    return warnings


# ---------------------------------------------------------------------------
# Risk level (simple mapping from CES quantile rank)
# ---------------------------------------------------------------------------

def _risk_level_from_ces(ces_score: float, thresholds: dict) -> str:
    """Map a CES score to a risk level string using stored quantile thresholds."""
    if not thresholds:
        # Fallback when thresholds have not been assigned yet
        return "UNKNOWN"
    low = thresholds.get("low", 0.75)
    medium = thresholds.get("medium", 0.90)
    high = thresholds.get("high", 0.97)
    if ces_score < low:
        return "LOW"
    elif ces_score < medium:
        return "MEDIUM"
    elif ces_score < high:
        return "HIGH"
    else:
        return "CRITICAL"


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class CESResult:
    """Result of a CES computation.

    Attributes:
        ces_score: The CES ranking score (NOT a probability).
        cdf_mean: F0(mean_entropy) -- quantile rank of the mean entropy.
        cdf_max: F0(max_entropy) -- quantile rank of the max entropy.
        mean_entropy: Arithmetic mean of per-token entropies.
        max_entropy: Maximum per-token entropy.
        token_count: Number of tokens in the scored sequence.
        risk_level: Risk band string (LOW / MEDIUM / HIGH / CRITICAL / UNKNOWN).
        warnings: List of diagnostic warning strings.
    """

    ces_score: float
    cdf_mean: float
    cdf_max: float
    mean_entropy: float
    max_entropy: float
    token_count: int
    risk_level: str
    warnings: list[str]


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_ces(
    entropy_sequence: np.ndarray,
    calibration_artifact: Optional[CalibrationArtifact] = None,
) -> CESResult:
    """Compute the Calibrated Entropy Score for a single generation.

    Args:
        entropy_sequence: 1-D array of per-token entropy values (nats by default).
        calibration_artifact: Calibration artifact providing the reference
            ECDF (F0).  If ``None``, the returned CDF values are 0 and the
            score is uncalibrated.

    Returns:
        A :class:`CESResult` with the CES score, component CDF values,
        descriptive statistics, risk level, and warnings.

    Raises:
        ValueError: If *entropy_sequence* is empty.
    """
    entropy_sequence = np.asarray(entropy_sequence, dtype=np.float64).ravel()
    if entropy_sequence.size == 0:
        raise ValueError("entropy_sequence must be non-empty")

    token_count = int(entropy_sequence.size)
    mean_entropy = float(np.mean(entropy_sequence))
    max_entropy = float(np.max(entropy_sequence))

    # Determine entropy mode from calibration artifact (best-effort)
    entropy_mode = "full"
    if calibration_artifact is not None:
        entropy_mode = calibration_artifact.entropy_mode or "full"

    warnings = _collect_warnings(token_count, entropy_mode, calibration_artifact)

    # Evaluate reference ECDF
    if calibration_artifact is not None and calibration_artifact.ecdf_values:
        cdf_mean = calibration_artifact.f0(mean_entropy)
        cdf_max = calibration_artifact.f0(max_entropy)
    else:
        cdf_mean = 0.0
        cdf_max = 0.0
        warnings.append(
            "no_calibration: no calibration artifact provided.  "
            "CES score is uncalibrated."
        )

    # CES = geometric mean of the two CDF quantiles
    ces_score = math.sqrt(cdf_mean * cdf_max)

    # Risk level from thresholds stored in artifact
    thresholds = {}
    if calibration_artifact is not None:
        thresholds = calibration_artifact.thresholds or {}
    risk_level = _risk_level_from_ces(ces_score, thresholds)

    return CESResult(
        ces_score=ces_score,
        cdf_mean=cdf_mean,
        cdf_max=cdf_max,
        mean_entropy=mean_entropy,
        max_entropy=max_entropy,
        token_count=token_count,
        risk_level=risk_level,
        warnings=warnings,
    )
