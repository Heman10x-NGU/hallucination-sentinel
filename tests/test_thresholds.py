"""Tests for the thresholds module.

Covers:
- Threshold assignment correctness (quantile policy)
- Supervised threshold policy (Youden's J / max-TPR-at-FPR)
- RiskLevel enum
- Edge cases
"""

import numpy as np
import pytest

from hallucination_sentinel.calibration import CalibrationArtifact, build_calibration
from hallucination_sentinel.thresholds import (
    RiskLevel,
    ThresholdPolicy,
    assign_thresholds,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_artifact(n_seq: int = 20, n_tokens: int = 50, seed: int = 42):
    """Build a deterministic calibration artifact."""
    rng = np.random.RandomState(seed)
    seqs = [rng.uniform(0.5, 3.0, size=n_tokens) for _ in range(n_seq)]
    return build_calibration(seqs, mode="unsupervised")


# ---------------------------------------------------------------------------
# Quantile policy (default)
# ---------------------------------------------------------------------------

class TestReferenceQuantilePolicy:
    """Default quantile-based threshold assignment."""

    def test_default_quantiles(self):
        """LOW < p75, MEDIUM p75-p90, HIGH p90-p97, CRITICAL >= p97."""
        art = _make_artifact()
        thresholds = assign_thresholds(art, ThresholdPolicy.REFERENCE_QUANTILE)
        assert "low" in thresholds
        assert "medium" in thresholds
        assert "high" in thresholds
        assert thresholds["low"] < thresholds["medium"] < thresholds["high"]

    def test_thresholds_stored_in_artifact(self):
        """assign_thresholds writes thresholds into the artifact."""
        art = _make_artifact()
        assert art.thresholds == {}
        assign_thresholds(art)
        assert art.thresholds != {}
        assert "low" in art.thresholds

    def test_custom_quantiles(self):
        """Changing quantile positions shifts thresholds."""
        art = _make_artifact()
        t1 = assign_thresholds(art, low_q=50, medium_q=75, high_q=95)
        art2 = _make_artifact()
        t2 = assign_thresholds(art2, low_q=80, medium_q=90, high_q=99)
        # Higher quantiles should produce higher thresholds
        assert t2["low"] > t1["low"]
        assert t2["medium"] > t1["medium"]

    def test_thresholds_from_ecdf_values(self):
        """Thresholds must be values from the ECDF range."""
        rng = np.random.RandomState(123)
        seqs = [rng.uniform(0.0, 5.0, size=100) for _ in range(10)]
        art = build_calibration(seqs)
        thresholds = assign_thresholds(art)
        ecdf_min = min(art.ecdf_values)
        ecdf_max = max(art.ecdf_values)
        for v in thresholds.values():
            assert ecdf_min <= v <= ecdf_max

    def test_string_policy_value(self):
        """Policy can be passed as a string."""
        art = _make_artifact()
        thresholds = assign_thresholds(art, "reference_quantile")
        assert "low" in thresholds


# ---------------------------------------------------------------------------
# Supervised policy (max-TPR-at-FPR)
# ---------------------------------------------------------------------------

class TestMaxTPRAtFPRPolicy:
    """Supervised threshold selection via Youden's J."""

    def test_basic_supervised(self):
        """With separated distributions, should find a reasonable threshold."""
        art = _make_artifact()
        # Faithful CES scores: low values
        faithful = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        # Hallucinated CES scores: high values
        halluc = np.array([0.6, 0.7, 0.8, 0.9, 1.0])
        thresholds = assign_thresholds(
            art,
            ThresholdPolicy.MAX_TPR_AT_FPR,
            ces_scores_faithful=faithful,
            ces_scores_hallucinated=halluc,
            max_fpr=0.05,
        )
        assert "low" in thresholds
        assert "medium" in thresholds
        assert "high" in thresholds
        assert thresholds["low"] < thresholds["medium"] < thresholds["high"]

    def test_threshold_above_faithful_scores(self):
        """Critical threshold should be above most faithful scores."""
        art = _make_artifact()
        faithful = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        halluc = np.array([0.7, 0.8, 0.9, 1.0, 1.1])
        thresholds = assign_thresholds(
            art,
            ThresholdPolicy.MAX_TPR_AT_FPR,
            ces_scores_faithful=faithful,
            ces_scores_hallucinated=halluc,
            max_fpr=0.05,
        )
        # The high (critical) threshold should be above the median faithful
        assert thresholds["high"] > np.median(faithful)

    def test_missing_scores_raises(self):
        """Supervised policy without scores must raise."""
        art = _make_artifact()
        with pytest.raises(ValueError, match="requires both"):
            assign_thresholds(art, ThresholdPolicy.MAX_TPR_AT_FPR)

    def test_missing_hallucinated_raises(self):
        art = _make_artifact()
        with pytest.raises(ValueError, match="requires both"):
            assign_thresholds(
                art,
                ThresholdPolicy.MAX_TPR_AT_FPR,
                ces_scores_faithful=np.array([0.1, 0.2]),
            )

    def test_empty_arrays_fallback(self):
        """Empty score arrays should fall back to default thresholds."""
        art = _make_artifact()
        thresholds = assign_thresholds(
            art,
            ThresholdPolicy.MAX_TPR_AT_FPR,
            ces_scores_faithful=np.array([]),
            ces_scores_hallucinated=np.array([]),
        )
        assert thresholds == {"low": 0.75, "medium": 0.90, "high": 0.97}

    def test_string_policy_value(self):
        """Policy can be passed as a string."""
        art = _make_artifact()
        thresholds = assign_thresholds(
            art,
            "max_tpr_at_fpr",
            ces_scores_faithful=np.array([0.1, 0.2]),
            ces_scores_hallucinated=np.array([0.8, 0.9]),
        )
        assert "high" in thresholds


# ---------------------------------------------------------------------------
# RiskLevel enum
# ---------------------------------------------------------------------------

class TestRiskLevel:

    def test_values(self):
        assert RiskLevel.LOW == "low"
        assert RiskLevel.MEDIUM == "medium"
        assert RiskLevel.HIGH == "high"
        assert RiskLevel.CRITICAL == "critical"

    def test_is_string(self):
        assert isinstance(RiskLevel.LOW, str)


# ---------------------------------------------------------------------------
# ThresholdPolicy enum
# ---------------------------------------------------------------------------

class TestThresholdPolicyEnum:

    def test_values(self):
        assert ThresholdPolicy.REFERENCE_QUANTILE == "reference_quantile"
        assert ThresholdPolicy.MAX_TPR_AT_FPR == "max_tpr_at_fpr"

    def test_invalid_policy_raises(self):
        art = _make_artifact()
        with pytest.raises(ValueError):
            assign_thresholds(art, "nonexistent_policy")
