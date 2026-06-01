"""Tests for the CES (Calibrated Entropy Score) algorithm.

Covers:
- CES determinism on fixture data
- CDF component values (cdf_mean, cdf_max)
- Geometric-mean formula verification
- Warning generation
- Edge cases (empty, single token, no calibration)
"""

import math

import numpy as np
import pytest

from hallucination_sentinel.calibration import CalibrationArtifact, build_calibration
from hallucination_sentinel.ces import CESResult, compute_ces


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_artifact(seed: int = 42, n_seq: int = 20, n_tokens: int = 50):
    """Build a deterministic calibration artifact for testing."""
    rng = np.random.RandomState(seed)
    seqs = [rng.uniform(0.5, 3.0, size=n_tokens) for _ in range(n_seq)]
    return build_calibration(seqs, mode="unsupervised")


def _deterministic_entropy(seed: int = 99, n_tokens: int = 30):
    """Return a deterministic entropy sequence."""
    rng = np.random.RandomState(seed)
    return rng.uniform(0.5, 3.0, size=n_tokens)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestCESDeterminism:
    """CES must be deterministic on the same input."""

    def test_same_input_same_score(self):
        art = _make_artifact()
        seq = _deterministic_entropy()
        r1 = compute_ces(seq, art)
        r2 = compute_ces(seq, art)
        assert r1.ces_score == r2.ces_score
        assert r1.cdf_mean == r2.cdf_mean
        assert r1.cdf_max == r2.cdf_max
        assert r1.mean_entropy == r2.mean_entropy
        assert r1.max_entropy == r2.max_entropy

    def test_deterministic_across_calls(self):
        """Multiple calls with identical inputs produce bit-identical results."""
        art = _make_artifact()
        seq = _deterministic_entropy(seed=7, n_tokens=40)
        scores = [compute_ces(seq, art).ces_score for _ in range(10)]
        assert all(s == scores[0] for s in scores)


# ---------------------------------------------------------------------------
# Geometric mean formula
# ---------------------------------------------------------------------------

class TestCESFormula:
    """CES = sqrt(F0(mean_entropy) * F0(max_entropy))."""

    def test_geometric_mean(self):
        art = _make_artifact()
        seq = _deterministic_entropy()
        result = compute_ces(seq, art)
        expected = math.sqrt(result.cdf_mean * result.cdf_max)
        assert abs(result.ces_score - expected) < 1e-15

    def test_components_are_cdf_values(self):
        """cdf_mean and cdf_max must be valid CDF values in [0, 1]."""
        art = _make_artifact()
        seq = _deterministic_entropy()
        result = compute_ces(seq, art)
        assert 0.0 <= result.cdf_mean <= 1.0
        assert 0.0 <= result.cdf_max <= 1.0

    def test_cdf_max_ge_cdf_mean(self):
        """max entropy >= mean entropy, so F0(max) >= F0(mean) (monotonicity)."""
        art = _make_artifact()
        seq = _deterministic_entropy()
        result = compute_ces(seq, art)
        assert result.cdf_max >= result.cdf_mean

    def test_uniform_entropy_low_score(self):
        """Constant entropy should yield a low CES score close to F0(constant)."""
        art = _make_artifact()
        # All tokens have the same entropy
        seq = np.ones(50) * 1.5
        result = compute_ces(seq, art)
        # mean == max == 1.5, so cdf_mean == cdf_max, ces == cdf_mean
        assert result.cdf_mean == result.cdf_max
        assert abs(result.ces_score - result.cdf_mean) < 1e-15

    def test_high_entropy_produces_high_ces(self):
        """A sequence with very high entropy should have a high CES score."""
        art = _make_artifact()
        seq = np.ones(30) * 10.0  # way above the reference distribution
        result = compute_ces(seq, art)
        assert result.ces_score > 0.9

    def test_low_entropy_produces_low_ces(self):
        """A sequence with very low entropy should have a low CES score."""
        art = _make_artifact()
        seq = np.ones(30) * 0.01  # way below the reference distribution
        result = compute_ces(seq, art)
        assert result.ces_score < 0.1


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------

class TestCESStatistics:
    """mean_entropy, max_entropy, token_count must be correct."""

    def test_mean_entropy(self):
        art = _make_artifact()
        seq = np.array([1.0, 2.0, 3.0])
        result = compute_ces(seq, art)
        assert result.mean_entropy == pytest.approx(2.0)

    def test_max_entropy(self):
        art = _make_artifact()
        seq = np.array([1.0, 2.0, 3.0])
        result = compute_ces(seq, art)
        assert result.max_entropy == pytest.approx(3.0)

    def test_token_count(self):
        art = _make_artifact()
        seq = np.array([1.0, 2.0, 3.0])
        result = compute_ces(seq, art)
        assert result.token_count == 3


# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------

class TestCESWarnings:
    """CES must emit appropriate warnings."""

    def test_short_text_warning(self):
        art = _make_artifact()
        seq = np.array([1.0, 2.0])  # < 10 tokens
        result = compute_ces(seq, art)
        assert any("short_text" in w for w in result.warnings)

    def test_low_power_warning(self):
        art = _make_artifact()
        seq = np.ones(20)  # 10 <= n < 100
        result = compute_ces(seq, art)
        assert any("low_power" in w for w in result.warnings)

    def test_top_k_warning(self):
        art = _make_artifact()
        art.entropy_mode = "top_k_with_residual"
        seq = np.ones(50)
        result = compute_ces(seq, art)
        assert any("top_k_entropy" in w for w in result.warnings)

    def test_no_calibration_warning(self):
        seq = np.ones(30)
        result = compute_ces(seq, calibration_artifact=None)
        assert any("no_calibration" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestCESEdgeCases:

    def test_empty_sequence_raises(self):
        art = _make_artifact()
        with pytest.raises(ValueError, match="non-empty"):
            compute_ces(np.array([]), art)

    def test_single_token(self):
        """Single-token sequence is valid; mean == max."""
        art = _make_artifact()
        seq = np.array([1.5])
        result = compute_ces(seq, art)
        assert result.token_count == 1
        assert result.mean_entropy == pytest.approx(1.5)
        assert result.max_entropy == pytest.approx(1.5)
        assert result.cdf_mean == result.cdf_max

    def test_no_artifact_yields_zero_cdf(self):
        """Without calibration, CDF values are 0 and CES is 0."""
        seq = np.ones(30)
        result = compute_ces(seq, calibration_artifact=None)
        assert result.cdf_mean == 0.0
        assert result.cdf_max == 0.0
        assert result.ces_score == 0.0

    def test_result_type(self):
        art = _make_artifact()
        seq = _deterministic_entropy()
        result = compute_ces(seq, art)
        assert isinstance(result, CESResult)


# ---------------------------------------------------------------------------
# Risk level mapping
# ---------------------------------------------------------------------------

class TestCESRiskLevel:

    def test_risk_level_unknown_without_thresholds(self):
        art = _make_artifact()
        art.thresholds = {}
        seq = _deterministic_entropy()
        result = compute_ces(seq, art)
        assert result.risk_level == "UNKNOWN"

    def test_risk_level_low(self):
        art = _make_artifact()
        art.thresholds = {"low": 0.5, "medium": 0.8, "high": 0.95}
        # Very low entropy => low CES
        seq = np.ones(30) * 0.01
        result = compute_ces(seq, art)
        assert result.risk_level == "LOW"

    def test_risk_level_critical(self):
        art = _make_artifact()
        art.thresholds = {"low": 0.5, "medium": 0.8, "high": 0.95}
        # Very high entropy => high CES
        seq = np.ones(30) * 10.0
        result = compute_ces(seq, art)
        assert result.risk_level == "CRITICAL"
