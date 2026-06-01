"""Tests for the CES (Conditional Entropy Sweep) algorithm."""

import numpy as np
import pytest

from hallucination_sentinel.ces import CESResult, compute_ces


class TestComputeCES:
    """Tests for compute_ces."""

    def test_uniform_entropy_low_score(self):
        """Uniform entropy profile should produce low CES score (near 0)."""
        entropies = np.ones(50) * 1.5
        result = compute_ces(entropies)
        assert isinstance(result, CESResult)
        assert result.ces_score < 1e-10

    def test_spike_produces_high_score(self):
        """A sharp entropy spike should produce a higher CES score."""
        entropies = np.ones(50) * 0.1
        entropies[25] = 5.0  # spike
        result = compute_ces(entropies)
        assert result.ces_score > 0.0

    def test_empty_raises(self):
        """Empty entropy array should raise ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            compute_ces(np.array([]))

    def test_single_token(self):
        """Single token should return CES score of 0."""
        result = compute_ces(np.array([1.0]))
        assert result.ces_score == 0.0

    def test_window_size_clamped(self):
        """Window size larger than data should be clamped."""
        entropies = np.array([1.0, 2.0, 3.0])
        result = compute_ces(entropies, window_size=100)
        assert result.window_size == 3
        assert len(result.warnings) == 1

    def test_default_window_size(self):
        """Default window size should be sqrt(n)."""
        entropies = np.ones(100)
        result = compute_ces(entropies)
        assert result.window_size == 10  # sqrt(100)

    def test_sweep_positions_shape(self):
        """Sweep positions and entropies should have consistent shape."""
        entropies = np.ones(20)
        result = compute_ces(entropies, window_size=5)
        assert len(result.sweep_positions) == len(result.sweep_entropies)
        assert len(result.sweep_positions) == 20 - 5 + 1
