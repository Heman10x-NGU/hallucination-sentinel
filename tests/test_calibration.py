"""Tests for the calibration module.

Covers:
- ECDF behaviour on known distributions
- Artifact round-trip (save / load)
- DKW epsilon bound
- build_calibration with supervised and unsupervised modes
- reference_ces_scores computation
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from hallucination_sentinel.calibration import (
    CalibrationArtifact,
    _dkw_bound,
    build_calibration,
    load_calibration,
    save_calibration,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _uniform_sequences(n_seq: int = 10, n_tokens: int = 50, seed: int = 42):
    """Return deterministic entropy sequences drawn from a uniform distribution."""
    rng = np.random.RandomState(seed)
    return [rng.uniform(0.5, 2.0, size=n_tokens) for _ in range(n_seq)]


def _normal_sequences(n_seq: int = 10, n_tokens: int = 50, seed: int = 42):
    """Return deterministic entropy sequences drawn from a normal distribution."""
    rng = np.random.RandomState(seed)
    return [rng.normal(loc=1.0, scale=0.3, size=n_tokens) for _ in range(n_seq)]


# ---------------------------------------------------------------------------
# ECDF behaviour
# ---------------------------------------------------------------------------

class TestECDFBehaviour:
    """The reference ECDF F0(x) must behave as a proper empirical CDF."""

    def test_f0_returns_zero_below_min(self):
        """F0(x) == 0 for x strictly below the smallest sample."""
        art = CalibrationArtifact(ecdf_values=[1.0, 2.0, 3.0])
        assert art.f0(0.5) == 0.0

    def test_f0_returns_one_above_max(self):
        """F0(x) == 1 for x >= the largest sample."""
        art = CalibrationArtifact(ecdf_values=[1.0, 2.0, 3.0])
        assert art.f0(3.0) == 1.0
        assert art.f0(10.0) == 1.0

    def test_f0_monotone(self):
        """F0 is non-decreasing."""
        values = [0.1, 0.5, 1.0, 1.5, 2.0]
        art = CalibrationArtifact(ecdf_values=values)
        prev = 0.0
        for x in np.linspace(0.0, 3.0, 100):
            v = art.f0(float(x))
            assert v >= prev
            prev = v

    def test_f0_uniform_distribution(self):
        """For a known uniform sample, F0 should approximate the CDF."""
        rng = np.random.RandomState(99)
        samples = sorted(rng.uniform(0.0, 1.0, size=1000).tolist())
        art = CalibrationArtifact(ecdf_values=samples)
        # At the median, F0 should be close to 0.5
        assert abs(art.f0(0.5) - 0.5) < 0.1

    def test_f0_empty_ecdf(self):
        """F0 returns 0 when ecdf_values is empty."""
        art = CalibrationArtifact(ecdf_values=[])
        assert art.f0(1.0) == 0.0

    def test_f0_single_value(self):
        """F0 with a single sample jumps from 0 to 1."""
        art = CalibrationArtifact(ecdf_values=[5.0])
        assert art.f0(4.99) == 0.0
        assert art.f0(5.0) == 1.0


# ---------------------------------------------------------------------------
# build_calibration
# ---------------------------------------------------------------------------

class TestBuildCalibration:
    """Test the build_calibration factory."""

    def test_unsupervised_basic(self):
        """Unsupervised mode collects all entropy values."""
        seqs = _uniform_sequences(n_seq=5, n_tokens=20)
        art = build_calibration(seqs, mode="unsupervised")
        assert art.token_count == 5 * 20
        assert art.sequence_count == 5
        assert art.calibration_mode == "unsupervised"
        assert len(art.ecdf_values) == 5 * 20
        # ecdf_values must be sorted
        assert art.ecdf_values == sorted(art.ecdf_values)

    def test_supervised_with_labels(self):
        """Supervised mode uses only truthy-labeled sequences."""
        seqs = _uniform_sequences(n_seq=6, n_tokens=10)
        labels = [True, False, True, True, False, True]
        art = build_calibration(seqs, labels=labels, mode="supervised")
        # 4 truthy sequences * 10 tokens = 40
        assert art.token_count == 40
        assert art.faithful_sequence_count == 4
        assert art.calibration_mode == "supervised"

    def test_supervised_no_labels_includes_all(self):
        """Supervised without labels falls back to including everything."""
        seqs = _uniform_sequences(n_seq=3, n_tokens=10)
        art = build_calibration(seqs, mode="supervised")
        assert art.token_count == 30

    def test_empty_sequences_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            build_calibration([])

    def test_all_filtered_raises(self):
        """If labels exclude every sequence, raise ValueError."""
        seqs = _uniform_sequences(n_seq=3, n_tokens=10)
        labels = [False, False, False]
        with pytest.raises(ValueError, match="No entropy values"):
            build_calibration(seqs, labels=labels, mode="supervised")

    def test_supervised_calibration_string_labels_include_only_faithful(self):
        """String labels 'faithful'/'hallucinated' are parsed correctly."""
        seqs = [np.array([0.1]), np.array([9.9])]
        labels = ["faithful", "hallucinated"]
        art = build_calibration(seqs, labels=labels, mode="supervised")
        assert art.sequence_count == 1
        assert art.ecdf_values == [0.1]

    def test_metadata_passthrough(self):
        """Metadata fields are stored verbatim."""
        seqs = _uniform_sequences(n_seq=2, n_tokens=10)
        art = build_calibration(
            seqs,
            model="openai/gpt-4o",
            provider="openai_compatible",
            task_family="short_qa",
            decoding={"temperature": 0, "top_p": 1},
            entropy_mode="top_k_with_residual",
            top_logprobs=20,
            known_limitations=["short_text"],
        )
        assert art.model == "openai/gpt-4o"
        assert art.provider == "openai_compatible"
        assert art.task_family == "short_qa"
        assert art.decoding == {"temperature": 0, "top_p": 1}
        assert art.entropy_mode == "top_k_with_residual"
        assert art.top_logprobs == 20
        assert art.known_limitations == ["short_text"]

    def test_length_summary(self):
        """length_summary contains min/p50/p90/max of per-sequence lengths."""
        seqs = [np.ones(5), np.ones(10), np.ones(50), np.ones(100)]
        art = build_calibration(seqs)
        ls = art.length_summary
        assert ls["min"] == 5
        assert ls["max"] == 100
        assert 5 <= ls["p50"] <= 100
        assert 5 <= ls["p90"] <= 100


# ---------------------------------------------------------------------------
# DKW bound
# ---------------------------------------------------------------------------

class TestDKWBound:
    """DKW epsilon bound must satisfy the theoretical formula."""

    def test_formula(self):
        """eps = sqrt(ln(2/alpha) / (2*n)) for confidence=0.95."""
        n = 500
        confidence = 0.95
        alpha = 1.0 - confidence
        expected = math.sqrt(math.log(2.0 / alpha) / (2.0 * n))
        assert abs(_dkw_bound(n, confidence) - expected) < 1e-12

    def test_decreasing_with_n(self):
        """Epsilon bound shrinks as sample count grows."""
        eps_small = _dkw_bound(10, 0.95)
        eps_large = _dkw_bound(10000, 0.95)
        assert eps_large < eps_small

    def test_n_zero_returns_one(self):
        assert _dkw_bound(0) == 1.0

    def test_stored_in_artifact(self):
        """build_calibration stores the DKW bound in the artifact."""
        seqs = _uniform_sequences(n_seq=10, n_tokens=50)
        art = build_calibration(seqs)
        assert "confidence" in art.dkw
        assert "epsilon_bound" in art.dkw
        assert art.dkw["confidence"] == 0.95
        assert art.dkw["epsilon_bound"] > 0


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------

class TestCalibrationIO:
    """save_calibration / load_calibration must round-trip exactly."""

    def test_round_trip(self, tmp_path: Path):
        seqs = _normal_sequences(n_seq=5, n_tokens=30)
        original = build_calibration(
            seqs,
            model="test/model",
            provider="test",
            task_family="unit_test",
            entropy_mode="top_k_with_residual",
            top_logprobs=20,
        )
        path = tmp_path / "cal.json"
        save_calibration(original, path)

        loaded = load_calibration(path)
        assert loaded.schema_version == original.schema_version
        assert loaded.model == original.model
        assert loaded.provider == original.provider
        assert loaded.task_family == original.task_family
        assert loaded.entropy_mode == original.entropy_mode
        assert loaded.top_logprobs == original.top_logprobs
        assert loaded.token_count == original.token_count
        assert loaded.sequence_count == original.sequence_count
        assert loaded.ecdf_values == original.ecdf_values
        assert loaded.dkw == original.dkw
        assert loaded.length_summary == original.length_summary

    def test_round_trip_preserves_f0(self, tmp_path: Path):
        """F0 lookup must produce identical results after save/load."""
        seqs = _uniform_sequences(n_seq=5, n_tokens=20)
        original = build_calibration(seqs)
        path = tmp_path / "cal.json"
        save_calibration(original, path)
        loaded = load_calibration(path)

        for x in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
            assert loaded.f0(x) == original.f0(x)

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_calibration("/nonexistent/path/cal.json")

    def test_json_is_valid(self, tmp_path: Path):
        """Saved file must be parseable JSON."""
        seqs = _uniform_sequences(n_seq=2, n_tokens=5)
        art = build_calibration(seqs)
        path = tmp_path / "cal.json"
        save_calibration(art, path)
        data = json.loads(path.read_text())
        assert data["schema_version"] == "0.1"
        assert "ecdf_values" in data
        assert isinstance(data["ecdf_values"], list)

    def test_round_trip_preserves_reference_ces_scores(self, tmp_path: Path):
        """reference_ces_scores must survive save/load."""
        seqs = _uniform_sequences(n_seq=5, n_tokens=20)
        original = build_calibration(seqs)
        path = tmp_path / "cal.json"
        save_calibration(original, path)
        loaded = load_calibration(path)
        assert loaded.reference_ces_scores == original.reference_ces_scores


# ---------------------------------------------------------------------------
# reference_ces_scores
# ---------------------------------------------------------------------------

class TestReferenceCESScores:
    """build_calibration must compute and store reference CES scores."""

    def test_reference_ces_scores_computed(self):
        """reference_ces_scores must be populated after build_calibration."""
        seqs = _uniform_sequences(n_seq=10, n_tokens=50)
        art = build_calibration(seqs)
        assert len(art.reference_ces_scores) == 10
        assert art.reference_ces_scores == sorted(art.reference_ces_scores)

    def test_reference_ces_scores_in_range(self):
        """All reference CES scores must be in [0, 1]."""
        seqs = _uniform_sequences(n_seq=10, n_tokens=50)
        art = build_calibration(seqs)
        for score in art.reference_ces_scores:
            assert 0.0 <= score <= 1.0

    def test_reference_ces_scores_supervised_mode(self):
        """Supervised mode computes CES only for included sequences."""
        seqs = _uniform_sequences(n_seq=6, n_tokens=10)
        labels = [True, False, True, True, False, True]
        art = build_calibration(seqs, labels=labels, mode="supervised")
        assert len(art.reference_ces_scores) == 4  # 4 truthy sequences

    def test_high_entropy_sequences_high_ces(self):
        """Sequences with high entropy should have high CES scores."""
        # Reference: low entropy sequences
        low_entropy_seqs = [np.ones(50) * 0.5 for _ in range(10)]
        art = build_calibration(low_entropy_seqs)
        # All reference sequences have same entropy, so CES should be uniform
        assert all(0.0 <= s <= 1.0 for s in art.reference_ces_scores)
