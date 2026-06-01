"""Tests for the calibration module."""

import json
import tempfile
from pathlib import Path

import pytest

from hallucination_sentinel.calibration import (
    CalibrationArtifact,
    load_calibration,
    save_calibration,
)


class TestCalibrationArtifact:
    """Tests for CalibrationArtifact.calibrate."""

    def test_sigmoid_midpoint(self):
        """Sigmoid with slope=1, intercept=0 maps 0.0 to 0.5."""
        artifact = CalibrationArtifact(method="sigmoid", params={"slope": 1.0, "intercept": 0.0})
        result = artifact.calibrate(0.0)
        assert abs(result - 0.5) < 1e-10

    def test_sigmoid_high_score(self):
        """Sigmoid maps large positive scores to ~1.0."""
        artifact = CalibrationArtifact(method="sigmoid", params={"slope": 1.0, "intercept": 0.0})
        result = artifact.calibrate(10.0)
        assert result > 0.99

    def test_platt_scaling(self):
        """Platt scaling with a=-1, b=0 maps 0.0 to 0.5."""
        artifact = CalibrationArtifact(method="platt", params={"a": -1.0, "b": 0.0})
        result = artifact.calibrate(0.0)
        assert abs(result - 0.5) < 1e-10

    def test_isotonic_basic(self):
        """Isotonic regression should interpolate between calibration points."""
        artifact = CalibrationArtifact(
            method="isotonic",
            params={"xs": [0.0, 0.5, 1.0], "ys": [0.1, 0.5, 0.9]},
        )
        # Below first point
        assert artifact.calibrate(-0.1) == pytest.approx(0.1)
        # At midpoint
        assert artifact.calibrate(0.5) == pytest.approx(0.5)
        # Above last point
        assert artifact.calibrate(1.5) == pytest.approx(0.9)

    def test_isotonic_empty_fallback(self):
        """Isotonic with empty params returns 0.5 (uncalibrated fallback)."""
        artifact = CalibrationArtifact(method="isotonic", params={})
        assert artifact.calibrate(0.5) == 0.5

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown calibration method"):
            artifact = CalibrationArtifact(method="unknown")
            artifact.calibrate(0.5)


class TestCalibrationIO:
    """Tests for load_calibration and save_calibration."""

    def test_round_trip(self, tmp_path):
        """Save then load should produce an equivalent artifact."""
        original = CalibrationArtifact(
            method="sigmoid",
            params={"slope": 2.0, "intercept": -1.0},
            metadata={"train_date": "2026-01-01"},
        )
        path = tmp_path / "cal.json"
        save_calibration(original, path)

        loaded = load_calibration(path)
        assert loaded.method == original.method
        assert loaded.params == original.params
        assert loaded.metadata == original.metadata

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_calibration("/nonexistent/path/cal.json")
