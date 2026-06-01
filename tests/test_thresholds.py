"""Tests for the thresholds module."""

import pytest

from hallucination_sentinel.thresholds import (
    RiskLevel,
    ThresholdPolicy,
    assign_thresholds,
)


class TestThresholdPolicy:
    """Tests for ThresholdPolicy.classify."""

    def test_low_default(self):
        """Score below medium threshold => LOW."""
        policy = ThresholdPolicy()
        assert policy.classify(0.1) == RiskLevel.LOW

    def test_medium_default(self):
        """Score between medium and high => MEDIUM."""
        policy = ThresholdPolicy()
        assert policy.classify(0.45) == RiskLevel.MEDIUM

    def test_high_default(self):
        """Score between high and critical => HIGH."""
        policy = ThresholdPolicy()
        assert policy.classify(0.75) == RiskLevel.HIGH

    def test_critical_default(self):
        """Score above critical threshold => CRITICAL."""
        policy = ThresholdPolicy()
        assert policy.classify(0.95) == RiskLevel.CRITICAL

    def test_boundary_medium(self):
        """Exactly at medium threshold => MEDIUM."""
        policy = ThresholdPolicy(medium=0.30)
        assert policy.classify(0.30) == RiskLevel.MEDIUM

    def test_boundary_high(self):
        """Exactly at high threshold => HIGH."""
        policy = ThresholdPolicy(high=0.60)
        assert policy.classify(0.60) == RiskLevel.HIGH

    def test_custom_thresholds(self):
        """Custom thresholds should shift boundaries."""
        policy = ThresholdPolicy(medium=0.50, high=0.80, critical=0.95)
        assert policy.classify(0.40) == RiskLevel.LOW
        assert policy.classify(0.60) == RiskLevel.MEDIUM
        assert policy.classify(0.85) == RiskLevel.HIGH
        assert policy.classify(0.96) == RiskLevel.CRITICAL


class TestAssignThresholds:
    """Tests for assign_thresholds."""

    def test_batch_classification(self):
        """Should classify a list of probabilities."""
        probs = [0.1, 0.4, 0.7, 0.9]
        levels = assign_thresholds(probs)
        assert levels == [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]

    def test_empty_list(self):
        """Empty input returns empty output."""
        assert assign_thresholds([]) == []

    def test_custom_policy(self):
        """Should accept a custom policy."""
        policy = ThresholdPolicy(medium=0.5, high=0.75, critical=0.9)
        levels = assign_thresholds([0.3, 0.6, 0.8, 0.95], policy=policy)
        assert levels == [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
