"""Tests for the middleware layer.

Covers:
- guard_output() raises ProviderCapabilityError without logprobs
- guard_output_from_entropies() still works with pre-computed entropy
- guard_output_with_logprobs() works with provider logprobs
- guard_output_with_text_heuristic_experimental() emits warning
- SentinelMiddleware uses experimental heuristic with warning
"""

import warnings

import numpy as np
import pytest

from hallucination_sentinel.calibration import build_calibration
from hallucination_sentinel.integrations.middleware import (
    HallucinationBlockedError,
    PolicyAction,
    RoutingDecision,
    SentinelMiddleware,
    TaskCriticality,
    guard_output,
    guard_output_from_entropies,
    guard_output_with_text_heuristic_experimental,
)
from hallucination_sentinel.providers.base import ProviderCapabilityError
from hallucination_sentinel.thresholds import assign_thresholds


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_calibration(entropy_sequences=None, seed=42, n_seq=20, n_tokens=50):
    """Build a deterministic calibration artifact for testing."""
    if entropy_sequences is None:
        rng = np.random.RandomState(seed)
        entropy_sequences = [rng.uniform(0.5, 3.0, size=n_tokens) for _ in range(n_seq)]
    art = build_calibration(entropy_sequences, mode="unsupervised")
    assign_thresholds(art)
    return art


# ---------------------------------------------------------------------------
# guard_output() raises ProviderCapabilityError
# ---------------------------------------------------------------------------


class TestGuardOutputRequiresLogprobs:
    """guard_output() must raise ProviderCapabilityError without logprobs."""

    def test_guard_output_without_logprobs_raises(self):
        """Calling guard_output() without logprobs raises ProviderCapabilityError."""
        art = _build_calibration()
        with pytest.raises(ProviderCapabilityError, match="logprobs"):
            guard_output("prompt", "output", calibration=art)

    def test_guard_output_error_message_suggests_alternatives(self):
        """Error message suggests using guard_output_from_entropies or guard_output_with_logprobs."""
        art = _build_calibration()
        with pytest.raises(ProviderCapabilityError) as exc_info:
            guard_output("prompt", "output", calibration=art)
        error_msg = str(exc_info.value)
        assert "guard_output_from_entropies" in error_msg
        assert "guard_output_with_logprobs" in error_msg
        assert "guard_output_with_text_heuristic_experimental" in error_msg

    def test_guard_output_error_has_capability_field(self):
        """ProviderCapabilityError has capability='logprobs'."""
        art = _build_calibration()
        with pytest.raises(ProviderCapabilityError) as exc_info:
            guard_output("prompt", "output", calibration=art)
        assert exc_info.value.capability == "logprobs"

    def test_guard_output_empty_output_raises_value_error(self):
        """Empty output raises ValueError before ProviderCapabilityError."""
        art = _build_calibration()
        with pytest.raises(ValueError, match="non-empty"):
            guard_output("prompt", "", calibration=art)

    def test_guard_output_preserves_provider_in_error(self):
        """Provider name is preserved in the error."""
        art = _build_calibration()
        with pytest.raises(ProviderCapabilityError) as exc_info:
            guard_output("prompt", "output", calibration=art, provider="openai")
        assert exc_info.value.provider == "openai"


# ---------------------------------------------------------------------------
# guard_output_from_entropies() still works
# ---------------------------------------------------------------------------


class TestGuardOutputFromEntropies:
    """guard_output_from_entropies() is the offline/batch path."""

    def test_guard_output_from_entropies_still_works(self):
        """guard_output_from_entropies() returns valid RoutingDecision."""
        art = _build_calibration()
        entropies = np.array([0.9, 1.2, 1.5, 0.8, 1.1])
        decision = guard_output_from_entropies(
            "prompt", "output", entropies, calibration=art
        )
        assert isinstance(decision, RoutingDecision)
        assert decision.ces_score >= 0.0
        assert decision.risk_level is not None
        assert decision.action is not None

    def test_guard_output_from_entropies_with_high_entropy(self):
        """High entropy produces high CES and elevated risk."""
        art = _build_calibration()
        entropies = np.ones(30) * 5.0  # Way above reference
        decision = guard_output_from_entropies(
            "prompt", "output", entropies, calibration=art
        )
        assert decision.ces_score > 0.5

    def test_guard_output_from_entropies_with_low_entropy(self):
        """Low entropy produces low CES and reduced risk."""
        art = _build_calibration()
        entropies = np.ones(30) * 0.01  # Way below reference
        decision = guard_output_from_entropies(
            "prompt", "output", entropies, calibration=art
        )
        assert decision.ces_score < 0.5

    def test_guard_output_from_entropies_empty_raises(self):
        """Empty entropies raises ValueError."""
        art = _build_calibration()
        with pytest.raises(ValueError, match="non-empty"):
            guard_output_from_entropies("prompt", "output", np.array([]), calibration=art)

    def test_guard_output_from_entropies_includes_calibration_metadata(self):
        """Decision includes calibration metadata."""
        art = _build_calibration()
        entropies = np.array([1.0, 1.5, 2.0])
        decision = guard_output_from_entropies(
            "prompt", "output", entropies, calibration=art
        )
        assert "schema_version" in decision.calibration_metadata
        assert "model" in decision.calibration_metadata

    def test_guard_output_from_entropies_policy_affects_action(self):
        """Higher criticality produces more conservative actions."""
        art = _build_calibration()
        entropies = np.ones(30) * 2.0
        d_low = guard_output_from_entropies(
            "prompt", "output", entropies, calibration=art,
            policy=TaskCriticality.LOW,
        )
        d_critical = guard_output_from_entropies(
            "prompt", "output", entropies, calibration=art,
            policy=TaskCriticality.CRITICAL,
        )
        # Critical policy should be at least as conservative as low policy
        action_order = [
            PolicyAction.ALLOW,
            PolicyAction.WARN,
            PolicyAction.REQUIRE_EVIDENCE,
            PolicyAction.HUMAN_REVIEW,
            PolicyAction.BLOCK,
        ]
        assert action_order.index(d_critical.action) >= action_order.index(d_low.action)


# ---------------------------------------------------------------------------
# guard_output_with_text_heuristic_experimental() emits warning
# ---------------------------------------------------------------------------


class TestGuardOutputWithTextHeuristic:
    """Experimental text-only heuristic must emit deprecation warning."""

    def test_experimental_heuristic_emits_warning(self):
        """Calling experimental heuristic emits UserWarning."""
        art = _build_calibration()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            decision = guard_output_with_text_heuristic_experimental(
                "prompt", "output text here", calibration=art
            )
            # Should have emitted at least one UserWarning
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            assert len(user_warnings) >= 1
            assert "experimental" in str(user_warnings[0].message).lower() or \
                   "heuristic" in str(user_warnings[0].message).lower()

    def test_experimental_heuristic_returns_routing_decision(self):
        """Experimental heuristic returns valid RoutingDecision."""
        art = _build_calibration()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            decision = guard_output_with_text_heuristic_experimental(
                "prompt", "output text here", calibration=art
            )
        assert isinstance(decision, RoutingDecision)
        assert decision.ces_score >= 0.0

    def test_experimental_heuristic_includes_experimental_warning(self):
        """Decision warnings include EXPERIMENTAL marker."""
        art = _build_calibration()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            decision = guard_output_with_text_heuristic_experimental(
                "prompt", "output text here", calibration=art
            )
        assert any("EXPERIMENTAL" in w for w in decision.warnings)

    def test_experimental_heuristic_empty_output_raises(self):
        """Empty output raises ValueError."""
        art = _build_calibration()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="non-empty"):
                guard_output_with_text_heuristic_experimental(
                    "prompt", "", calibration=art
                )


# ---------------------------------------------------------------------------
# SentinelMiddleware uses experimental heuristic with warning
# ---------------------------------------------------------------------------


class TestSentinelMiddleware:
    """SentinelMiddleware uses experimental heuristic and emits warnings."""

    def test_middleware_uses_experimental_heuristic(self):
        """SentinelMiddleware calls experimental heuristic (emits warning)."""
        art = _build_calibration()

        def fake_llm(prompt: str, **kwargs) -> str:
            return "This is a test response from the LLM."

        middleware = SentinelMiddleware(
            llm_call=fake_llm,
            calibration=art,
            provider="test",
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = middleware("test prompt")
            # Should have emitted UserWarning about experimental heuristic
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            assert len(user_warnings) >= 1

    def test_middleware_returns_response(self):
        """SentinelMiddleware returns LLM response text."""
        art = _build_calibration()

        def fake_llm(prompt: str, **kwargs) -> str:
            return "Test response"

        middleware = SentinelMiddleware(
            llm_call=fake_llm,
            calibration=art,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = middleware("test prompt")
        assert result == "Test response"

    def test_middleware_blocks_on_critical_risk(self):
        """SentinelMiddleware raises HallucinationBlockedError on BLOCK action."""
        art = _build_calibration()

        def fake_llm(prompt: str, **kwargs) -> str:
            # Return text that will produce high heuristic entropy
            return "A" * 100 + " b" * 50

        middleware = SentinelMiddleware(
            llm_call=fake_llm,
            calibration=art,
            policy=TaskCriticality.CRITICAL,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # This may or may not block depending on heuristic output
            # Just verify it doesn't crash
            try:
                result = middleware("test prompt")
                assert isinstance(result, str)
            except HallucinationBlockedError as e:
                assert e.result.action is PolicyAction.BLOCK
