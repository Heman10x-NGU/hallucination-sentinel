"""Tests for the entropy calculation module.

All tests use deterministic inputs with analytically known entropy values.
"""

import math

import numpy as np
import pytest

from hallucination_sentinel.entropy import (
    EPSILON,
    EntropyResult,
    entropy_from_logprobs,
    entropy_from_probs,
    entropy_from_topk_logprobs,
    length_normalized_entropy,
    perplexity_from_logprobs,
)


# ---------------------------------------------------------------------------
# entropy_from_probs
# ---------------------------------------------------------------------------
class TestEntropyFromProbs:
    """Tests for entropy_from_probs."""

    # --- known-value cases ---

    def test_uniform_distribution_10(self):
        """Uniform distribution over 10 outcomes has entropy ln(10)."""
        n = 10
        probs = np.ones(n) / n
        result = entropy_from_probs(probs)
        assert abs(result - np.log(n)) < 1e-10

    @pytest.mark.parametrize("n", [2, 3, 4, 8, 16, 64, 100, 1000])
    def test_uniform_distribution_parametrized(self, n):
        """Uniform distribution over N outcomes has entropy ln(N)."""
        probs = np.ones(n) / n
        result = entropy_from_probs(probs)
        assert abs(result - np.log(n)) < 1e-10

    def test_peaked_distribution_near_zero(self):
        """A peaked distribution (one dominant outcome) has entropy near 0."""
        probs = np.array([0.999, 0.0005, 0.0005])
        result = entropy_from_probs(probs)
        assert result < 0.01

    def test_certain_distribution(self):
        """A certain distribution (one outcome) has entropy ~0."""
        probs = np.array([1.0, 0.0, 0.0])
        result = entropy_from_probs(probs)
        # EPSILON clipping introduces a tiny residual
        assert abs(result) < 1e-7

    def test_binary_uniform_is_ln2(self):
        """Uniform binary distribution has entropy ln(2)."""
        probs = np.array([0.5, 0.5])
        result = entropy_from_probs(probs)
        assert abs(result - np.log(2)) < 1e-10

    def test_known_binary_asymmetric(self):
        """Binary distribution [0.25, 0.75] has analytically known entropy."""
        p = 0.25
        expected = -(p * np.log(p) + (1 - p) * np.log(1 - p))
        result = entropy_from_probs(np.array([p, 1 - p]))
        assert abs(result - expected) < 1e-10

    def test_known_ternary(self):
        """Ternary [0.5, 0.25, 0.25] has analytically known entropy."""
        probs = np.array([0.5, 0.25, 0.25])
        expected = -(0.5 * np.log(0.5) + 0.25 * np.log(0.25) + 0.25 * np.log(0.25))
        result = entropy_from_probs(probs)
        assert abs(result - expected) < 1e-10

    # --- base-2 ---

    def test_base2_uniform_8_is_3_bits(self):
        """Entropy in bits for uniform distribution over 8 outcomes is 3."""
        probs = np.ones(8) / 8
        result = entropy_from_probs(probs, base="2")
        assert abs(result - 3.0) < 1e-10

    def test_base2_uniform_parametrized(self):
        """Entropy in bits for uniform over 2^n is n."""
        for n in [1, 2, 4, 8, 16]:
            k = 2**n
            probs = np.ones(k) / k
            result = entropy_from_probs(probs, base="2")
            assert abs(result - n) < 1e-10

    def test_base2_binary_uniform_is_1(self):
        """Uniform binary distribution has entropy 1 bit."""
        result = entropy_from_probs(np.array([0.5, 0.5]), base="2")
        assert abs(result - 1.0) < 1e-10

    # --- edge cases ---

    def test_single_token_returns_zero(self):
        """Single-element distribution has exactly zero entropy."""
        result = entropy_from_probs(np.array([1.0]))
        assert result == 0.0

    def test_single_token_base2_returns_zero(self):
        """Single-element distribution has exactly zero entropy in bits."""
        result = entropy_from_probs(np.array([1.0]), base="2")
        assert result == 0.0

    def test_two_tokens_close_to_uniform(self):
        """Two tokens with slightly off probabilities still close to ln(2)."""
        probs = np.array([0.51, 0.49])
        result = entropy_from_probs(probs)
        assert abs(result - np.log(2)) < 0.01

    # --- error cases ---

    def test_negative_probs_raise(self):
        """Negative probabilities should raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            entropy_from_probs(np.array([-0.1, 0.5, 0.6]))

    def test_empty_array_raises(self):
        """Empty probability array should raise ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            entropy_from_probs(np.array([]))

    def test_nan_probs_raise(self):
        """NaN probabilities should raise ValueError."""
        with pytest.raises(ValueError, match="NaN"):
            entropy_from_probs(np.array([0.5, float("nan"), 0.5]))

    def test_inf_probs_raise(self):
        """Inf probabilities should raise ValueError."""
        with pytest.raises(ValueError, match="NaN|Inf"):
            entropy_from_probs(np.array([0.5, float("inf"), 0.5]))

    def test_unknown_base_raises(self):
        """Unknown base should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown base"):
            entropy_from_probs(np.array([0.5, 0.5]), base="10")


# ---------------------------------------------------------------------------
# entropy_from_logprobs
# ---------------------------------------------------------------------------
class TestEntropyFromLogprobs:
    """Tests for entropy_from_logprobs."""

    def test_matches_probs_uniform(self):
        """Entropy from logprobs should match entropy from probs (uniform)."""
        probs = np.array([0.25, 0.25, 0.25, 0.25])
        logprobs = np.log(probs)
        assert abs(entropy_from_logprobs(logprobs) - entropy_from_probs(probs)) < 1e-10

    def test_matches_probs_asymmetric(self):
        """Entropy from logprobs matches probs for asymmetric distribution."""
        probs = np.array([0.1, 0.3, 0.6])
        logprobs = np.log(probs)
        assert abs(entropy_from_logprobs(logprobs) - entropy_from_probs(probs)) < 1e-10

    def test_known_uniform_value(self):
        """Uniform over 4 has entropy ln(4)."""
        logprobs = np.log(np.ones(4) / 4)
        result = entropy_from_logprobs(logprobs)
        assert abs(result - np.log(4)) < 1e-10

    def test_base2_matches_probs(self):
        """Base-2 entropy from logprobs matches base-2 from probs."""
        probs = np.array([0.5, 0.25, 0.25])
        logprobs = np.log(probs)
        assert abs(
            entropy_from_logprobs(logprobs, base="2") - entropy_from_probs(probs, base="2")
        ) < 1e-10

    def test_single_token_returns_zero(self):
        """Single logprob yields zero entropy."""
        assert entropy_from_logprobs(np.array([-0.5])) == 0.0

    def test_empty_raises(self):
        """Empty logprobs array should raise ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            entropy_from_logprobs(np.array([]))

    def test_nan_raises(self):
        """NaN in logprobs should raise ValueError."""
        with pytest.raises(ValueError, match="NaN"):
            entropy_from_logprobs(np.array([-0.5, float("nan")]))

    def test_peaked_distribution_near_zero(self):
        """Peaked distribution (high confidence) has entropy near 0."""
        # log(0.999) ~ -0.001, log(0.001) ~ -6.9
        logprobs = np.array([np.log(0.999), np.log(0.0005), np.log(0.0005)])
        result = entropy_from_logprobs(logprobs)
        assert result < 0.01

    def test_unknown_base_raises(self):
        """Unknown base should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown base"):
            entropy_from_logprobs(np.array([-0.5, -0.5]), base="16")


# ---------------------------------------------------------------------------
# entropy_from_topk_logprobs
# ---------------------------------------------------------------------------
class TestEntropyFromTopkLogprobs:
    """Tests for entropy_from_topk_logprobs."""

    def test_basic_topk_returns_entropy_result(self):
        """Top-k entropy should produce a valid EntropyResult."""
        topk = [{"a": np.log(0.5), "b": np.log(0.3), "c": np.log(0.2)}]
        result = entropy_from_topk_logprobs(topk, top_k=3)
        assert isinstance(result, EntropyResult)
        assert result.entropy >= 0
        assert result.token_count == 1
        assert result.top_k == 3
        assert result.entropy_base == "e"

    def test_topk_approximates_full_when_mass_near_1(self):
        """When top-k captures ~100% mass, entropy approximates full entropy."""
        # 3 tokens summing to 1.0
        topk = [{"a": np.log(0.5), "b": np.log(0.3), "c": np.log(0.2)}]
        full_entropy = entropy_from_probs(np.array([0.5, 0.3, 0.2]))
        result = entropy_from_topk_logprobs(topk, top_k=3, add_residual=True)
        assert abs(result.entropy - full_entropy) < 0.01

    def test_residual_mode_increases_entropy(self):
        """With residual bucket, entropy should be higher than without."""
        # Only 60% of mass in top-k
        topk = [{"a": np.log(0.4), "b": np.log(0.2)}]
        with_residual = entropy_from_topk_logprobs(topk, top_k=2, add_residual=True)
        without_residual = entropy_from_topk_logprobs(topk, top_k=2, add_residual=False)
        assert with_residual.entropy >= without_residual.entropy

    def test_residual_mode_has_higher_entropy_for_sparse(self):
        """Residual mode gives notably higher entropy for sparse top-k."""
        # Only 30% of mass in top-k — large residual
        topk = [{"a": np.log(0.2), "b": np.log(0.1)}]
        with_residual = entropy_from_topk_logprobs(topk, top_k=2, add_residual=True)
        without_residual = entropy_from_topk_logprobs(topk, top_k=2, add_residual=False)
        # With residual should be significantly higher
        assert with_residual.entropy > without_residual.entropy + 0.1

    def test_observed_mass_tracked(self):
        """observed_mass_mean should reflect actual observed probability mass."""
        topk = [{"a": np.log(0.5), "b": np.log(0.3)}]
        result = entropy_from_topk_logprobs(topk, top_k=2)
        # Observed mass is exp(log(0.5)) + exp(log(0.3)) = 0.8
        assert abs(result.observed_mass_mean - 0.8) < 1e-6

    def test_multiple_positions(self):
        """Multiple positions produce per-token entropies array."""
        topk = [
            {"a": np.log(0.5), "b": np.log(0.5)},
            {"c": np.log(0.9), "d": np.log(0.1)},
        ]
        result = entropy_from_topk_logprobs(topk, top_k=2)
        assert result.token_count == 2
        assert len(result.entropies) == 2
        # First position (uniform) should have higher entropy than second (peaked)
        assert result.entropies[0] > result.entropies[1]

    def test_entropy_is_mean_of_per_token(self):
        """result.entropy should be the mean of per-token entropies."""
        topk = [
            {"a": np.log(0.5), "b": np.log(0.5)},
            {"c": np.log(0.9), "d": np.log(0.1)},
        ]
        result = entropy_from_topk_logprobs(topk, top_k=2)
        assert abs(result.entropy - np.mean(result.entropies)) < 1e-10

    def test_base2_mode(self):
        """Base-2 should produce entropy in bits."""
        topk = [{"a": np.log(0.5), "b": np.log(0.5)}]
        result = entropy_from_topk_logprobs(topk, top_k=2, base="2")
        assert result.entropy_base == "2"
        # ln(2)/ln(2) = 1 bit
        assert abs(result.entropy - 1.0) < 0.05

    def test_single_token_topk(self):
        """Single token in top-k should produce near-zero entropy."""
        topk = [{"only": np.log(1.0)}]
        result = entropy_from_topk_logprobs(topk, top_k=1)
        assert result.entropy < 0.01

    def test_warnings_on_low_observed_mass(self):
        """Warnings should appear when observed mass is below 80%."""
        # Very sparse: only 10% of mass observed
        topk = [{"a": np.log(0.05), "b": np.log(0.05)}]
        result = entropy_from_topk_logprobs(topk, top_k=2, add_residual=False)
        assert len(result.warnings) > 0
        assert "observed probability mass" in result.warnings[0].lower()

    def test_no_warnings_when_mass_near_1(self):
        """No warnings when observed mass is near 100%."""
        topk = [{"a": np.log(0.6), "b": np.log(0.4)}]
        result = entropy_from_topk_logprobs(topk, top_k=2)
        assert len(result.warnings) == 0

    def test_empty_list_raises(self):
        """Empty topk_logprobs list should raise ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            entropy_from_topk_logprobs([], top_k=3)

    def test_known_entropy_with_residual(self):
        """Verify entropy of top-k with residual matches analytical calculation."""
        # Top-2 tokens capture 80% mass; residual is 20%
        topk = [{"a": np.log(0.5), "b": np.log(0.3)}]
        result = entropy_from_topk_logprobs(topk, top_k=2, add_residual=True)

        # Analytical: entropy of [0.5, 0.3, 0.2] after clip+normalize
        expected_probs = np.array([0.5, 0.3, 0.2])
        expected_probs = np.clip(expected_probs, EPSILON, None)
        expected_probs = expected_probs / expected_probs.sum()
        expected_entropy = -np.sum(expected_probs * np.log(expected_probs))
        assert abs(result.entropy - expected_entropy) < 0.01


# ---------------------------------------------------------------------------
# perplexity_from_logprobs
# ---------------------------------------------------------------------------
class TestPerplexity:
    """Tests for perplexity_from_logprobs."""

    def test_high_confidence_low_perplexity(self):
        """High-confidence logprobs should yield low perplexity."""
        logprobs = np.array([-0.01, -0.01, -0.01])
        assert perplexity_from_logprobs(logprobs) < 1.1

    def test_uniform_perplexity_is_n(self):
        """Perplexity of uniform distribution over N tokens is N."""
        n = 10
        logprobs = np.log(np.ones(n) / n)  # each is -ln(n)
        result = perplexity_from_logprobs(logprobs)
        assert abs(result - n) < 1e-10

    def test_uniform_perplexity_parametrized(self):
        """Perplexity of uniform over N is N for various N."""
        for n in [2, 4, 8, 16, 64]:
            logprobs = np.log(np.ones(n) / n)
            result = perplexity_from_logprobs(logprobs)
            assert abs(result - n) < 1e-10

    def test_peaked_perplexity_near_one(self):
        """All-same token logprobs near 0 yield perplexity near 1."""
        # When all logprobs are close to 0 (i.e., all tokens are nearly certain),
        # perplexity = exp(-mean(~0)) ~ 1
        logprobs = np.array([-0.001, -0.001, -0.001])
        result = perplexity_from_logprobs(logprobs)
        assert result < 1.01

    def test_single_token_perplexity_is_one(self):
        """Single token has perplexity exactly 1."""
        result = perplexity_from_logprobs(np.array([0.0]))  # log(1) = 0
        assert abs(result - 1.0) < 1e-10

    def test_empty_raises(self):
        """Empty logprobs should raise ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            perplexity_from_logprobs(np.array([]))

    def test_nan_raises(self):
        """NaN in logprobs should raise ValueError."""
        with pytest.raises(ValueError, match="NaN"):
            perplexity_from_logprobs(np.array([-0.5, float("nan")]))


# ---------------------------------------------------------------------------
# length_normalized_entropy
# ---------------------------------------------------------------------------
class TestLengthNormalizedEntropy:
    """Tests for length_normalized_entropy."""

    def test_is_mean(self):
        """Length-normalized entropy is the mean of per-token entropies."""
        entropies = np.array([1.0, 2.0, 3.0])
        assert abs(length_normalized_entropy(entropies) - 2.0) < 1e-10

    def test_single_token(self):
        """Single token entropy is just that entropy value."""
        assert abs(length_normalized_entropy(np.array([3.5])) - 3.5) < 1e-10

    def test_uniform_entropies(self):
        """All-same entropies yield that value."""
        entropies = np.array([2.0, 2.0, 2.0, 2.0])
        assert abs(length_normalized_entropy(entropies) - 2.0) < 1e-10

    def test_empty_raises(self):
        """Empty entropies array should raise ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            length_normalized_entropy(np.array([]))

    def test_nan_raises(self):
        """NaN in entropies should raise ValueError."""
        with pytest.raises(ValueError, match="NaN"):
            length_normalized_entropy(np.array([1.0, float("nan")]))


# ---------------------------------------------------------------------------
# EntropyResult dataclass
# ---------------------------------------------------------------------------
class TestEntropyResult:
    """Tests for the EntropyResult dataclass."""

    def test_warnings_default_to_empty_list(self):
        """Warnings should default to an empty list."""
        result = EntropyResult(
            entropy=1.0,
            entropies=np.array([1.0]),
            entropy_mode="full",
            entropy_base="e",
            token_count=1,
        )
        assert result.warnings == []

    def test_all_fields_accessible(self):
        """All declared fields should be accessible."""
        result = EntropyResult(
            entropy=2.5,
            entropies=np.array([1.0, 2.0]),
            entropy_mode="top_k_with_residual",
            entropy_base="e",
            token_count=2,
            top_k=5,
            observed_mass_mean=0.95,
            warnings=["test warning"],
        )
        assert result.entropy == 2.5
        assert len(result.entropies) == 2
        assert result.entropy_mode == "top_k_with_residual"
        assert result.entropy_base == "e"
        assert result.token_count == 2
        assert result.top_k == 5
        assert result.observed_mass_mean == 0.95
        assert result.warnings == ["test warning"]
