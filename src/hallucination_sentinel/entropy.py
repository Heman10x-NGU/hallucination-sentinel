"""
Entropy calculation utilities for token probability distributions.

Implements Shannon entropy from full probabilities, log probabilities,
and approximate entropy from top-k log probabilities with residual bucket.

Uses natural logarithm by default (nats), matching the paper's convention.
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


# Numerical stability epsilon
EPSILON = 1e-10


@dataclass
class EntropyResult:
    """Result of entropy calculation with metadata."""

    # Core entropy values
    entropy: float  # Shannon entropy in nats
    entropies: np.ndarray  # Per-token entropy values

    # Metadata
    entropy_mode: str  # "full" | "top_k" | "top_k_with_residual" | "selected_only"
    entropy_base: str  # "e" (nats) or "2" (bits)
    token_count: int

    # Top-k specific
    top_k: Optional[int] = None
    observed_mass_mean: Optional[float] = None

    # Warnings
    warnings: list[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


def entropy_from_probs(probs: np.ndarray, base: str = "e") -> float:
    """
    Calculate Shannon entropy from probability distribution.

    Args:
        probs: Probability distribution (must sum to ~1)
        base: Logarithm base ("e" for nats, "2" for bits)

    Returns:
        Shannon entropy value

    Raises:
        ValueError: If probabilities are invalid
    """
    probs = np.asarray(probs, dtype=np.float64)

    # Validate non-empty
    if probs.size == 0:
        raise ValueError("Probability array must be non-empty")

    # Validate no NaN/Inf
    if np.any(np.isnan(probs)) or np.any(np.isinf(probs)):
        raise ValueError("Probabilities must not contain NaN or Inf")

    # Validate non-negative
    if np.any(probs < 0):
        raise ValueError("Probabilities must be non-negative")

    # Single-token case: entropy is exactly 0
    if probs.size == 1:
        return 0.0

    # Clip to avoid log(0)
    probs = np.clip(probs, EPSILON, None)

    # Normalize (handle floating point errors)
    probs = probs / probs.sum()

    # Calculate entropy
    if base == "e":
        return -np.sum(probs * np.log(probs))
    elif base == "2":
        return -np.sum(probs * np.log2(probs))
    else:
        raise ValueError(f"Unknown base: {base}. Use 'e' or '2'.")


def entropy_from_logprobs(logprobs: np.ndarray, base: str = "e") -> float:
    """
    Calculate Shannon entropy from log probabilities.

    Args:
        logprobs: Log probabilities (natural log for base="e")
        base: Logarithm base ("e" for nats, "2" for bits)

    Returns:
        Shannon entropy value

    Raises:
        ValueError: If logprobs are invalid
    """
    logprobs = np.asarray(logprobs, dtype=np.float64)

    # Validate non-empty
    if logprobs.size == 0:
        raise ValueError("Logprobs array must be non-empty")

    # Validate no NaN/Inf
    if np.any(np.isnan(logprobs)):
        raise ValueError("Logprobs must not contain NaN")

    # Single-token case: entropy is exactly 0
    if logprobs.size == 1:
        return 0.0

    # Convert to probabilities (logprobs should be <= 0)
    probs = np.exp(logprobs)

    # Calculate entropy: -sum(p * log(p))
    # Since logprobs = log(p), we have log(p) directly
    if base == "e":
        return -np.sum(probs * logprobs)
    elif base == "2":
        return -np.sum(probs * logprobs / np.log(2))
    else:
        raise ValueError(f"Unknown base: {base}. Use 'e' or '2'.")


def entropy_from_topk_logprobs(
    topk_logprobs: list[dict[str, float]],
    top_k: int,
    base: str = "e",
    add_residual: bool = True,
) -> EntropyResult:
    """
    Calculate approximate entropy from top-k log probabilities.

    Many APIs only expose top-k token logprobs, not full vocabulary.
    This function computes approximate entropy with optional residual bucket.

    Args:
        topk_logprobs: List of dicts mapping token -> logprob for each position
        top_k: Number of top tokens returned
        base: Logarithm base ("e" for nats, "2" for bits)
        add_residual: Whether to add residual bucket for unseen probability mass

    Returns:
        EntropyResult with entropy values and metadata

    Raises:
        ValueError: If inputs are invalid
    """
    if not topk_logprobs:
        raise ValueError("topk_logprobs must be non-empty")

    entropies = []
    observed_masses = []
    warnings = []

    for position_logprobs in topk_logprobs:
        # Get probabilities from logprobs
        tokens = list(position_logprobs.keys())
        logprob_values = np.array(list(position_logprobs.values()), dtype=np.float64)
        probs = np.exp(logprob_values)

        # Track observed probability mass
        observed_mass = np.sum(probs)
        observed_masses.append(observed_mass)

        if add_residual and observed_mass < 1.0 - EPSILON:
            # Add residual bucket for unseen probability mass
            residual = max(0.0, 1.0 - observed_mass)
            probs_with_residual = np.append(probs, residual)

            # Calculate entropy with residual
            probs_with_residual = np.clip(probs_with_residual, EPSILON, None)
            probs_with_residual = probs_with_residual / probs_with_residual.sum()

            if base == "e":
                entropy = -np.sum(probs_with_residual * np.log(probs_with_residual))
            else:
                entropy = -np.sum(probs_with_residual * np.log2(probs_with_residual))

            mode = "top_k_with_residual"
        else:
            # Use only observed probabilities
            probs = np.clip(probs, EPSILON, None)
            probs = probs / probs.sum()

            if base == "e":
                entropy = -np.sum(probs * np.log(probs))
            else:
                entropy = -np.sum(probs * np.log2(probs))

            mode = "top_k"

        entropies.append(entropy)

    # Check for low observed mass (approximation warning)
    mean_observed_mass = np.mean(observed_masses) if observed_masses else 1.0
    if mean_observed_mass < 0.8:
        warnings.append(
            f"Low observed probability mass ({mean_observed_mass:.2%}). "
            "Entropy approximation may be inaccurate."
        )

    return EntropyResult(
        entropy=float(np.mean(entropies)),
        entropies=np.array(entropies),
        entropy_mode=mode,
        entropy_base=base,
        token_count=len(topk_logprobs),
        top_k=top_k,
        observed_mass_mean=float(mean_observed_mass),
        warnings=warnings,
    )


def perplexity_from_logprobs(logprobs: np.ndarray) -> float:
    """
    Calculate perplexity from log probabilities.

    Perplexity = exp(mean(-log(p))) = exp(H) where H is cross-entropy.

    Args:
        logprobs: Log probabilities (natural log)

    Returns:
        Perplexity value

    Raises:
        ValueError: If logprobs are invalid
    """
    logprobs = np.asarray(logprobs, dtype=np.float64)

    if logprobs.size == 0:
        raise ValueError("Logprobs array must be non-empty")

    if np.any(np.isnan(logprobs)):
        raise ValueError("Logprobs must not contain NaN")

    return float(np.exp(-np.mean(logprobs)))


def length_normalized_entropy(entropies: np.ndarray, base: str = "e") -> float:
    """
    Calculate length-normalized entropy (mean entropy per token).

    This is the baseline "LN-Entropy" from the paper.

    Args:
        entropies: Per-token entropy values
        base: Logarithm base

    Returns:
        Length-normalized entropy

    Raises:
        ValueError: If entropies are invalid
    """
    entropies = np.asarray(entropies, dtype=np.float64)

    if entropies.size == 0:
        raise ValueError("Entropies array must be non-empty")

    if np.any(np.isnan(entropies)):
        raise ValueError("Entropies must not contain NaN")

    return float(np.mean(entropies))


def compute_token_entropies(
    token_data: list[dict],
    entropy_mode: str = "auto",
    base: str = "e",
) -> EntropyResult:
    """
    Compute per-token entropies from token data.

    Args:
        token_data: List of token data dicts with logprobs/probs
        entropy_mode: "auto", "full", "top_k", or "selected_only"
        base: Logarithm base

    Returns:
        EntropyResult with all entropy values and metadata

    Raises:
        ValueError: If token_data is empty or contains invalid entries
    """
    if not token_data:
        raise ValueError("token_data must be non-empty")

    entropies = []
    warnings = []

    for token_info in token_data:
        if "logprobs" in token_info and token_info["logprobs"]:
            # Full logprobs available
            logprobs = token_info["logprobs"]
            entropy = entropy_from_logprobs(logprobs, base)
            entropies.append(entropy)
        elif "top_logprobs" in token_info and token_info["top_logprobs"]:
            # Top-k logprobs available
            topk = token_info["top_logprobs"]
            result = entropy_from_topk_logprobs([topk], len(topk), base)
            entropies.append(result.entropy)
        elif "selected_logprob" in token_info:
            # Only selected token logprob (not enough for CES)
            warnings.append(
                "Only selected-token logprob available. "
                "Cannot compute token entropy for CES. "
                "Use perplexity baseline only."
            )
            # Use 0 as placeholder (will be filtered)
            entropies.append(0.0)
        else:
            raise ValueError(f"Unknown token data format: {token_info.keys()}")

    if not entropies:
        raise ValueError("No valid token data found")

    # Determine mode
    if entropy_mode == "auto":
        if "full" in [t.get("mode", "") for t in token_data]:
            mode = "full"
        elif "top_k" in [t.get("mode", "") for t in token_data]:
            mode = "top_k"
        else:
            mode = "unknown"
    else:
        mode = entropy_mode

    return EntropyResult(
        entropy=float(np.mean(entropies)),
        entropies=np.array(entropies),
        entropy_mode=mode,
        entropy_base=base,
        token_count=len(token_data),
        warnings=warnings,
    )
