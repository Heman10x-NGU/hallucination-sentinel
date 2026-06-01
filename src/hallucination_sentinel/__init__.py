"""
Hallucination Sentinel - Single-pass uncertainty firewall for LLM outputs.

Uses calibrated token entropy profiles to flag risky generations before
your agent or RAG system acts on them.

Based on "Entropy Distribution as a Fingerprint for Hallucinations in
Generative Models" (CES algorithm) by Villani et al., 2026.
"""

__version__ = "0.1.0"
__author__ = "Heman10x-NGU"

from .ces import compute_ces
from .entropy import (
    entropy_from_logprobs,
    entropy_from_probs,
    entropy_from_topk_logprobs,
)
from .calibration import CalibrationArtifact, load_calibration, save_calibration
from .thresholds import ThresholdPolicy, assign_thresholds
