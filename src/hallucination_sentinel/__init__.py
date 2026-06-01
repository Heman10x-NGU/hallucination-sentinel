"""
Hallucination Sentinel - Single-pass uncertainty firewall for LLM outputs.

Uses calibrated token entropy profiles to flag risky generations before
your agent or RAG system acts on them.

Based on "Entropy Distribution as a Fingerprint for Hallucinations in
Generative Models" (CES algorithm) by Villani et al., 2026.
"""

__version__ = "0.1.0"
__author__ = "Heman10x-NGU"

from .calibration import (
    CalibrationArtifact,
    build_calibration,
    load_calibration,
    save_calibration,
)
from .ces import CESResult, compute_ces
from .entropy import (
    entropy_from_logprobs,
    entropy_from_probs,
    entropy_from_topk_logprobs,
)
from .eval import (
    BaselineResult,
    BootstrapCI,
    ConfusionMetrics,
    EvalRecord,
    EvalReport,
    LengthBucketCoverage,
    build_eval_report,
    compute_baselines,
    compute_metrics,
    load_eval_data,
    report_to_json,
    report_to_markdown,
    save_report_json,
    save_report_markdown,
    split_calibration_eval,
)
from .thresholds import RiskLevel, ThresholdPolicy, assign_thresholds
