"""
Evaluation harness for the Hallucination Sentinel CES algorithm.

Computes discrimination metrics, baseline comparisons, calibration diagnostics,
and bootstrap confidence intervals from scored evaluation data.

Requires scikit-learn for AUROC/AUPRC (gracefully degrades without it).
Requires numpy for bootstrap confidence intervals.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .schemas import parse_label

# Optional sklearn dependency
try:
    from sklearn.metrics import (
        average_precision_score,
        confusion_matrix,
        roc_auc_score,
    )

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

# Length buckets for calibration coverage diagnostics
LENGTH_BUCKETS: list[tuple[str, int, Optional[int]]] = [
    ("<5", 0, 5),
    ("5-9", 5, 10),
    ("10-49", 10, 50),
    ("50-99", 50, 100),
    (">=100", 100, None),
]


@dataclass(frozen=True)
class EvalRecord:
    """A single scored evaluation record loaded from JSONL."""

    ces_score: float
    label: int  # 1 = hallucination, 0 = faithful
    token_count: int
    token_entropies: tuple[float, ...]
    calibrated_probability: Optional[float] = None


def load_eval_data(path: str | Path) -> list[EvalRecord]:
    """Load evaluation records from a JSONL file.

    Each line must be a JSON object with at minimum:
      - ``ces_score`` (float)
      - ``label`` (int, 1 = hallucination, 0 = faithful)
      - ``token_count`` (int)
      - ``token_entropies`` (list[float])

    Optional fields:
      - ``calibrated_probability`` (float)

    Args:
        path: Path to the JSONL file.

    Returns:
        A list of :class:`EvalRecord` instances.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If a record is missing required fields.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Eval data file not found: {path}")

    records: list[EvalRecord] = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {lineno}: {exc}") from exc

            missing = {"ces_score", "label", "token_count", "token_entropies"} - obj.keys()
            if missing:
                raise ValueError(
                    f"Line {lineno}: missing required fields: {sorted(missing)}"
                )

            parsed = parse_label(obj["label"], context=f"eval line {lineno}")

            records.append(
                EvalRecord(
                    ces_score=float(obj["ces_score"]),
                    label=1 if parsed.hallucinated else 0,
                    token_count=int(obj["token_count"]),
                    token_entropies=tuple(float(v) for v in obj["token_entropies"]),
                    calibrated_probability=(
                        float(obj["calibrated_probability"])
                        if "calibrated_probability" in obj
                        else None
                    ),
                )
            )

    if not records:
        raise ValueError(f"No records found in {path}")

    return records


def normalize_to_eval_record(
    obj: dict,
    lineno: int,
    calibration: Any = None,
) -> EvalRecord:
    """Normalize a JSONL dict into an :class:`EvalRecord`.

    Supports two input formats:

    **Pre-scored rows** (all required fields present)::

        {"ces_score": 0.8, "label": 1, "token_count": 10,
         "token_entropies": [1.2, 0.8, ...]}

    **Raw entropy rows** (requires *calibration* artifact)::

        {"entropy": [1.2, 0.8, 0.5, ...], "label": 1}

    When *calibration* is provided and the row has ``entropy`` but no
    ``ces_score``, CES is computed on the fly.

    Args:
        obj: Parsed JSON dict from one JSONL line.
        lineno: Line number (1-indexed) for error messages.
        calibration: Optional calibration artifact for computing CES
            from raw entropy rows.

    Returns:
        A populated :class:`EvalRecord`.

    Raises:
        ValueError: If required fields are missing or invalid.
    """
    parsed = parse_label(obj.get("label"), context=f"eval line {lineno}")

    # Pre-scored row: has ces_score already
    if "ces_score" in obj:
        missing = {"token_count", "token_entropies"} - obj.keys()
        if missing:
            raise ValueError(
                f"Line {lineno}: pre-scored row missing fields: {sorted(missing)}"
            )
        return EvalRecord(
            ces_score=float(obj["ces_score"]),
            label=1 if parsed.hallucinated else 0,
            token_count=int(obj["token_count"]),
            token_entropies=tuple(float(v) for v in obj["token_entropies"]),
            calibrated_probability=(
                float(obj["calibrated_probability"])
                if "calibrated_probability" in obj
                else None
            ),
        )

    # Raw entropy row: needs calibration to compute CES
    entropy_vals = obj.get("entropy")
    if entropy_vals is None:
        raise ValueError(
            f"Line {lineno}: row must have either 'ces_score' or 'entropy' field."
        )
    if calibration is None:
        raise ValueError(
            f"Line {lineno}: raw entropy row requires --calibration to compute CES."
        )

    entropies = tuple(float(v) for v in entropy_vals)
    if not entropies:
        raise ValueError(f"Line {lineno}: 'entropy' array is empty.")

    from .ces import compute_ces

    ces_result = compute_ces(np.array(entropies, dtype=np.float64), calibration)

    return EvalRecord(
        ces_score=float(ces_result.ces_score),
        label=1 if parsed.hallucinated else 0,
        token_count=len(entropies),
        token_entropies=entropies,
    )


# ---------------------------------------------------------------------------
# Calibration / evaluation set split
# ---------------------------------------------------------------------------

def split_calibration_eval(
    records: list[EvalRecord],
    calibration_fraction: float = 0.3,
    seed: int = 42,
) -> tuple[list[EvalRecord], list[EvalRecord]]:
    """Split records into calibration and evaluation subsets.

    The split is stratified by label to preserve the hallucination base rate
    in both subsets.

    Args:
        records: Full list of evaluation records.
        calibration_fraction: Fraction of records to use for calibration (0-1).
        seed: Random seed for reproducibility.

    Returns:
        A ``(calibration_records, eval_records)`` tuple.
    """
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be in (0, 1)")

    rng = np.random.RandomState(seed)
    indices = np.arange(len(records))
    labels = np.array([r.label for r in records])

    cal_indices = []
    eval_indices = []

    for label_val in np.unique(labels):
        group = indices[labels == label_val]
        rng.shuffle(group)
        n_cal = max(1, int(len(group) * calibration_fraction))
        cal_indices.extend(group[:n_cal].tolist())
        eval_indices.extend(group[n_cal:].tolist())

    cal_indices.sort()
    eval_indices.sort()

    return [records[i] for i in cal_indices], [records[i] for i in eval_indices]


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

@dataclass
class ConfusionMetrics:
    """Confusion matrix metrics at a single threshold."""

    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int
    tpr: float  # sensitivity / recall
    fpr: float  # false positive rate
    fnr: float  # false negative rate
    precision: float
    f1: float


def _confusion_at_threshold(
    predictions: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> ConfusionMetrics:
    """Compute confusion matrix metrics at a given threshold.

    Predictions >= threshold are classified as hallucination (positive).
    """
    pred_binary = (predictions >= threshold).astype(int)
    tp = int(np.sum((pred_binary == 1) & (labels == 1)))
    fp = int(np.sum((pred_binary == 1) & (labels == 0)))
    tn = int(np.sum((pred_binary == 0) & (labels == 0)))
    fn = int(np.sum((pred_binary == 0) & (labels == 1)))

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (
        2 * precision * tpr / (precision + tpr)
        if (precision + tpr) > 0
        else 0.0
    )

    return ConfusionMetrics(
        threshold=threshold,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        tpr=tpr,
        fpr=fpr,
        fnr=fnr,
        precision=precision,
        f1=f1,
    )


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

@dataclass
class BootstrapCI:
    """Bootstrap confidence interval for a metric."""

    point_estimate: float
    lower: float
    upper: float
    confidence: float = 0.95
    n_bootstrap: int = 1000


def _bootstrap_ci(
    values: np.ndarray,
    statistic_fn: Any = np.mean,
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> BootstrapCI:
    """Compute a bootstrap confidence interval.

    Args:
        values: 1-D array of values to bootstrap.
        statistic_fn: Function to compute the statistic (default: mean).
        confidence: Confidence level (default 0.95).
        n_bootstrap: Number of bootstrap resamples.
        seed: Random seed for reproducibility.

    Returns:
        A :class:`BootstrapCI` with the point estimate and bounds.
    """
    values = np.asarray(values, dtype=np.float64)
    point = float(statistic_fn(values))

    if values.size < 2:
        return BootstrapCI(
            point_estimate=point, lower=point, upper=point,
            confidence=confidence, n_bootstrap=n_bootstrap,
        )

    rng = np.random.RandomState(seed)
    boot_stats = np.empty(n_bootstrap)
    n = len(values)
    for i in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        boot_stats[i] = float(statistic_fn(sample))

    alpha = 1.0 - confidence
    lower = float(np.percentile(boot_stats, 100 * alpha / 2))
    upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))

    return BootstrapCI(
        point_estimate=point, lower=lower, upper=upper,
        confidence=confidence, n_bootstrap=n_bootstrap,
    )


# ---------------------------------------------------------------------------
# Lag-1 autocorrelation
# ---------------------------------------------------------------------------

def _lag1_autocorrelation(values: np.ndarray) -> float:
    """Compute lag-1 autocorrelation of a sequence.

    Measures temporal dependence in the score sequence. Values near 0
    indicate independence; values near 1 indicate strong positive
    autocorrelation.

    Args:
        values: 1-D array of values.

    Returns:
        The lag-1 autocorrelation coefficient, or 0.0 for sequences
        shorter than 2 elements.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        return 0.0
    mean = np.mean(values)
    var = np.var(values)
    if var < 1e-15:
        return 0.0
    return float(np.mean((values[:-1] - mean) * (values[1:] - mean)) / var)


# ---------------------------------------------------------------------------
# Calibration coverage by token length bucket
# ---------------------------------------------------------------------------

@dataclass
class LengthBucketCoverage:
    """Calibration coverage statistics for a token length bucket."""

    bucket_name: str
    min_tokens: int
    max_tokens: Optional[int]
    count: int
    mean_ces: float
    hallucination_rate: float


def _calibration_coverage(
    records: list[EvalRecord],
) -> list[LengthBucketCoverage]:
    """Compute calibration coverage statistics by token length bucket.

    Buckets: <5, 5-9, 10-49, 50-99, >=100 tokens.

    Args:
        records: List of evaluation records.

    Returns:
        A list of :class:`LengthBucketCoverage` for each bucket.
    """
    results: list[LengthBucketCoverage] = []

    for bucket_name, min_tok, max_tok in LENGTH_BUCKETS:
        bucket_records = [
            r for r in records
            if r.token_count >= min_tok and (max_tok is None or r.token_count < max_tok)
        ]
        count = len(bucket_records)
        if count == 0:
            results.append(
                LengthBucketCoverage(
                    bucket_name=bucket_name,
                    min_tokens=min_tok,
                    max_tokens=max_tok,
                    count=0,
                    mean_ces=0.0,
                    hallucination_rate=0.0,
                )
            )
        else:
            mean_ces = float(np.mean([r.ces_score for r in bucket_records]))
            hall_rate = float(np.mean([r.label for r in bucket_records]))
            results.append(
                LengthBucketCoverage(
                    bucket_name=bucket_name,
                    min_tokens=min_tok,
                    max_tokens=max_tok,
                    count=count,
                    mean_ces=mean_ces,
                    hallucination_rate=hall_rate,
                )
            )

    return results


# ---------------------------------------------------------------------------
# Baseline comparisons
# ---------------------------------------------------------------------------

@dataclass
class BaselineResult:
    """Result of evaluating a single baseline method."""

    name: str
    predictions: np.ndarray
    auroc: Optional[float] = None
    auprc: Optional[float] = None
    bootstrap_auroc: Optional[BootstrapCI] = None
    bootstrap_auprc: Optional[BootstrapCI] = None


def _compute_raw_geometric_mean(records: list[EvalRecord]) -> np.ndarray:
    """Compute raw geometric mean of F0(mean_ent) and F0(max_ent) without
    calibration artifact -- uses ECDF of the evaluation set itself."""
    all_entropies = []
    for r in records:
        all_entropies.extend(r.token_entropies)
    sorted_ent = np.sort(np.array(all_entropies, dtype=np.float64))

    scores = []
    for r in records:
        ent = np.array(r.token_entropies, dtype=np.float64)
        mean_e = float(np.mean(ent))
        max_e = float(np.max(ent))
        cdf_mean = float(np.searchsorted(sorted_ent, mean_e, side="right") / len(sorted_ent))
        cdf_max = float(np.searchsorted(sorted_ent, max_e, side="right") / len(sorted_ent))
        scores.append(math.sqrt(cdf_mean * cdf_max))
    return np.array(scores)


def _compute_perplexity_proxy(records: list[EvalRecord]) -> np.ndarray:
    """Perplexity proxy: exp(mean(token_entropies))."""
    return np.array([float(np.exp(np.mean(r.token_entropies))) for r in records])


def _compute_generation_length(records: list[EvalRecord]) -> np.ndarray:
    """Generation length baseline: token count as a predictor."""
    return np.array([float(r.token_count) for r in records])


def _compute_mean_entropy(records: list[EvalRecord]) -> np.ndarray:
    """Mean entropy baseline."""
    return np.array([float(np.mean(r.token_entropies)) for r in records])


def _compute_max_entropy(records: list[EvalRecord]) -> np.ndarray:
    """Max entropy baseline."""
    return np.array([float(np.max(r.token_entropies)) for r in records])


def _compute_length_normalized_entropy(records: list[EvalRecord]) -> np.ndarray:
    """Length-normalized entropy baseline (same as mean entropy for per-token values)."""
    return np.array([float(np.mean(r.token_entropies)) for r in records])


BASELINE_COMPUTERS = {
    "perplexity": _compute_perplexity_proxy,
    "generation_length": _compute_generation_length,
    "mean_entropy": _compute_mean_entropy,
    "max_entropy": _compute_max_entropy,
    "length_normalized_entropy": _compute_length_normalized_entropy,
    "raw_geometric_mean": _compute_raw_geometric_mean,
}


def compute_baselines(
    records: list[EvalRecord],
    labels: np.ndarray,
    bootstrap: bool = True,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, BaselineResult]:
    """Compute discrimination metrics for all baseline methods.

    Args:
        records: Evaluation records.
        labels: Ground truth labels (1 = hallucination).
        bootstrap: Whether to compute bootstrap CIs.
        n_bootstrap: Number of bootstrap resamples.
        seed: Random seed.

    Returns:
        Dict mapping baseline name to :class:`BaselineResult`.
    """
    results: dict[str, BaselineResult] = {}

    for name, computer in BASELINE_COMPUTERS.items():
        preds = computer(records)
        auroc, auprc = None, None
        boot_auroc, boot_auprc = None, None

        if HAS_SKLEARN and len(np.unique(labels)) > 1:
            auroc = float(roc_auc_score(labels, preds))
            auprc = float(average_precision_score(labels, preds))
            if bootstrap:
                boot_auroc = _bootstrap_metric(labels, preds, roc_auc_score, n_bootstrap, seed)
                boot_auprc = _bootstrap_metric(labels, preds, average_precision_score, n_bootstrap, seed)

        results[name] = BaselineResult(
            name=name,
            predictions=preds,
            auroc=auroc,
            auprc=auprc,
            bootstrap_auroc=boot_auroc,
            bootstrap_auprc=boot_auprc,
        )

    return results


def _bootstrap_metric(
    labels: np.ndarray,
    predictions: np.ndarray,
    metric_fn: Any,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> BootstrapCI:
    """Bootstrap a sklearn metric function over paired (label, prediction) samples."""
    point = float(metric_fn(labels, predictions))
    rng = np.random.RandomState(seed)
    n = len(labels)
    boot_stats = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        y_boot = labels[idx]
        p_boot = predictions[idx]
        if len(np.unique(y_boot)) < 2:
            boot_stats[i] = point  # skip degenerate resamples
        else:
            boot_stats[i] = float(metric_fn(y_boot, p_boot))

    alpha = 1.0 - 0.95
    lower = float(np.percentile(boot_stats, 100 * alpha / 2))
    upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))

    return BootstrapCI(
        point_estimate=point, lower=lower, upper=upper,
        confidence=0.95, n_bootstrap=n_bootstrap,
    )


# ---------------------------------------------------------------------------
# EvalReport dataclass
# ---------------------------------------------------------------------------

@dataclass
class EvalReport:
    """Complete evaluation report for the CES algorithm and baselines.

    Attributes:
        method: Name of the primary method (default "CES").
        n_samples: Total number of evaluation samples.
        n_positive: Number of hallucination samples (label=1).
        n_negative: Number of faithful samples (label=0).
        hallucination_rate: Fraction of positive samples.
        auroc: Area under ROC curve (CES).
        auprc: Area under precision-recall curve (CES).
        confusion_matrices: Confusion metrics at configured thresholds.
        bootstrap_auroc: Bootstrap CI for AUROC (CES).
        bootstrap_auprc: Bootstrap CI for AUPRC (CES).
        lag1_autocorrelation: Lag-1 autocorrelation of CES scores.
        calibration_coverage: Per-length-bucket coverage stats.
        baselines: Discrimination metrics for baseline methods.
        created_at: ISO timestamp of report creation.
    """

    method: str = "CES"
    n_samples: int = 0
    n_positive: int = 0
    n_negative: int = 0
    hallucination_rate: float = 0.0

    # Primary metrics (CES)
    auroc: Optional[float] = None
    auprc: Optional[float] = None
    confusion_matrices: list[ConfusionMetrics] = field(default_factory=list)
    bootstrap_auroc: Optional[BootstrapCI] = None
    bootstrap_auprc: Optional[BootstrapCI] = None

    # Diagnostics
    lag1_autocorrelation: float = 0.0
    calibration_coverage: list[LengthBucketCoverage] = field(default_factory=list)

    # Baselines
    baselines: dict[str, BaselineResult] = field(default_factory=dict)

    created_at: str = ""


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    thresholds: Optional[list[float]] = None,
    bootstrap: bool = True,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute discrimination metrics for a set of predictions.

    Args:
        predictions: 1-D array of prediction scores.
        labels: 1-D array of binary labels (1 = positive/hallucination).
        thresholds: List of decision thresholds for confusion matrices.
            Defaults to [0.5, 0.7, 0.8, 0.9].
        bootstrap: Whether to compute bootstrap confidence intervals.
        n_bootstrap: Number of bootstrap resamples.
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys: ``auroc``, ``auprc``, ``confusion_matrices``,
        ``bootstrap_auroc``, ``bootstrap_auprc``.
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    if thresholds is None:
        thresholds = [0.5, 0.7, 0.8, 0.9]

    result: dict[str, Any] = {
        "auroc": None,
        "auprc": None,
        "confusion_matrices": [],
        "bootstrap_auroc": None,
        "bootstrap_auprc": None,
    }

    if HAS_SKLEARN and len(np.unique(labels)) > 1:
        result["auroc"] = float(roc_auc_score(labels, predictions))
        result["auprc"] = float(average_precision_score(labels, predictions))

        if bootstrap:
            result["bootstrap_auroc"] = _bootstrap_metric(
                labels, predictions, roc_auc_score, n_bootstrap, seed
            )
            result["bootstrap_auprc"] = _bootstrap_metric(
                labels, predictions, average_precision_score, n_bootstrap, seed
            )

    for t in thresholds:
        cm = _confusion_at_threshold(predictions, labels, t)
        result["confusion_matrices"].append(cm)

    return result


# ---------------------------------------------------------------------------
# Full report builder
# ---------------------------------------------------------------------------

def build_eval_report(
    records: list[EvalRecord],
    thresholds: Optional[list[float]] = None,
    bootstrap: bool = True,
    n_bootstrap: int = 1000,
    seed: int = 42,
    method: str = "CES",
) -> EvalReport:
    """Build a full evaluation report from scored records.

    Args:
        records: Evaluation records (from :func:`load_eval_data`).
        thresholds: Decision thresholds for confusion matrices.
        bootstrap: Whether to compute bootstrap CIs.
        n_bootstrap: Number of bootstrap resamples.
        seed: Random seed.
        method: Name of the primary method.

    Returns:
        A complete :class:`EvalReport`.
    """
    predictions = np.array([r.ces_score for r in records], dtype=np.float64)
    labels = np.array([r.label for r in records], dtype=np.int64)

    metrics = compute_metrics(
        predictions, labels, thresholds, bootstrap, n_bootstrap, seed
    )

    # Lag-1 autocorrelation on ordered CES scores
    lag1 = _lag1_autocorrelation(predictions)

    # Calibration coverage
    coverage = _calibration_coverage(records)

    # Baselines
    baselines = compute_baselines(records, labels, bootstrap, n_bootstrap, seed)

    return EvalReport(
        method=method,
        n_samples=len(records),
        n_positive=int(np.sum(labels)),
        n_negative=int(len(labels) - np.sum(labels)),
        hallucination_rate=float(np.mean(labels)),
        auroc=metrics["auroc"],
        auprc=metrics["auprc"],
        confusion_matrices=metrics["confusion_matrices"],
        bootstrap_auroc=metrics["bootstrap_auroc"],
        bootstrap_auprc=metrics["bootstrap_auprc"],
        lag1_autocorrelation=lag1,
        calibration_coverage=coverage,
        baselines=baselines,
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


# ---------------------------------------------------------------------------
# Report serialization: JSON
# ---------------------------------------------------------------------------

def _ci_to_dict(ci: BootstrapCI) -> dict:
    return {
        "point_estimate": ci.point_estimate,
        "lower": ci.lower,
        "upper": ci.upper,
        "confidence": ci.confidence,
        "n_bootstrap": ci.n_bootstrap,
    }


def _cm_to_dict(cm: ConfusionMetrics) -> dict:
    return {
        "threshold": cm.threshold,
        "tp": cm.tp,
        "fp": cm.fp,
        "tn": cm.tn,
        "fn": cm.fn,
        "tpr": cm.tpr,
        "fpr": cm.fpr,
        "fnr": cm.fnr,
        "precision": cm.precision,
        "f1": cm.f1,
    }


def _coverage_to_dict(c: LengthBucketCoverage) -> dict:
    return {
        "bucket": c.bucket_name,
        "min_tokens": c.min_tokens,
        "max_tokens": c.max_tokens,
        "count": c.count,
        "mean_ces": c.mean_ces,
        "hallucination_rate": c.hallucination_rate,
    }


def report_to_json(report: EvalReport) -> str:
    """Serialize an :class:`EvalReport` to a JSON string.

    Args:
        report: The evaluation report.

    Returns:
        JSON string with full report contents.
    """
    data: dict[str, Any] = {
        "method": report.method,
        "created_at": report.created_at,
        "dataset": {
            "n_samples": report.n_samples,
            "n_positive": report.n_positive,
            "n_negative": report.n_negative,
            "hallucination_rate": round(report.hallucination_rate, 6),
        },
        "primary_metrics": {
            "auroc": report.auroc,
            "auprc": report.auprc,
            "bootstrap_auroc": _ci_to_dict(report.bootstrap_auroc) if report.bootstrap_auroc else None,
            "bootstrap_auprc": _ci_to_dict(report.bootstrap_auprc) if report.bootstrap_auprc else None,
        },
        "confusion_matrices": [_cm_to_dict(cm) for cm in report.confusion_matrices],
        "diagnostics": {
            "lag1_autocorrelation": round(report.lag1_autocorrelation, 6),
            "calibration_coverage": [_coverage_to_dict(c) for c in report.calibration_coverage],
        },
        "baselines": {},
    }

    for name, bl in report.baselines.items():
        entry: dict[str, Any] = {
            "auroc": bl.auroc,
            "auprc": bl.auprc,
        }
        if bl.bootstrap_auroc:
            entry["bootstrap_auroc"] = _ci_to_dict(bl.bootstrap_auroc)
        if bl.bootstrap_auprc:
            entry["bootstrap_auprc"] = _ci_to_dict(bl.bootstrap_auprc)
        data["baselines"][name] = entry

    return json.dumps(data, indent=2)


def save_report_json(report: EvalReport, path: str | Path) -> None:
    """Save an evaluation report as JSON to disk.

    Args:
        report: The evaluation report.
        path: Destination file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_to_json(report))


# ---------------------------------------------------------------------------
# Report serialization: Markdown
# ---------------------------------------------------------------------------

def report_to_markdown(report: EvalReport) -> str:
    """Render an :class:`EvalReport` as a Markdown string.

    Args:
        report: The evaluation report.

    Returns:
        Markdown-formatted report.
    """
    lines: list[str] = []
    lines.append(f"# Evaluation Report: {report.method}")
    lines.append("")
    lines.append(f"**Generated:** {report.created_at}")
    lines.append("")

    # Dataset summary
    lines.append("## Dataset")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total samples | {report.n_samples} |")
    lines.append(f"| Hallucinations (positive) | {report.n_positive} |")
    lines.append(f"| Faithful (negative) | {report.n_negative} |")
    lines.append(f"| Hallucination rate | {report.hallucination_rate:.4f} |")
    lines.append("")

    # Primary metrics
    lines.append("## Primary Metrics")
    lines.append("")
    lines.append(f"| Metric | Value | 95% CI |")
    lines.append(f"|--------|-------|--------|")
    auroc_str = f"{report.auroc:.4f}" if report.auroc is not None else "N/A"
    auprc_str = f"{report.auprc:.4f}" if report.auprc is not None else "N/A"
    auroc_ci = ""
    auprc_ci = ""
    if report.bootstrap_auroc:
        auroc_ci = f"[{report.bootstrap_auroc.lower:.4f}, {report.bootstrap_auroc.upper:.4f}]"
    if report.bootstrap_auprc:
        auprc_ci = f"[{report.bootstrap_auprc.lower:.4f}, {report.bootstrap_auprc.upper:.4f}]"
    lines.append(f"| AUROC | {auroc_str} | {auroc_ci} |")
    lines.append(f"| AUPRC | {auprc_str} | {auprc_ci} |")
    lines.append("")

    # Confusion matrices
    if report.confusion_matrices:
        lines.append("## Confusion Matrices")
        lines.append("")
        lines.append("| Threshold | TP | FP | TN | FN | TPR | FPR | FNR | Precision | F1 |")
        lines.append("|-----------|----|----|----|----|-----|-----|-----|-----------|-----|")
        for cm in report.confusion_matrices:
            lines.append(
                f"| {cm.threshold:.2f} | {cm.tp} | {cm.fp} | {cm.tn} | {cm.fn} | "
                f"{cm.tpr:.4f} | {cm.fpr:.4f} | {cm.fnr:.4f} | {cm.precision:.4f} | {cm.f1:.4f} |"
            )
        lines.append("")

    # Diagnostics
    lines.append("## Diagnostics")
    lines.append("")
    lines.append(f"**Lag-1 autocorrelation:** {report.lag1_autocorrelation:.6f}")
    lines.append("")

    if report.calibration_coverage:
        lines.append("### Calibration Coverage by Token Length")
        lines.append("")
        lines.append("| Bucket | Count | Mean CES | Hallucination Rate |")
        lines.append("|--------|-------|----------|-------------------|")
        for c in report.calibration_coverage:
            lines.append(
                f"| {c.bucket_name} | {c.count} | {c.mean_ces:.4f} | {c.hallucination_rate:.4f} |"
            )
        lines.append("")

    # Baselines
    if report.baselines:
        lines.append("## Baseline Comparisons")
        lines.append("")
        lines.append("| Method | AUROC | AUPRC | AUROC 95% CI |")
        lines.append("|--------|-------|-------|--------------|")

        # CES row first
        ces_auroc_ci = ""
        if report.bootstrap_auroc:
            ces_auroc_ci = f"[{report.bootstrap_auroc.lower:.4f}, {report.bootstrap_auroc.upper:.4f}]"
        lines.append(
            f"| **{report.method}** | "
            f"{report.auroc:.4f} | "
            f"{report.auprc:.4f} | "
            f"{ces_auroc_ci} |"
        )

        for name, bl in report.baselines.items():
            bl_auroc = f"{bl.auroc:.4f}" if bl.auroc is not None else "N/A"
            bl_auprc = f"{bl.auprc:.4f}" if bl.auprc is not None else "N/A"
            bl_ci = ""
            if bl.bootstrap_auroc:
                bl_ci = f"[{bl.bootstrap_auroc.lower:.4f}, {bl.bootstrap_auroc.upper:.4f}]"
            lines.append(f"| {name} | {bl_auroc} | {bl_auprc} | {bl_ci} |")
        lines.append("")

    return "\n".join(lines)


def save_report_markdown(report: EvalReport, path: str | Path) -> None:
    """Save an evaluation report as Markdown to disk.

    Args:
        report: The evaluation report.
        path: Destination file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_to_markdown(report))
