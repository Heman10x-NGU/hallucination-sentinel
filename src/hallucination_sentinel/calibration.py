"""
Calibration artifacts for the Hallucination Sentinel CES algorithm.

Calibration builds a reference empirical CDF (F0) from token entropy sequences
of known-faithful or pooled outputs. The CES score is then computed as the
geometric mean of F0 applied to mean and max entropy of a new sequence.

Supports:
- Unsupervised calibration (all sequences, no labels)
- Supervised calibration (only faithful-labeled sequences for F0)

Calibration artifacts are versioned JSON with full metadata for reproducibility.
"""

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from .schemas import parse_label


SCHEMA_VERSION = "0.1"


@dataclass
class CalibrationArtifact:
    """A calibration artifact containing the reference ECDF and full metadata.

    The ECDF is built from per-token entropy values of reference sequences.
    ``ecdf_values`` is the sorted array of all observed entropy values from the
    reference pool.  F0(x) = fraction of ecdf_values <= x.
    """

    # -- schema / provenance ------------------------------------------------
    schema_version: str = SCHEMA_VERSION
    created_at: str = ""
    model: str = ""
    provider: str = ""
    task_family: str = ""

    # -- decoding config recorded at calibration time -----------------------
    decoding: dict = field(default_factory=dict)
    # e.g. {"temperature": 0, "top_p": 1, "max_new_tokens": 128}

    # -- entropy configuration ----------------------------------------------
    entropy_mode: str = "full"  # "full" | "top_k" | "top_k_with_residual"
    entropy_base: str = "e"  # "e" (nats) or "2" (bits)
    top_logprobs: int = 0  # 0 means full logits

    # -- calibration metadata -----------------------------------------------
    calibration_mode: str = "unsupervised"  # "unsupervised" | "supervised"
    token_count: int = 0  # total entropy tokens in reference pool
    sequence_count: int = 0  # number of sequences in reference pool
    faithful_sequence_count: int = 0  # supervised only

    # -- length summary (per-sequence token counts) -------------------------
    length_summary: dict = field(default_factory=dict)
    # e.g. {"p50": 14, "p90": 96, "min": 2, "max": 256}

    # -- DKW confidence band ------------------------------------------------
    dkw: dict = field(default_factory=dict)
    # e.g. {"confidence": 0.95, "epsilon_bound": 0.061}

    # -- reference ECDF values (sorted entropy samples) ---------------------
    ecdf_values: list = field(default_factory=list)
    # Sorted list of all per-token entropy values from reference sequences.

    # -- reference CES scores (sorted) -------------------------------------
    reference_ces_scores: list = field(default_factory=list)
    # Sorted list of CES scores computed for each reference sequence.
    # Used by thresholds.assign_thresholds to define risk bands in [0,1] space.

    # -- threshold quantiles (set by thresholds.assign_thresholds) ----------
    thresholds: dict = field(default_factory=dict)
    # e.g. {"low": 0.75, "medium": 0.90, "high": 0.97}

    # -- known limitations --------------------------------------------------
    known_limitations: list = field(default_factory=list)

    # ---- ECDF lookup -------------------------------------------------------

    def f0(self, x: float) -> float:
        """Evaluate the reference empirical CDF at *x*.

        F0(x) = fraction of ``ecdf_values`` that are <= x.

        Returns 0.0 if the artifact has no ecdf_values.
        """
        if not self.ecdf_values:
            return 0.0
        arr = np.asarray(self.ecdf_values, dtype=np.float64)
        return float(np.searchsorted(arr, x, side="right") / len(arr))


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def _percentile(values: list[float], q: float) -> float:
    """Return the q-th percentile of *values* (0-100 scale)."""
    if not values:
        return 0.0
    return float(np.percentile(values, q))


def _length_summary(sequence_lengths: list[int]) -> dict:
    if not sequence_lengths:
        return {}
    return {
        "min": min(sequence_lengths),
        "p50": _percentile(sequence_lengths, 50),
        "p90": _percentile(sequence_lengths, 90),
        "max": max(sequence_lengths),
    }


def _dkw_bound(n: int, confidence: float = 0.95) -> float:
    """DKW-style uniform epsilon bound.

    P(sup|F_n(x) - F(x)| > eps) <= 2 * exp(-2 * n * eps^2)

    Solve for eps given confidence = 1 - alpha:
        eps = sqrt(ln(2 / alpha) / (2 * n))
    """
    if n <= 0:
        return 1.0
    alpha = 1.0 - confidence
    return math.sqrt(math.log(2.0 / alpha) / (2.0 * n))


def build_calibration(
    entropy_sequences: list[np.ndarray],
    labels: Optional[list] = None,
    mode: str = "unsupervised",
    *,
    model: str = "",
    provider: str = "",
    task_family: str = "",
    decoding: Optional[dict] = None,
    entropy_mode: str = "full",
    entropy_base: str = "e",
    top_logprobs: int = 0,
    known_limitations: Optional[list[str]] = None,
) -> CalibrationArtifact:
    """Build a calibration artifact from a collection of entropy sequences.

    Args:
        entropy_sequences: List of 1-D arrays, one per generation, containing
            per-token entropy values (nats by default).
        labels: Optional per-sequence labels.  For ``mode="supervised"``,
            only sequences whose label is truthy contribute to the reference
            ECDF.  Ignored for unsupervised mode.
        mode: ``"unsupervised"`` (default) or ``"supervised"``.
        model, provider, task_family, decoding, entropy_mode, entropy_base,
            top_logprobs: Metadata stored in the artifact.
        known_limitations: Free-text limitation tags.

    Returns:
        A :class:`CalibrationArtifact` with the fitted ECDF and metadata.

    Raises:
        ValueError: If no entropy values are available after filtering.
    """
    if not entropy_sequences:
        raise ValueError("entropy_sequences must be non-empty")

    # Collect all per-token entropy values from selected sequences
    all_entropies: list[float] = []
    sequence_lengths: list[int] = []
    faithful_count = 0

    for idx, seq in enumerate(entropy_sequences):
        seq = np.asarray(seq, dtype=np.float64).ravel()
        if seq.size == 0:
            continue
        # Decide whether to include this sequence in the reference pool
        include = True
        if mode == "supervised" and labels is not None:
            parsed = parse_label(labels[idx], context=f"calibration sequence {idx}")
            include = parsed.faithful
            if include:
                faithful_count += 1
        elif mode == "supervised":
            # supervised but no labels => include all (same as unsupervised)
            faithful_count += 1
        if include:
            all_entropies.extend(float(v) for v in seq)
            sequence_lengths.append(len(seq))

    if not all_entropies:
        raise ValueError(
            "No entropy values available after filtering.  "
            "Check that entropy_sequences are non-empty and labels select "
            "at least one sequence in supervised mode."
        )

    # Sorted ECDF values
    ecdf_sorted = sorted(all_entropies)

    # DKW bound
    n = len(ecdf_sorted)
    confidence = 0.95
    eps = _dkw_bound(n, confidence)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    artifact = CalibrationArtifact(
        schema_version=SCHEMA_VERSION,
        created_at=now,
        model=model,
        provider=provider,
        task_family=task_family,
        decoding=decoding or {},
        entropy_mode=entropy_mode,
        entropy_base=entropy_base,
        top_logprobs=top_logprobs,
        calibration_mode=mode,
        token_count=n,
        sequence_count=len(sequence_lengths),
        faithful_sequence_count=faithful_count if mode == "supervised" else len(sequence_lengths),
        length_summary=_length_summary(sequence_lengths),
        dkw={"confidence": confidence, "epsilon_bound": round(eps, 6)},
        ecdf_values=ecdf_sorted,
        reference_ces_scores=[],
        thresholds={},
        known_limitations=known_limitations or [],
    )

    # Compute CES for each reference sequence using the artifact's F0.
    # This avoids importing compute_ces (which would create a circular import).
    ces_scores: list[float] = []
    for idx, seq in enumerate(entropy_sequences):
        seq = np.asarray(seq, dtype=np.float64).ravel()
        if seq.size == 0:
            continue
        include = True
        if mode == "supervised" and labels is not None:
            parsed = parse_label(labels[idx], context=f"calibration sequence {idx}")
            include = parsed.faithful
        if include:
            mean_ent = float(np.mean(seq))
            max_ent = float(np.max(seq))
            cdf_mean = artifact.f0(mean_ent)
            cdf_max = artifact.f0(max_ent)
            ces = math.sqrt(cdf_mean * cdf_max)
            ces_scores.append(ces)

    artifact.reference_ces_scores = sorted(ces_scores)

    return artifact


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------

def save_calibration(artifact: CalibrationArtifact, path: str | Path) -> None:
    """Serialize a calibration artifact to JSON on disk.

    Args:
        artifact: The calibration artifact to persist.
        path: Destination file path.  Parent directories are created if needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "schema_version": artifact.schema_version,
        "created_at": artifact.created_at,
        "model": artifact.model,
        "provider": artifact.provider,
        "task_family": artifact.task_family,
        "decoding": artifact.decoding,
        "entropy_mode": artifact.entropy_mode,
        "entropy_base": artifact.entropy_base,
        "top_logprobs": artifact.top_logprobs,
        "calibration_mode": artifact.calibration_mode,
        "token_count": artifact.token_count,
        "sequence_count": artifact.sequence_count,
        "faithful_sequence_count": artifact.faithful_sequence_count,
        "length_summary": artifact.length_summary,
        "dkw": artifact.dkw,
        "ecdf_values": artifact.ecdf_values,
        "reference_ces_scores": artifact.reference_ces_scores,
        "thresholds": artifact.thresholds,
        "known_limitations": artifact.known_limitations,
    }
    path.write_text(json.dumps(data, indent=2))


def load_calibration(path: str | Path) -> CalibrationArtifact:
    """Load a calibration artifact from a JSON file.

    Args:
        path: Path to the calibration JSON file.

    Returns:
        A :class:`CalibrationArtifact`.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found: {path}")
    data = json.loads(path.read_text())
    return CalibrationArtifact(
        schema_version=data.get("schema_version", SCHEMA_VERSION),
        created_at=data.get("created_at", ""),
        model=data.get("model", ""),
        provider=data.get("provider", ""),
        task_family=data.get("task_family", ""),
        decoding=data.get("decoding", {}),
        entropy_mode=data.get("entropy_mode", "full"),
        entropy_base=data.get("entropy_base", "e"),
        top_logprobs=data.get("top_logprobs", 0),
        calibration_mode=data.get("calibration_mode", "unsupervised"),
        token_count=data.get("token_count", 0),
        sequence_count=data.get("sequence_count", 0),
        faithful_sequence_count=data.get("faithful_sequence_count", 0),
        length_summary=data.get("length_summary", {}),
        dkw=data.get("dkw", {}),
        ecdf_values=data.get("ecdf_values", []),
        reference_ces_scores=data.get("reference_ces_scores", []),
        thresholds=data.get("thresholds", {}),
        known_limitations=data.get("known_limitations", []),
    )
