"""
Middleware for wrapping LLM calls with hallucination checks.

Provides:
- ``guard_output`` -- the primary gate function that scores a prompt/output
  pair against a calibration artifact and returns a ``RoutingDecision``.
  Requires real token entropy/logprobs from a provider.
- ``guard_output_from_entropies`` -- offline/batch path with pre-computed
  entropy sequences.
- ``guard_output_with_logprobs`` -- provider path with pre-fetched logprobs.
- ``guard_output_with_text_heuristic_experimental`` -- experimental text-only
  heuristic (NOT recommended for production use).
- ``SentinelMiddleware`` -- a callable wrapper that intercepts LLM outputs,
  runs the sentinel pipeline, and optionally blocks or annotates results.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np

from ..calibration import CalibrationArtifact
from ..ces import compute_ces
from ..entropy import entropy_from_topk_logprobs
from ..providers.base import CompletionLogprobs, ProviderCapabilityError, TokenLogprobResult
from ..schemas import SentinelResult
from ..thresholds import RiskLevel


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class PolicyAction(str, Enum):
    """Action the caller should take after evaluating the routing decision."""

    ALLOW = "allow"
    WARN = "warn"
    REQUIRE_EVIDENCE = "require_evidence"
    HUMAN_REVIEW = "human_review"
    BLOCK = "block"


class TaskCriticality(str, Enum):
    """How critical the downstream task is.

    A toy chatbot can tolerate LOW; a medical-records writer needs HIGH.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Diagnostic peak
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiagnosticPeak:
    """A contiguous region of locally elevated entropy.

    Labelled ``local_entropy_peak`` (never "hallucinated spans") because the
    score is a ranking signal, not a binary classification.
    """

    label: str = "local_entropy_peak"
    start_token: int = 0
    end_token: int = 0
    peak_entropy: float = 0.0
    mean_entropy: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "start_token": self.start_token,
            "end_token": self.end_token,
            "peak_entropy": round(self.peak_entropy, 6),
            "mean_entropy": round(self.mean_entropy, 6),
        }


# ---------------------------------------------------------------------------
# Routing decision
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoutingDecision:
    """Immutable result returned by ``guard_output``.

    Attributes:
        action: What the caller should do next (see :class:`PolicyAction`).
        risk_level: Discretised risk band from the CES algorithm.
        ces_score: The raw CES ranking score (not a probability).
        warnings: Human-readable diagnostic strings.
        diagnostic_peaks: Entropy regions that drove the score up, labelled
            as ``local_entropy_peak`` -- never "hallucinated spans".
        calibration_metadata: Provenance from the calibration artifact so
            callers can audit which reference distribution was used.
    """

    action: PolicyAction
    risk_level: RiskLevel
    ces_score: float
    warnings: tuple[str, ...] = ()
    diagnostic_peaks: tuple[DiagnosticPeak, ...] = ()
    calibration_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "risk_level": self.risk_level.value,
            "ces_score": round(self.ces_score, 6),
            "warnings": list(self.warnings),
            "diagnostic_peaks": [p.to_dict() for p in self.diagnostic_peaks],
            "calibration_metadata": self.calibration_metadata,
        }


# ---------------------------------------------------------------------------
# Routing table: (risk_level, criticality) -> action
# ---------------------------------------------------------------------------

# Lower criticality tasks are more permissive; higher criticality tasks
# escalate actions even at the same risk level.
_ACTION_TABLE: dict[tuple[str, str], PolicyAction] = {
    # LOW risk
    ("low", "low"): PolicyAction.ALLOW,
    ("low", "medium"): PolicyAction.ALLOW,
    ("low", "high"): PolicyAction.ALLOW,
    ("low", "critical"): PolicyAction.WARN,
    # MEDIUM risk
    ("medium", "low"): PolicyAction.ALLOW,
    ("medium", "medium"): PolicyAction.WARN,
    ("medium", "high"): PolicyAction.WARN,
    ("medium", "critical"): PolicyAction.REQUIRE_EVIDENCE,
    # HIGH risk
    ("high", "low"): PolicyAction.WARN,
    ("high", "medium"): PolicyAction.REQUIRE_EVIDENCE,
    ("high", "high"): PolicyAction.HUMAN_REVIEW,
    ("high", "critical"): PolicyAction.HUMAN_REVIEW,
    # CRITICAL risk
    ("critical", "low"): PolicyAction.REQUIRE_EVIDENCE,
    ("critical", "medium"): PolicyAction.HUMAN_REVIEW,
    ("critical", "high"): PolicyAction.BLOCK,
    ("critical", "critical"): PolicyAction.BLOCK,
}

# UNKNOWN risk always escalates to at least WARN.
_ACTION_TABLE[("unknown", "low")] = PolicyAction.WARN
_ACTION_TABLE[("unknown", "medium")] = PolicyAction.REQUIRE_EVIDENCE
_ACTION_TABLE[("unknown", "high")] = PolicyAction.HUMAN_REVIEW
_ACTION_TABLE[("unknown", "critical")] = PolicyAction.BLOCK


def _resolve_action(
    risk_level: RiskLevel,
    criticality: TaskCriticality,
) -> PolicyAction:
    """Look up the routing action from the risk/criticality table."""
    key = (risk_level.value, criticality.value)
    return _ACTION_TABLE.get(key, PolicyAction.HUMAN_REVIEW)


# ---------------------------------------------------------------------------
# Diagnostic peak detection
# ---------------------------------------------------------------------------

_PEAK_THRESHOLD_FACTOR = 1.5  # entropy > 1.5x local mean => peak


def _detect_peaks(
    entropies: np.ndarray,
    threshold_factor: float = _PEAK_THRESHOLD_FACTOR,
) -> list[DiagnosticPeak]:
    """Find contiguous runs of locally elevated entropy.

    A token is considered part of a peak when its entropy exceeds
    ``threshold_factor * median(entropies)``.  Adjacent flagged tokens
    are merged into a single peak region.
    """
    if entropies.size == 0:
        return []

    median_ent = float(np.median(entropies))
    if median_ent < 1e-12:
        # All tokens have near-zero entropy; nothing to flag.
        return []

    threshold = threshold_factor * median_ent
    flagged = entropies > threshold

    peaks: list[DiagnosticPeak] = []
    start: Optional[int] = None

    for i, is_flagged in enumerate(flagged):
        if is_flagged and start is None:
            start = i
        elif not is_flagged and start is not None:
            region = entropies[start:i]
            peaks.append(DiagnosticPeak(
                start_token=start,
                end_token=i - 1,
                peak_entropy=float(np.max(region)),
                mean_entropy=float(np.mean(region)),
            ))
            start = None

    # Flush final run
    if start is not None:
        region = entropies[start:]
        peaks.append(DiagnosticPeak(
            start_token=start,
            end_token=len(entropies) - 1,
            peak_entropy=float(np.max(region)),
            mean_entropy=float(np.mean(region)),
        ))

    return peaks


# ---------------------------------------------------------------------------
# Calibration metadata extraction
# ---------------------------------------------------------------------------


def _calibration_metadata(artifact: CalibrationArtifact) -> dict[str, Any]:
    """Extract audit-friendly provenance from a calibration artifact."""
    return {
        "schema_version": artifact.schema_version,
        "created_at": artifact.created_at,
        "model": artifact.model,
        "provider": artifact.provider,
        "task_family": artifact.task_family,
        "calibration_mode": artifact.calibration_mode,
        "token_count": artifact.token_count,
        "sequence_count": artifact.sequence_count,
        "entropy_mode": artifact.entropy_mode,
        "entropy_base": artifact.entropy_base,
        "thresholds": dict(artifact.thresholds),
        "dkw": dict(artifact.dkw),
    }


# ---------------------------------------------------------------------------
# guard_output -- the primary gate function
# ---------------------------------------------------------------------------


def guard_output(
    prompt: str,
    output: str,
    *,
    calibration: CalibrationArtifact,
    provider: str = "unknown",
    policy: TaskCriticality = TaskCriticality.MEDIUM,
) -> RoutingDecision:
    """Score *output* against a calibration artifact and decide what to do.

    This is the main entry-point for the middleware layer.  Call it after
    you already have both the prompt and the model output (e.g. from a
    RAG pipeline, an agent tool call, or a batch QA run).

    **Important**: This function requires real token entropy/logprobs from
    a provider. It will raise :class:`ProviderCapabilityError` if called
    without a provider that supports logprob extraction. For offline/batch
    scoring with pre-computed entropies, use :func:`guard_output_from_entropies`.
    For experimental text-only heuristic scoring, use
    :func:`guard_output_with_text_heuristic_experimental`.

    Args:
        prompt: The prompt that produced *output*.
        output: The model-generated text to evaluate.
        calibration: A :class:`CalibrationArtifact` (loaded or freshly built).
        provider: Name of the LLM provider that generated *output*.
            Must be a real provider that supports logprob extraction.
        policy: How critical the downstream task is.  A ``MEDIUM`` policy
            with ``HIGH`` risk yields ``REQUIRE_EVIDENCE``; the same risk
            with ``CRITICAL`` policy yields ``HUMAN_REVIEW``.

    Returns:
        A :class:`RoutingDecision` with the recommended action, risk level,
        CES score, warnings, diagnostic peaks, and calibration metadata.

    Raises:
        ValueError: If *output* is empty or *calibration* has no ECDF values.
        ProviderCapabilityError: If provider does not support logprob extraction.
    """
    if not output or not output.strip():
        raise ValueError("output must be a non-empty string")

    # guard_output requires real logprobs from a provider. Text-only heuristic
    # is no longer used in production flow. Use guard_output_from_entropies()
    # for offline/batch scoring, or guard_output_with_logprobs() when you
    # already have logprob data from a provider call.
    raise ProviderCapabilityError(
        "guard_output() requires real token entropy/logprobs from a provider. "
        "Text-only heuristic scoring is not supported in the production flow. "
        "Use guard_output_from_entropies() for offline/batch scoring with "
        "pre-computed entropy sequences, or guard_output_with_logprobs() "
        "when you already have logprob data from a provider call. "
        "For experimental text-only heuristic, use "
        "guard_output_with_text_heuristic_experimental().",
        capability="logprobs",
        provider=provider,
    )


# ---------------------------------------------------------------------------
# guard_output_with_logprobs -- variant when caller already has logprobs
# ---------------------------------------------------------------------------


def guard_output_with_logprobs(
    prompt: str,
    output: str,
    logprobs: CompletionLogprobs,
    *,
    calibration: CalibrationArtifact,
    provider: str = "unknown",
    policy: TaskCriticality = TaskCriticality.MEDIUM,
) -> RoutingDecision:
    """Score *output* using pre-fetched logprobs and decide what to do.

    Use this variant when you already have a :class:`CompletionLogprobs`
    object from a provider call (e.g. via ``BaseProvider.score_text``).
    This avoids a redundant API round-trip.

    Args:
        prompt: The prompt that produced *output*.
        output: The model-generated text (used for peak detection only).
        logprobs: Pre-fetched logprob data from a provider.
        calibration: A :class:`CalibrationArtifact`.
        provider: Provider name for metadata.
        policy: Task criticality level.

    Returns:
        A :class:`RoutingDecision`.
    """
    # Compute entropy from logprobs
    if logprobs.has_top_k():
        topk = logprobs.topk_logprobs
        entropy_result = entropy_from_topk_logprobs(topk, logprobs.top_k)
        entropies = entropy_result.entropies
    else:
        # Fallback: use selected-token logprobs as a proxy
        selected = np.array(logprobs.selected_logprobs, dtype=np.float64)
        entropies = -selected  # negative logprob ≈ surprise

    ces_result = compute_ces(entropies, calibration)
    risk_level = RiskLevel(ces_result.risk_level.lower())
    peaks = _detect_peaks(entropies)
    action = _resolve_action(risk_level, policy)

    all_warnings = list(ces_result.warnings)
    if provider and provider != "unknown":
        all_warnings.append(f"provider={provider}")

    return RoutingDecision(
        action=action,
        risk_level=risk_level,
        ces_score=ces_result.ces_score,
        warnings=tuple(all_warnings),
        diagnostic_peaks=tuple(peaks),
        calibration_metadata=_calibration_metadata(calibration),
    )


# ---------------------------------------------------------------------------
# guard_output_from_entropies -- variant with pre-computed entropy sequence
# ---------------------------------------------------------------------------


def guard_output_from_entropies(
    prompt: str,
    output: str,
    entropies: np.ndarray,
    *,
    calibration: CalibrationArtifact,
    provider: str = "unknown",
    policy: TaskCriticality = TaskCriticality.MEDIUM,
) -> RoutingDecision:
    """Score *output* from a pre-computed entropy sequence.

    Use this when you already have per-token entropy values (e.g. from a
    custom provider or a batch scoring pipeline).

    Args:
        prompt: The prompt that produced *output*.
        output: The model-generated text.
        entropies: 1-D array of per-token entropy values (nats).
        calibration: A :class:`CalibrationArtifact`.
        provider: Provider name for metadata.
        policy: Task criticality level.

    Returns:
        A :class:`RoutingDecision`.
    """
    entropies = np.asarray(entropies, dtype=np.float64).ravel()
    if entropies.size == 0:
        raise ValueError("entropies must be non-empty")

    ces_result = compute_ces(entropies, calibration)
    risk_level = RiskLevel(ces_result.risk_level.lower())
    peaks = _detect_peaks(entropies)
    action = _resolve_action(risk_level, policy)

    all_warnings = list(ces_result.warnings)
    if provider and provider != "unknown":
        all_warnings.append(f"provider={provider}")

    return RoutingDecision(
        action=action,
        risk_level=risk_level,
        ces_score=ces_result.ces_score,
        warnings=tuple(all_warnings),
        diagnostic_peaks=tuple(peaks),
        calibration_metadata=_calibration_metadata(calibration),
    )


# ---------------------------------------------------------------------------
# Text-only entropy scoring (EXPERIMENTAL - NOT for production use)
# ---------------------------------------------------------------------------


def _score_text_entropies(
    text: str,
) -> np.ndarray:
    """Produce per-token entropy values for *text* using a character-level heuristic.

    .. warning::
        This is an **experimental** heuristic that estimates entropy from
        token length, digit presence, and capitalization patterns. It does
        NOT produce meaningful uncertainty estimates and should NOT be used
        in production. For real CES scoring, use providers with logprob support.

    The heuristic assigns higher entropy to tokens that are rare or
    unusual-looking (long, mixed-case, containing digits).  This is a
    *very* rough proxy and is NOT a substitute for real token entropy.
    """
    # Tokenise naively by whitespace
    tokens = text.split()
    if not tokens:
        return np.array([0.0])

    # Simple heuristic entropy: longer, more varied tokens get higher scores.
    entropies = np.zeros(len(tokens), dtype=np.float64)
    for i, tok in enumerate(tokens):
        length_factor = min(len(tok) / 10.0, 3.0)
        has_digit = any(c.isdigit() for c in tok)
        has_upper = any(c.isupper() for c in tok)
        has_lower = any(c.islower() for c in tok)
        diversity = sum([has_digit, has_upper, has_lower]) / 3.0
        entropies[i] = length_factor * (0.5 + diversity)

    return entropies


def guard_output_with_text_heuristic_experimental(
    prompt: str,
    output: str,
    *,
    calibration: CalibrationArtifact,
    provider: str = "unknown",
    policy: TaskCriticality = TaskCriticality.MEDIUM,
) -> RoutingDecision:
    """Score *output* using an experimental text-only heuristic.

    .. danger::
        This function uses a **character-level heuristic** that estimates
        entropy from token length, digits, and capitalization. It does NOT
        produce meaningful uncertainty estimates and should NOT be used in
        production. This exists only for offline experimentation and testing.

        For production use, use:
        - :func:`guard_output_from_entropies` with real entropy sequences
        - :func:`guard_output_with_logprobs` with provider logprob data

    The heuristic assigns higher entropy to tokens that are rare or
    unusual-looking (long, mixed-case, containing digits). This is a
    *very* rough proxy; CES scores from this function are NOT comparable
    to scores from real entropy data.

    Args:
        prompt: The prompt that produced *output*.
        output: The model-generated text to evaluate.
        calibration: A :class:`CalibrationArtifact` (loaded or freshly built).
        provider: Name of the LLM provider (for metadata only).
        policy: How critical the downstream task is.

    Returns:
        A :class:`RoutingDecision` (but scores are unreliable).

    Raises:
        ValueError: If *output* is empty or *calibration* has no ECDF values.
    """
    warnings.warn(
        "guard_output_with_text_heuristic_experimental() uses a character-level "
        "heuristic that does NOT produce meaningful uncertainty estimates. "
        "CES scores from this function are NOT comparable to scores from real "
        "entropy data. Use guard_output_from_entropies() or "
        "guard_output_with_logprobs() for production scoring.",
        UserWarning,
        stacklevel=2,
    )

    if not output or not output.strip():
        raise ValueError("output must be a non-empty string")

    entropies = _score_text_entropies(output)

    ces_result = compute_ces(entropies, calibration)
    risk_level = RiskLevel(ces_result.risk_level.lower())
    peaks = _detect_peaks(entropies)
    action = _resolve_action(risk_level, policy)

    all_warnings: list[str] = list(ces_result.warnings)
    all_warnings.append("EXPERIMENTAL: text-only heuristic (not real entropy)")
    if provider and provider != "unknown":
        all_warnings.append(f"provider={provider}")

    return RoutingDecision(
        action=action,
        risk_level=risk_level,
        ces_score=ces_result.ces_score,
        warnings=tuple(all_warnings),
        diagnostic_peaks=tuple(peaks),
        calibration_metadata=_calibration_metadata(calibration),
    )


# ---------------------------------------------------------------------------
# SentinelMiddleware (callable wrapper, backward-compatible)
# ---------------------------------------------------------------------------


class HallucinationBlockedError(Exception):
    """Raised when a generation is blocked due to critical hallucination risk."""

    def __init__(self, result: RoutingDecision):
        self.result = result
        super().__init__(
            f"Hallucination blocked: risk={result.risk_level.value}, "
            f"action={result.action.value}, ces_score={result.ces_score:.4f}"
        )


@dataclass
class SentinelMiddleware:
    """Wraps an LLM callable to automatically check outputs for hallucinations.

    Usage::

        middleware = SentinelMiddleware(
            llm_call=my_llm_function,
            calibration=load_calibration("calibration.json"),
            policy=TaskCriticality.HIGH,
        )
        result = middleware("What is the capital of France?")

    The wrapper calls the LLM, scores the output, and either returns the
    text unchanged, annotates it with warnings, or raises
    :class:`HallucinationBlockedError` depending on the routing decision.

    .. note::
        This middleware uses the experimental text-only heuristic for scoring.
        For production use with real entropy/logprobs, use
        :func:`guard_output_from_entropies` or :func:`guard_output_with_logprobs`
        directly.
    """

    llm_call: Callable[..., str]
    calibration: CalibrationArtifact
    provider: str = "unknown"
    policy: TaskCriticality = TaskCriticality.MEDIUM
    on_decision: Optional[Callable[[RoutingDecision], None]] = None

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        """Call LLM and gate the result using experimental text-only heuristic.

        Returns the LLM response text unchanged (even when warnings are
        emitted).  Raises :class:`HallucinationBlockedError` when the
        routing action is ``BLOCK``.

        .. warning::
            Uses experimental text-only heuristic. For production, use
            :func:`guard_output_from_entropies` or
            :func:`guard_output_with_logprobs` directly.

        Args:
            prompt: The prompt to send to the LLM.
            **kwargs: Additional arguments passed to the LLM call.

        Returns:
            The LLM response text.

        Raises:
            HallucinationBlockedError: If the routing action is ``BLOCK``.
        """
        response = self.llm_call(prompt, **kwargs)

        decision = guard_output_with_text_heuristic_experimental(
            prompt,
            response,
            calibration=self.calibration,
            provider=self.provider,
            policy=self.policy,
        )

        # Notify the caller if a callback was registered.
        if self.on_decision is not None:
            self.on_decision(decision)

        if decision.action is PolicyAction.BLOCK:
            raise HallucinationBlockedError(decision)

        return response
