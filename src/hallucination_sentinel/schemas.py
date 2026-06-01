"""
Data schemas for Hallucination Sentinel.

Defines the structured request/response types used across the package.
All types are frozen dataclasses for immutability.
"""

from dataclasses import dataclass, field
from typing import Optional

from .thresholds import RiskLevel


# ---------------------------------------------------------------------------
# Label parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParsedLabel:
    """Canonical label representation.

    Exactly one of ``faithful`` or ``hallucinated`` is True.
    """

    faithful: bool
    hallucinated: bool


def parse_label(value: object, *, context: str = "") -> ParsedLabel:
    """Parse a label value into canonical faithful/hallucinated semantics.

    Mapping rules (checked in order):

    * **bool** -- ``True`` → faithful, ``False`` → hallucinated.
      Must be checked *before* int because Python ``isinstance(True, int)``
      is ``True``.
    * **int** -- ``0`` → faithful, ``1`` → hallucinated.
    * **str** -- ``"faithful"`` → faithful, ``"hallucinated"`` → hallucinated
      (case-insensitive, whitespace stripped).

    Args:
        value: Raw label from JSONL or CLI.
        context: Optional diagnostic string included in error messages.

    Returns:
        A :class:`ParsedLabel` with exactly one flag set.

    Raises:
        ValueError: If *value* cannot be mapped.
    """
    # bool MUST come before int -- isinstance(True, int) is True in Python.
    if isinstance(value, bool):
        return ParsedLabel(faithful=value, hallucinated=not value)

    if isinstance(value, int):
        if value == 0:
            return ParsedLabel(faithful=True, hallucinated=False)
        if value == 1:
            return ParsedLabel(faithful=False, hallucinated=True)
        raise ValueError(
            f"Invalid label integer {value!r}"
            f"{' (' + context + ')' if context else ''}; expected 0 or 1"
        )

    if isinstance(value, str):
        lower = value.strip().lower()
        if lower == "faithful":
            return ParsedLabel(faithful=True, hallucinated=False)
        if lower == "hallucinated":
            return ParsedLabel(faithful=False, hallucinated=True)
        raise ValueError(
            f"Invalid label string {value!r}"
            f"{' (' + context + ')' if context else ''}; expected 'faithful' or 'hallucinated'"
        )

    raise ValueError(
        f"Unsupported label type {type(value).__name__}: {value!r}"
        f"{' (' + context + ')' if context else ''}"
    )


@dataclass(frozen=True)
class TokenScore:
    """Per-token risk assessment."""

    index: int
    token: str
    entropy: float
    is_flagged: bool


@dataclass(frozen=True)
class SentinelResult:
    """Complete result of a hallucination assessment."""

    ces_score: float
    calibrated_probability: float
    risk_level: RiskLevel
    token_count: int
    token_entropies: tuple[float, ...]
    flagged_tokens: tuple[TokenScore, ...] = ()
    warnings: tuple[str, ...] = ()
    provider: Optional[str] = None
    model: Optional[str] = None

    @property
    def is_hallucination(self) -> bool:
        """True if risk level is HIGH or CRITICAL."""
        return self.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)

    @property
    def should_block(self) -> bool:
        """True if risk level is CRITICAL (action should be blocked)."""
        return self.risk_level == RiskLevel.CRITICAL
