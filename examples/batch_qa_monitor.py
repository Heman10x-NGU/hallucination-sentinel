"""
Batch QA Monitor
================

Shows how to use ``guard_output_from_entropies`` to monitor a batch of
question-answer pairs and produce a summary report.

The pattern:
  1. Load a batch of (question, answer, entropy_sequence) triples.
  2. Gate each answer through the sentinel.
  3. Aggregate the routing decisions into a summary report.

This is useful for quality-assurance pipelines that evaluate a model's
outputs offline, e.g. before deploying a new model version.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from hallucination_sentinel.calibration import load_calibration
from hallucination_sentinel.integrations.middleware import (
    PolicyAction,
    RoutingDecision,
    TaskCriticality,
    guard_output_from_entropies,
)


# ---------------------------------------------------------------------------
# Batch record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QARecord:
    """A single question-answer pair with pre-computed entropy."""

    question: str
    answer: str
    entropies: tuple[float, ...]


def load_batch(path: str | Path) -> list[QARecord]:
    """Load a batch of QA records from a JSONL file.

    Each line must have:
      - ``question`` (str)
      - ``answer`` (str)
      - ``entropies`` (list[float])

    Args:
        path: Path to the JSONL file.

    Returns:
        A list of :class:`QARecord`.
    """
    path = Path(path)
    records: list[QARecord] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records.append(QARecord(
                question=obj["question"],
                answer=obj["answer"],
                entropies=tuple(float(v) for v in obj["entropies"]),
            ))
    return records


# ---------------------------------------------------------------------------
# Batch monitor
# ---------------------------------------------------------------------------


@dataclass
class BatchSummary:
    """Aggregated results from a batch QA monitoring run."""

    total: int = 0
    action_counts: dict[str, int] = field(default_factory=dict)
    risk_counts: dict[str, int] = field(default_factory=dict)
    mean_ces: float = 0.0
    max_ces: float = 0.0
    blocked: list[dict[str, Any]] = field(default_factory=list)
    flagged: list[dict[str, Any]] = field(default_factory=list)
    all_decisions: list[dict[str, Any]] = field(default_factory=list)


def monitor_batch(
    records: list[QARecord],
    calibration_path: str,
    criticality: TaskCriticality = TaskCriticality.MEDIUM,
) -> BatchSummary:
    """Gate every record in *records* and aggregate the results.

    Args:
        records: QA records with pre-computed entropy sequences.
        calibration_path: Path to a calibration artifact JSON file.
        criticality: Task criticality for routing decisions.

    Returns:
        A :class:`BatchSummary` with per-action/risk counts and lists of
        blocked and flagged records.
    """
    calibration = load_calibration(calibration_path)

    summary = BatchSummary()
    ces_scores: list[float] = []

    for rec in records:
        ent = np.array(rec.entropies, dtype=np.float64)

        decision = guard_output_from_entropies(
            rec.question,
            rec.answer,
            ent,
            calibration=calibration,
            policy=criticality,
        )

        ces_scores.append(decision.ces_score)

        # Count by action
        action_key = decision.action.value
        summary.action_counts[action_key] = summary.action_counts.get(action_key, 0) + 1

        # Count by risk level
        risk_key = decision.risk_level.value
        summary.risk_counts[risk_key] = summary.risk_counts.get(risk_key, 0) + 1

        # Collect blocked and flagged records
        entry = {
            "question": rec.question,
            "answer": rec.answer,
            "decision": decision.to_dict(),
        }
        summary.all_decisions.append(entry)

        if decision.action is PolicyAction.BLOCK:
            summary.blocked.append(entry)
        elif decision.action in (
            PolicyAction.HUMAN_REVIEW,
            PolicyAction.REQUIRE_EVIDENCE,
        ):
            summary.flagged.append(entry)

    summary.total = len(records)
    if ces_scores:
        summary.mean_ces = float(np.mean(ces_scores))
        summary.max_ces = float(np.max(ces_scores))

    return summary


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------


def _make_demo_records() -> list[QARecord]:
    """Create a small set of synthetic QA records for demonstration."""
    return [
        QARecord(
            question="What is the capital of France?",
            answer="The capital of France is Paris.",
            entropies=(0.1, 0.2, 0.15, 0.3, 0.1, 0.2),
        ),
        QARecord(
            question="Who wrote Hamlet?",
            answer="Hamlet was written by William Shakespeare.",
            entropies=(0.1, 0.3, 0.2, 0.15, 0.1, 0.25),
        ),
        QARecord(
            question="What is the speed of light?",
            answer=(
                "The speed of light in vacuum is approximately "
                "299,792,458 metres per second."
            ),
            # Simulated high-entropy region (uncertain middle section)
            entropies=(0.1, 0.2, 3.5, 4.0, 3.8, 0.2, 0.1, 0.15),
        ),
        QARecord(
            question="Explain quantum entanglement.",
            answer=(
                "Quantum entanglement is a phenomenon where two particles "
                "become correlated such that the quantum state of one "
                "instantly influences the other, regardless of distance."
            ),
            # High overall entropy (model is uncertain)
            entropies=(2.0, 2.5, 3.0, 2.8, 3.2, 2.1, 2.9, 3.1, 2.4, 2.7),
        ),
    ]


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python batch_qa_monitor.py <calibration.json> [batch.jsonl]")
        print()
        print("If batch.jsonl is omitted, a built-in demo set is used.")
        sys.exit(1)

    calibration_path = sys.argv[1]

    if len(sys.argv) >= 3:
        records = load_batch(sys.argv[2])
    else:
        records = _make_demo_records()

    summary = monitor_batch(
        records,
        calibration_path,
        criticality=TaskCriticality.MEDIUM,
    )

    output = {
        "total": summary.total,
        "mean_ces": round(summary.mean_ces, 6),
        "max_ces": round(summary.max_ces, 6),
        "action_counts": summary.action_counts,
        "risk_counts": summary.risk_counts,
        "blocked_count": len(summary.blocked),
        "flagged_count": len(summary.flagged),
        "blocked": summary.blocked,
        "flagged": summary.flagged,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
