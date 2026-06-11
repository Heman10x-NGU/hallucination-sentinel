"""Tests for the Hallucination Sentinel MCP tool layer."""

from __future__ import annotations

import numpy as np
import pytest

from hallucination_sentinel.calibration import build_calibration, save_calibration
from hallucination_sentinel.mcp_server import (
    RISK_SIGNAL_NOTE,
    inspect_calibration_tool,
    score_entropy_sequence_tool,
    score_provider_output_tool,
    score_topk_logprobs_tool,
    smoke_provider_tool,
)
from hallucination_sentinel.providers.base import CompletionLogprobs, TokenLogprob


def _calibration_path(tmp_path):
    """Create a reusable calibration artifact for MCP tests."""
    rng = np.random.RandomState(42)
    sequences = [rng.uniform(0.1, 2.0, size=30) for _ in range(25)]
    artifact = build_calibration(
        sequences,
        mode="unsupervised",
        model="test-model",
        provider="test-provider",
        task_family="unit-test",
    )
    from hallucination_sentinel.thresholds import assign_thresholds

    assign_thresholds(artifact)
    path = tmp_path / "calibration.json"
    save_calibration(artifact, path)
    return path


def test_score_entropy_sequence_tool_returns_structured_payload(tmp_path):
    path = _calibration_path(tmp_path)

    payload = score_entropy_sequence_tool(
        entropies=[0.2, 0.4, 0.8, 1.6, 1.2, 0.5, 0.3, 0.7, 0.9, 1.1],
        calibration_path=str(path),
        provider="unit-test",
        policy="high",
    )

    assert payload["status"] == "ok"
    assert payload["source"] == "entropy_sequence"
    assert 0.0 <= payload["ces_score"] <= 1.0
    assert payload["risk_level"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL", "UNKNOWN"}
    assert payload["action"] in {"allow", "warn", "require_evidence", "human_review", "block"}
    assert payload["token_count"] == 10
    assert payload["mean_entropy"] > 0
    assert payload["max_entropy"] > 0
    assert payload["risk_signal_note"] == RISK_SIGNAL_NOTE
    assert payload["calibration"]["model"] == "test-model"


def test_score_entropy_sequence_rejects_empty_input(tmp_path):
    path = _calibration_path(tmp_path)

    with pytest.raises(ValueError, match="at least one"):
        score_entropy_sequence_tool([], str(path))


def test_score_entropy_sequence_rejects_nan(tmp_path):
    path = _calibration_path(tmp_path)

    with pytest.raises(ValueError, match="finite"):
        score_entropy_sequence_tool([0.1, float("nan")], str(path))


def test_score_topk_logprobs_tool_computes_entropy_and_score(tmp_path):
    path = _calibration_path(tmp_path)
    topk = [
        {" A": -0.1, " B": -2.0, " C": -3.0},
        {" cat": -0.3, " dog": -1.8, " bird": -2.5},
        {" sat": -0.5, " slept": -1.1, " ran": -2.0},
    ]

    payload = score_topk_logprobs_tool(
        topk_logprobs=topk,
        calibration_path=str(path),
        provider="mock-provider",
        policy="medium",
    )

    assert payload["status"] == "ok"
    assert payload["source"] == "topk_logprobs"
    assert payload["entropy_mode"] in {"top_k", "top_k_with_residual"}
    assert payload["provider"] == "mock-provider"
    assert payload["token_count"] == 3
    assert payload["risk_signal_note"] == RISK_SIGNAL_NOTE


def test_score_topk_logprobs_tool_rejects_selected_only_positions(tmp_path):
    path = _calibration_path(tmp_path)

    with pytest.raises(ValueError, match="empty"):
        score_topk_logprobs_tool([{}], str(path))


def test_score_provider_output_tool_uses_provider_logprobs(monkeypatch, tmp_path):
    path = _calibration_path(tmp_path)

    class FakeProvider:
        @classmethod
        def from_preset(cls, provider, api_key=None, model=None, base_url=None):
            assert provider == "together"
            assert model == "fake-model"
            return cls()

        def score_output(self, prompt, output, top_k=None):
            assert prompt == "Question?"
            assert output == "Answer."
            return CompletionLogprobs(
                tokens=[
                    TokenLogprob(
                        token="Answer",
                        logprob=-0.2,
                        top_logprobs={"Answer": -0.2, "Reply": -1.7},
                    ),
                    TokenLogprob(
                        token=".",
                        logprob=-0.1,
                        top_logprobs={".": -0.1, "!": -2.0},
                    ),
                ],
                provider="Together AI",
                model="fake-model",
                top_k=2,
                echo_used=True,
            )

    monkeypatch.setattr(
        "hallucination_sentinel.mcp_server.OpenAICompatibleProvider",
        FakeProvider,
    )

    payload = score_provider_output_tool(
        prompt="Question?",
        output="Answer.",
        provider="together",
        model="fake-model",
        calibration_path=str(path),
    )

    assert payload["status"] == "ok"
    assert payload["source"] == "provider_output"
    assert payload["provider"] == "together"
    assert payload["model"] == "fake-model"
    assert payload["token_count"] == 2
    assert payload["risk_signal_note"] == RISK_SIGNAL_NOTE


def test_score_provider_output_rejects_plain_text_without_prompt(tmp_path):
    path = _calibration_path(tmp_path)

    with pytest.raises(ValueError, match="prompt"):
        score_provider_output_tool(
            prompt="",
            output="Answer.",
            provider="together",
            model="fake-model",
            calibration_path=str(path),
        )


def test_inspect_calibration_tool_returns_metadata(tmp_path):
    path = _calibration_path(tmp_path)

    payload = inspect_calibration_tool(str(path))

    assert payload["status"] == "ok"
    assert payload["model"] == "test-model"
    assert payload["provider"] == "test-provider"
    assert payload["task_family"] == "unit-test"
    assert payload["ecdf_size"] > 0
    assert payload["risk_signal_note"] == RISK_SIGNAL_NOTE


def test_smoke_provider_tool_wraps_health_check(monkeypatch):
    class FakeProvider:
        @classmethod
        def from_preset(cls, provider, api_key=None, model=None, base_url=None):
            assert provider == "openai"
            return cls()

        def check_health(self):
            return {
                "healthy": True,
                "provider": "OpenAI",
                "model": "fake-model",
                "top_k_available": True,
                "token_count": 2,
            }

    monkeypatch.setattr(
        "hallucination_sentinel.mcp_server.OpenAICompatibleProvider",
        FakeProvider,
    )

    payload = smoke_provider_tool("openai", model="fake-model")

    assert payload["healthy"] is True
    assert payload["top_k_available"] is True
    assert payload["risk_signal_note"] == RISK_SIGNAL_NOTE
