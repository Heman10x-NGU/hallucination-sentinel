"""MCP server for Hallucination Sentinel.

The MCP layer intentionally does not score plain text by heuristic.  Every
scoring tool requires real entropy values, top-k logprobs, or a provider call
that returns logprobs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .calibration import CalibrationArtifact, load_calibration
from .ces import CESResult, compute_ces
from .entropy import entropy_from_topk_logprobs
from .integrations import TaskCriticality, guard_output_from_entropies
from .providers.base import ProviderCapabilityError
from .providers.openai_compatible import (
    OpenAICompatibleProvider,
    PROVIDER_SPECS,
)


RISK_SIGNAL_NOTE = (
    "CES is a calibrated entropy risk signal, not a truth oracle. "
    "A high score does not prove hallucination, and a low score does not prove correctness."
)


def _round_float(value: float) -> float:
    """Round finite floats for stable JSON output."""
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("MCP responses cannot contain NaN or infinite values")
    return round(value, 6)


def _load_calibration(calibration_path: str) -> CalibrationArtifact:
    """Load a calibration artifact with a clear MCP-facing error."""
    path = Path(calibration_path).expanduser()
    if not path.exists():
        raise ValueError(f"calibration_path does not exist: {path}")
    return load_calibration(path)


def _policy_from_string(policy: str) -> TaskCriticality:
    """Parse a policy string into a TaskCriticality enum."""
    try:
        return TaskCriticality(policy.lower())
    except ValueError as exc:
        valid = ", ".join(p.value for p in TaskCriticality)
        raise ValueError(f"Unknown policy '{policy}'. Valid values: {valid}.") from exc


def _result_payload(
    result: CESResult,
    *,
    source: str,
    calibration: CalibrationArtifact,
    entropy_mode: str,
    provider: str = "unknown",
    model: str = "",
    action: str = "",
    extra_warnings: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Build the structured payload returned by MCP scoring tools."""
    warnings = list(result.warnings)
    if extra_warnings:
        warnings.extend(extra_warnings)

    payload: dict[str, Any] = {
        "status": "ok",
        "source": source,
        "ces_score": _round_float(result.ces_score),
        "risk_level": result.risk_level,
        "cdf_mean": _round_float(result.cdf_mean),
        "cdf_max": _round_float(result.cdf_max),
        "mean_entropy": _round_float(result.mean_entropy),
        "max_entropy": _round_float(result.max_entropy),
        "token_count": result.token_count,
        "warnings": warnings,
        "risk_signal_note": RISK_SIGNAL_NOTE,
        "entropy_mode": entropy_mode,
        "provider": provider,
        "model": model,
        "calibration": {
            "model": calibration.model,
            "provider": calibration.provider,
            "task_family": calibration.task_family,
            "calibration_mode": calibration.calibration_mode,
            "sequence_count": calibration.sequence_count,
            "token_count": calibration.token_count,
            "thresholds": calibration.thresholds,
        },
    }
    if action:
        payload["action"] = action
    return payload


def score_entropy_sequence_tool(
    entropies: list[float],
    calibration_path: str,
    provider: str = "mcp",
    policy: str = "medium",
) -> dict[str, Any]:
    """Score a pre-computed entropy sequence against a calibration artifact."""
    arr = np.asarray(entropies, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError("entropies must contain at least one value")
    if np.any(~np.isfinite(arr)):
        raise ValueError("entropies must contain only finite numbers")

    calibration = _load_calibration(calibration_path)
    ces_result = compute_ces(arr, calibration)
    decision = guard_output_from_entropies(
        prompt="",
        output="entropy sequence",
        entropies=arr,
        calibration=calibration,
        provider=provider,
        policy=_policy_from_string(policy),
    )

    return _result_payload(
        ces_result,
        source="entropy_sequence",
        calibration=calibration,
        entropy_mode="precomputed",
        provider=provider,
        action=decision.action.value,
    )


def score_topk_logprobs_tool(
    topk_logprobs: list[dict[str, float]],
    calibration_path: str,
    top_k: Optional[int] = None,
    provider: str = "mcp",
    policy: str = "medium",
    add_residual: bool = True,
) -> dict[str, Any]:
    """Convert top-k logprobs to entropy and score the resulting sequence."""
    if not topk_logprobs:
        raise ValueError("topk_logprobs must contain at least one token position")
    for idx, position in enumerate(topk_logprobs, start=1):
        if not position:
            raise ValueError(f"topk_logprobs position {idx} is empty")
        for token, logprob in position.items():
            if not token:
                raise ValueError(f"topk_logprobs position {idx} has an empty token")
            if not np.isfinite(float(logprob)):
                raise ValueError(f"logprob for token {token!r} at position {idx} is not finite")

    inferred_top_k = top_k or max(len(pos) for pos in topk_logprobs)
    entropy_result = entropy_from_topk_logprobs(
        topk_logprobs,
        top_k=inferred_top_k,
        add_residual=add_residual,
    )
    if entropy_result.entropies is None:
        raise ValueError("top-k logprobs did not produce per-token entropies")

    calibration = _load_calibration(calibration_path)
    ces_result = compute_ces(entropy_result.entropies, calibration)
    decision = guard_output_from_entropies(
        prompt="",
        output="top-k logprobs",
        entropies=entropy_result.entropies,
        calibration=calibration,
        provider=provider,
        policy=_policy_from_string(policy),
    )

    return _result_payload(
        ces_result,
        source="topk_logprobs",
        calibration=calibration,
        entropy_mode=entropy_result.entropy_mode,
        provider=provider,
        action=decision.action.value,
        extra_warnings=entropy_result.warnings,
    )


def score_provider_output_tool(
    prompt: str,
    output: str,
    provider: str,
    model: str,
    calibration_path: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    top_k: Optional[int] = None,
    policy: str = "medium",
) -> dict[str, Any]:
    """Score an output with provider-returned top-k logprobs.

    This requires a provider that supports echo or teacher-forced output
    scoring.  The MCP server never claims to recover logprobs from plain text.
    """
    if not prompt.strip():
        raise ValueError("prompt must be non-empty")
    if not output.strip():
        raise ValueError("output must be non-empty")

    provider_client = OpenAICompatibleProvider.from_preset(
        provider,
        api_key=api_key,
        model=model,
        base_url=base_url,
    )
    completion = provider_client.score_output(prompt, output, top_k=top_k)
    if not completion.has_top_k():
        raise ProviderCapabilityError(
            "Provider returned selected-token logprobs only. CES requires top-k "
            "logprobs or full logits to compute entropy.",
            capability="top_k_logprobs",
            provider=provider,
        )

    entropy_result = entropy_from_topk_logprobs(
        completion.topk_logprobs,
        top_k=completion.top_k,
    )
    if entropy_result.entropies is None:
        raise ValueError("provider logprobs did not produce per-token entropies")

    calibration = _load_calibration(calibration_path)
    ces_result = compute_ces(entropy_result.entropies, calibration)
    decision = guard_output_from_entropies(
        prompt=prompt,
        output=output,
        entropies=entropy_result.entropies,
        calibration=calibration,
        provider=provider,
        policy=_policy_from_string(policy),
    )

    return _result_payload(
        ces_result,
        source="provider_output",
        calibration=calibration,
        entropy_mode=entropy_result.entropy_mode,
        provider=provider,
        model=completion.model,
        action=decision.action.value,
        extra_warnings=entropy_result.warnings,
    )


def smoke_provider_tool(
    provider: str,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    max_top_k: Optional[int] = None,
) -> dict[str, Any]:
    """Check whether a provider can return logprobs for Sentinel scoring."""
    if provider == "custom":
        if not base_url or not model:
            raise ValueError("custom provider requires base_url and model")
        provider_client = OpenAICompatibleProvider.custom(
            base_url=base_url,
            model=model,
            api_key=api_key or "",
            max_top_k=max_top_k or 20,
        )
    else:
        if provider not in PROVIDER_SPECS:
            valid = ", ".join([*PROVIDER_SPECS.keys(), "custom"])
            raise ValueError(f"Unknown provider '{provider}'. Valid values: {valid}.")
        provider_client = OpenAICompatibleProvider.from_preset(
            provider,
            api_key=api_key,
            model=model,
            base_url=base_url,
        )

    health = provider_client.check_health()
    health["risk_signal_note"] = RISK_SIGNAL_NOTE
    return health


def inspect_calibration_tool(calibration_path: str) -> dict[str, Any]:
    """Return audit-friendly metadata about a calibration artifact."""
    calibration = _load_calibration(calibration_path)
    return {
        "status": "ok",
        "schema_version": calibration.schema_version,
        "created_at": calibration.created_at,
        "model": calibration.model,
        "provider": calibration.provider,
        "task_family": calibration.task_family,
        "calibration_mode": calibration.calibration_mode,
        "token_count": calibration.token_count,
        "sequence_count": calibration.sequence_count,
        "faithful_sequence_count": calibration.faithful_sequence_count,
        "entropy_mode": calibration.entropy_mode,
        "entropy_base": calibration.entropy_base,
        "top_logprobs": calibration.top_logprobs,
        "thresholds": calibration.thresholds,
        "dkw": calibration.dkw,
        "ecdf_size": len(calibration.ecdf_values),
        "known_limitations": calibration.known_limitations,
        "risk_signal_note": RISK_SIGNAL_NOTE,
    }


def create_mcp_server(default_calibration_path: Optional[str] = None) -> Any:
    """Create and register the FastMCP server.

    Importing MCP lazily keeps the base package usable without the optional MCP
    dependency installed.
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError(
            "MCP support requires the optional dependency: "
            'pip install "hallucination-sentinel[mcp] @ '
            'git+https://github.com/Heman10x-NGU/hallucination-sentinel.git"'
        ) from exc

    server = FastMCP("hallucination-sentinel")

    def _calibration(path: Optional[str]) -> str:
        resolved = path or default_calibration_path
        if not resolved:
            raise ValueError(
                "calibration_path is required. Pass it to the tool call or start "
                "sentinel-mcp with --calibration."
            )
        return resolved

    @server.tool()
    def score_entropy_sequence(
        entropies: list[float],
        calibration_path: Optional[str] = None,
        provider: str = "mcp",
        policy: str = "medium",
    ) -> dict[str, Any]:
        """Score pre-computed token entropies with CES."""
        return score_entropy_sequence_tool(
            entropies=entropies,
            calibration_path=_calibration(calibration_path),
            provider=provider,
            policy=policy,
        )

    @server.tool()
    def score_topk_logprobs(
        topk_logprobs: list[dict[str, float]],
        calibration_path: Optional[str] = None,
        top_k: Optional[int] = None,
        provider: str = "mcp",
        policy: str = "medium",
        add_residual: bool = True,
    ) -> dict[str, Any]:
        """Convert top-k logprobs to entropy and score them with CES."""
        return score_topk_logprobs_tool(
            topk_logprobs=topk_logprobs,
            calibration_path=_calibration(calibration_path),
            top_k=top_k,
            provider=provider,
            policy=policy,
            add_residual=add_residual,
        )

    @server.tool()
    def score_provider_output(
        prompt: str,
        output: str,
        provider: str,
        model: str,
        calibration_path: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        top_k: Optional[int] = None,
        policy: str = "medium",
    ) -> dict[str, Any]:
        """Score a prompt/output pair through a provider that returns top-k logprobs."""
        return score_provider_output_tool(
            prompt=prompt,
            output=output,
            provider=provider,
            model=model,
            calibration_path=_calibration(calibration_path),
            base_url=base_url,
            api_key=api_key,
            top_k=top_k,
            policy=policy,
        )

    @server.tool()
    def smoke_provider(
        provider: str,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        max_top_k: Optional[int] = None,
    ) -> dict[str, Any]:
        """Check whether a provider can return logprobs."""
        return smoke_provider_tool(
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
            max_top_k=max_top_k,
        )

    @server.tool()
    def inspect_calibration(calibration_path: Optional[str] = None) -> dict[str, Any]:
        """Inspect calibration metadata and limitations."""
        return inspect_calibration_tool(_calibration(calibration_path))

    return server


def main() -> None:
    """Run the MCP server over stdio."""
    parser = argparse.ArgumentParser(
        description="Hallucination Sentinel MCP server for CES risk scoring."
    )
    parser.add_argument(
        "--calibration",
        default=None,
        help="Default calibration artifact path for MCP tool calls.",
    )
    args = parser.parse_args()
    server = create_mcp_server(default_calibration_path=args.calibration)
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
