"""
CLI entry point for Hallucination Sentinel.

Commands:
    sentinel calibrate --input calibration.jsonl --output calibration.json
    sentinel inspect-calibration --calibration calibration.json
    sentinel score --entropy-json entropy_sequence.json --calibration calibration.json
    sentinel score-provider --prompt prompt.txt --output output.txt --provider ... --model ... --calibration ...
    sentinel eval --input eval.jsonl --calibration calibration.json --output eval_report.json
    sentinel smoke-provider --provider openai-compatible --base-url URL --model MODEL
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Optional

import click
import numpy as np

from .calibration import (
    CalibrationArtifact,
    build_calibration,
    load_calibration,
    save_calibration,
)
from .ces import compute_ces
from .schemas import parse_label
from .entropy import (
    entropy_from_logprobs,
    entropy_from_probs,
    length_normalized_entropy,
    perplexity_from_logprobs,
)
from .thresholds import assign_thresholds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file and return a list of dicts."""
    path = Path(path)
    if not path.exists():
        raise click.ClickException(f"File not found: {path}")
    records = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise click.ClickException(f"Invalid JSON on line {lineno} of {path}: {e}")
    if not records:
        raise click.ClickException(f"No records found in {path}")
    return records


def _write_json(path: str | Path, data: dict) -> None:
    """Write a dict as pretty-printed JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def _json_output(data: dict) -> None:
    """Print JSON to stdout."""
    click.echo(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(package_name="hallucination-sentinel")
def main():
    """Hallucination Sentinel - Single-pass uncertainty firewall for LLM outputs."""
    pass


# ---------------------------------------------------------------------------
# calibrate
# ---------------------------------------------------------------------------

@main.command()
@click.option("--input", "input_path", required=True, type=click.Path(exists=True),
              help="Path to calibration JSONL file.")
@click.option("--output", "output_path", default="calibration.json",
              type=click.Path(), help="Output path for calibration artifact JSON.")
@click.option("--mode", default="unsupervised",
              type=click.Choice(["unsupervised", "supervised"]),
              help="Calibration mode.")
@click.option("--model", default="", help="Model name for metadata.")
@click.option("--provider", default="", help="Provider name for metadata.")
@click.option("--task-family", default="", help="Task family for metadata.")
def calibrate(
    input_path: str,
    output_path: str,
    mode: str,
    model: str,
    provider: str,
    task_family: str,
):
    """Build a calibration artifact from a JSONL dataset.

    Each JSONL line must have an 'entropy' field containing a list of
    per-token entropy values.  For supervised mode, a 'label' field
    (boolean) is used to select faithful sequences.
    """
    records = _read_jsonl(input_path)

    entropy_sequences: list[np.ndarray] = []
    labels: list[bool] = []

    for idx, rec in enumerate(records):
        entropy_vals = rec.get("entropy")
        if entropy_vals is None:
            raise click.ClickException(
                f"Record {idx + 1} missing 'entropy' field.  "
                "Each JSONL line must have an 'entropy' list of per-token entropy values."
            )
        arr = np.asarray(entropy_vals, dtype=np.float64)
        if arr.size == 0:
            raise click.ClickException(
                f"Record {idx + 1} has empty 'entropy' array."
            )
        entropy_sequences.append(arr)
        if mode == "supervised":
            label = rec.get("label", True)
            labels.append(label)

    artifact = build_calibration(
        entropy_sequences,
        labels=labels if mode == "supervised" else None,
        mode=mode,
        model=model,
        provider=provider,
        task_family=task_family,
    )

    # Assign default quantile thresholds
    assign_thresholds(artifact)

    save_calibration(artifact, output_path)

    _json_output({
        "status": "ok",
        "output": str(output_path),
        "token_count": artifact.token_count,
        "sequence_count": artifact.sequence_count,
        "calibration_mode": artifact.calibration_mode,
    })


# ---------------------------------------------------------------------------
# inspect-calibration
# ---------------------------------------------------------------------------

@main.command("inspect-calibration")
@click.option("--calibration", required=True, type=click.Path(exists=True),
              help="Path to calibration artifact JSON.")
def inspect_calibration(calibration: str):
    """Print calibration metadata in readable format."""
    artifact = load_calibration(calibration)

    output = {
        "schema_version": artifact.schema_version,
        "created_at": artifact.created_at,
        "model": artifact.model,
        "provider": artifact.provider,
        "task_family": artifact.task_family,
        "calibration_mode": artifact.calibration_mode,
        "token_count": artifact.token_count,
        "sequence_count": artifact.sequence_count,
        "faithful_sequence_count": artifact.faithful_sequence_count,
        "entropy_mode": artifact.entropy_mode,
        "entropy_base": artifact.entropy_base,
        "top_logprobs": artifact.top_logprobs,
        "decoding": artifact.decoding,
        "length_summary": artifact.length_summary,
        "dkw": artifact.dkw,
        "thresholds": artifact.thresholds,
        "ecdf_size": len(artifact.ecdf_values),
        "known_limitations": artifact.known_limitations,
    }
    _json_output(output)


# ---------------------------------------------------------------------------
# score / check (shared implementation)
# ---------------------------------------------------------------------------

def _score_entropy_file(entropy_json: str, calibration: str) -> None:
    """Compute CES score from an entropy sequence file.

    Shared implementation for both ``score`` and ``check`` commands.
    """
    data = json.loads(Path(entropy_json).read_text())
    entropy_vals = data.get("entropy")
    if entropy_vals is None:
        raise click.ClickException(
            "Entropy JSON must contain an 'entropy' field with a list of per-token entropy values."
        )
    entropy_sequence = np.asarray(entropy_vals, dtype=np.float64)
    if entropy_sequence.size == 0:
        raise click.ClickException("Entropy sequence is empty.")

    artifact = load_calibration(calibration)
    result = compute_ces(entropy_sequence, artifact)

    _json_output({
        "ces_score": round(result.ces_score, 6),
        "risk_level": result.risk_level,
        "cdf_mean": round(result.cdf_mean, 6),
        "cdf_max": round(result.cdf_max, 6),
        "mean_entropy": round(result.mean_entropy, 6),
        "max_entropy": round(result.max_entropy, 6),
        "token_count": result.token_count,
        "warnings": result.warnings,
    })


@main.command()
@click.option("--entropy-json", required=True, type=click.Path(exists=True),
              help="Path to entropy sequence JSON file.")
@click.option("--calibration", required=True, type=click.Path(exists=True),
              help="Path to calibration artifact JSON.")
def score(entropy_json: str, calibration: str):
    """Compute CES score from an entropy sequence file.

    The entropy JSON must contain an 'entropy' field with a list of
    per-token entropy values.
    """
    _score_entropy_file(entropy_json, calibration)


@main.command()
@click.option("--entropy-json", required=True, type=click.Path(exists=True),
              help="Path to entropy sequence JSON file.")
@click.option("--calibration", required=True, type=click.Path(exists=True),
              help="Path to calibration artifact JSON.")
def check(entropy_json: str, calibration: str):
    """Alias for ``score`` -- compute CES score from an entropy sequence file.

    The entropy JSON must contain an 'entropy' field with a list of
    per-token entropy values.
    """
    _score_entropy_file(entropy_json, calibration)


# ---------------------------------------------------------------------------
# score-provider
# ---------------------------------------------------------------------------

@main.command("score-provider")
@click.option("--prompt", default=None, type=click.Path(exists=True),
              help="Path to prompt text file.")
@click.option("--output", "output_path", default=None, type=click.Path(exists=True),
              help="Path to output text file.")
@click.option("--provider", required=True, help="Provider preset name.")
@click.option("--model", required=True, help="Model name.")
@click.option("--calibration", required=True, type=click.Path(exists=True),
              help="Path to calibration artifact JSON.")
@click.option("--mode", "scoring_mode", default="score-output",
              type=click.Choice(["generated", "score-output", "score-text"]),
              help="Scoring mode: generated (generate+score), "
                   "score-output (score output given prompt), "
                   "score-text (score arbitrary text).")
@click.option("--base-url", default=None, help="Override provider base URL.")
@click.option("--api-key", default=None, help="Override API key.")
def score_provider(
    prompt: Optional[str],
    output_path: Optional[str],
    provider: str,
    model: str,
    calibration: str,
    scoring_mode: str,
    base_url: Optional[str],
    api_key: Optional[str],
):
    """Score LLM outputs using a provider's logprobs.

    Modes:
        generated     -- Generate text from --prompt, then score the generation.
                         Requires --prompt.  --output is ignored.
        score-output  -- Score an existing output conditioned on its prompt.
                         Requires both --prompt and --output.
                         Provider must support echo/teacher-forcing.
        score-text    -- Score arbitrary text in isolation (no prompt context).
                         Requires --output.
    """
    from .providers.openai_compatible import OpenAICompatibleProvider, PROVIDER_SPECS

    # Validate provider exists
    if provider not in PROVIDER_SPECS:
        raise click.ClickException(
            f"Unknown provider '{provider}'.  "
            f"Available: {', '.join(PROVIDER_SPECS)}"
        )

    # Validate mode-specific required options
    if scoring_mode in ("generated", "score-output") and not prompt:
        raise click.ClickException(
            f"--mode {scoring_mode} requires --prompt."
        )
    if scoring_mode in ("score-output", "score-text") and not output_path:
        raise click.ClickException(
            f"--mode {scoring_mode} requires --output."
        )

    # For score-output mode, verify provider supports echo before reading files
    if scoring_mode == "score-output":
        spec = PROVIDER_SPECS[provider]
        if not spec.echo_supported:
            raise click.ClickException(
                f"Provider '{provider}' does not support echo-based output scoring.  "
                "Use a provider with echo_supported=True (e.g. together, vllm, fireworks)."
            )

    # Read input files
    prompt_text = Path(prompt).read_text().strip() if prompt else ""
    output_text = Path(output_path).read_text().strip() if output_path else ""

    prov = OpenAICompatibleProvider.from_preset(
        provider,
        api_key=api_key,
        model=model,
        base_url=base_url,
    )

    # Dispatch to the correct provider method based on mode
    if scoring_mode == "generated":
        completion = prov.generate_with_logprobs(prompt_text)
    elif scoring_mode == "score-output":
        completion = prov.score_output(prompt_text, output_text)
    elif scoring_mode == "score-text":
        completion = prov.score_text(output_text)
    else:
        raise click.ClickException(f"Unknown scoring mode: {scoring_mode}")

    # Compute entropy from top-k logprobs
    from .entropy import entropy_from_topk_logprobs

    topk = completion.topk_logprobs
    if not topk:
        raise click.ClickException("Provider returned no top-k logprobs.")

    entropy_result = entropy_from_topk_logprobs(topk, top_k=completion.top_k)

    # Compute CES
    artifact = load_calibration(calibration)
    ces_result = compute_ces(entropy_result.entropies, artifact)

    _json_output({
        "scoring_mode": scoring_mode,
        "ces_score": round(ces_result.ces_score, 6),
        "risk_level": ces_result.risk_level,
        "cdf_mean": round(ces_result.cdf_mean, 6),
        "cdf_max": round(ces_result.cdf_max, 6),
        "mean_entropy": round(ces_result.mean_entropy, 6),
        "max_entropy": round(ces_result.max_entropy, 6),
        "token_count": ces_result.token_count,
        "warnings": ces_result.warnings,
        "provider": provider,
        "model": model,
    })


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------

@main.command()
@click.option("--input", "input_path", required=True, type=click.Path(exists=True),
              help="Path to eval JSONL file.")
@click.option("--calibration", "calibration_path", default=None,
              type=click.Path(exists=True),
              help="Path to calibration artifact JSON. Required for raw entropy rows.")
@click.option("--output", "output_path", default="eval_report.json",
              type=click.Path(), help="Output path for eval report.")
@click.option("--markdown", "markdown_output", is_flag=True, default=False,
              help="Output report as Markdown instead of JSON.")
@click.option("--threshold", "thresholds", multiple=True, type=float,
              help="Decision thresholds for confusion matrices (repeatable). "
                   "Defaults to [0.5, 0.7, 0.8, 0.9].")
@click.option("--bootstrap/--no-bootstrap", default=True,
              help="Compute bootstrap confidence intervals (default: on).")
@click.option("--n-bootstrap", default=1000, type=int,
              help="Number of bootstrap resamples (default: 1000).")
def eval_cmd(
    input_path: str,
    calibration_path: Optional[str],
    output_path: str,
    markdown_output: bool,
    thresholds: tuple[float, ...],
    bootstrap: bool,
    n_bootstrap: int,
):
    """Evaluate CES against baseline metrics on labeled data.

    Input JSONL lines can be in two formats:

    \b
    Pre-scored (no calibration needed):
      {"ces_score": 0.8, "label": 1, "token_count": 10,
       "token_entropies": [1.2, 0.8, ...]}

    \b
    Raw entropy (requires --calibration):
      {"entropy": [1.2, 0.8, 0.5, ...], "label": 1}

    The report includes AUROC/AUPRC with bootstrap confidence intervals,
    confusion matrices at configured thresholds, baseline comparisons,
    per-length-bucket calibration coverage, and lag-1 autocorrelation
    diagnostics.
    """
    from .eval import (
        build_eval_report,
        normalize_to_eval_record,
        report_to_json,
        report_to_markdown,
        save_report_json,
        save_report_markdown,
    )

    records_raw = _read_jsonl(input_path)

    # Load calibration only if needed
    calibration = None
    if calibration_path:
        calibration = load_calibration(calibration_path)

    # Detect input format: if any row has 'entropy' but no 'ces_score',
    # calibration is required.
    has_raw = any("entropy" in r and "ces_score" not in r for r in records_raw)
    if has_raw and calibration is None:
        raise click.ClickException(
            "Input contains raw entropy rows but --calibration was not provided.  "
            "Provide --calibration to compute CES scores from entropy sequences."
        )

    # Normalize all rows to EvalRecord
    eval_records = []
    for idx, rec in enumerate(records_raw):
        try:
            er = normalize_to_eval_record(rec, lineno=idx + 1, calibration=calibration)
        except ValueError as exc:
            raise click.ClickException(str(exc))
        eval_records.append(er)

    if not eval_records:
        raise click.ClickException("No evaluation records loaded.")

    # Build report via eval.py
    threshold_list = list(thresholds) if thresholds else None
    report = build_eval_report(
        eval_records,
        thresholds=threshold_list,
        bootstrap=bootstrap,
        n_bootstrap=n_bootstrap,
    )

    # Serialize and save
    ext = Path(output_path).suffix.lower()
    if markdown_output or ext in (".md", ".markdown"):
        save_report_markdown(report, output_path)
        _json_output({
            "status": "ok",
            "format": "markdown",
            "output": str(output_path),
            "n_samples": report.n_samples,
        })
    else:
        save_report_json(report, output_path)
        _json_output({
            "status": "ok",
            "format": "json",
            "output": str(output_path),
            "n_samples": report.n_samples,
        })


# ---------------------------------------------------------------------------
# smoke-provider
# ---------------------------------------------------------------------------

@main.command("smoke-provider")
@click.option("--provider", required=True, help="Provider preset name (or 'custom').")
@click.option("--base-url", default=None, help="Override provider base URL.")
@click.option("--model", default=None, help="Override model name.")
@click.option("--api-key", default=None, help="Override API key.")
@click.option("--max-top-k", default=None, type=int, help="Override max top-k (custom provider only).")
def smoke_provider(
    provider: str,
    base_url: Optional[str],
    model: Optional[str],
    api_key: Optional[str],
    max_top_k: Optional[int],
):
    """Test a provider's logprob support.

    Makes a minimal API call to verify the provider can return logprobs.
    Reports capabilities: selected-token logprobs, top-k, echo support.

    Use --provider custom with --base-url, --model, --api-key, --max-top-k
    for arbitrary OpenAI-compatible endpoints.
    """
    from .providers.openai_compatible import OpenAICompatibleProvider, PROVIDER_SPECS

    if provider == "custom":
        if not base_url or not model:
            raise click.ClickException(
                "--provider custom requires --base-url and --model."
            )
        prov = OpenAICompatibleProvider.custom(
            base_url=base_url,
            model=model,
            api_key=api_key or "",
            max_top_k=max_top_k or 20,
        )
        spec = prov.spec
    else:
        if provider not in PROVIDER_SPECS:
            raise click.ClickException(
                f"Unknown provider '{provider}'.  "
                f"Available: {', '.join(PROVIDER_SPECS)}, custom"
            )
        spec = PROVIDER_SPECS[provider]
        try:
            prov = OpenAICompatibleProvider.from_preset(
                provider,
                api_key=api_key,
                model=model,
                base_url=base_url,
            )
        except Exception as e:
            _json_output({
                "provider": provider,
                "status": "FAIL",
                "error": str(e),
            })
            return

    try:
        health = prov.check_health()
    except ValueError as e:
        _json_output({
            "provider": spec.name,
            "preset": provider,
            "base_url": base_url or spec.base_url,
            "model": model or spec.model,
            "status": "FAIL",
            "error": str(e),
        })
        return

    result = {
        "provider": spec.name,
        "preset": provider,
        "base_url": base_url or spec.base_url,
        "model": model or spec.model,
        "status": "PASS" if health.get("healthy") else "FAIL",
        "capabilities": {
            "echo_supported": spec.echo_supported,
            "max_top_k": spec.max_top_k,
            "selected_token_logprobs": health.get("healthy", False),
            "top_k_logprobs": health.get("top_k_available", False),
            "arbitrary_text_scoring": spec.echo_supported,
        },
    }

    if not health.get("healthy"):
        result["error"] = health.get("error", "Unknown error")

    _json_output(result)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
