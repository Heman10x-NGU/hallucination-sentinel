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
            labels.append(bool(label))

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
# score
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# score-provider
# ---------------------------------------------------------------------------

@main.command("score-provider")
@click.option("--prompt", required=True, type=click.Path(exists=True),
              help="Path to prompt text file.")
@click.option("--output", "output_path", required=True, type=click.Path(exists=True),
              help="Path to output text file.")
@click.option("--provider", required=True, help="Provider preset name.")
@click.option("--model", required=True, help="Model name.")
@click.option("--calibration", required=True, type=click.Path(exists=True),
              help="Path to calibration artifact JSON.")
@click.option("--base-url", default=None, help="Override provider base URL.")
@click.option("--api-key", default=None, help="Override API key.")
def score_provider(
    prompt: str,
    output_path: str,
    provider: str,
    model: str,
    calibration: str,
    base_url: Optional[str],
    api_key: Optional[str],
):
    """Score output text using a provider's logprobs.

    Requires the provider to support echo-based text scoring.
    Only available after the provider passes a smoke test.
    """
    from .providers.openai_compatible import OpenAICompatibleProvider, PROVIDER_SPECS

    prompt_text = Path(prompt).read_text().strip()
    output_text = Path(output_path).read_text().strip()

    # Validate provider exists
    if provider not in PROVIDER_SPECS:
        raise click.ClickException(
            f"Unknown provider '{provider}'.  "
            f"Available: {', '.join(PROVIDER_SPECS)}"
        )

    spec = PROVIDER_SPECS[provider]
    if not spec.echo_supported:
        raise click.ClickException(
            f"Provider '{provider}' does not support echo-based text scoring.  "
            "Use a provider with echo_supported=True (e.g. together, vllm, fireworks)."
        )

    prov = OpenAICompatibleProvider.from_preset(
        provider,
        api_key=api_key,
        model=model,
        base_url=base_url,
    )

    # Score the output text
    completion = prov.score_text(output_text)

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
@click.option("--calibration", required=True, type=click.Path(exists=True),
              help="Path to calibration artifact JSON.")
@click.option("--output", "output_path", default="eval_report.json",
              type=click.Path(), help="Output path for eval report JSON.")
def eval_cmd(
    input_path: str,
    calibration: str,
    output_path: str,
):
    """Evaluate CES against baseline metrics.

    Input JSONL lines must have:
      - entropy: list of per-token entropy values
      - label: boolean (true = faithful, false = hallucinated)
      - ln_entropy (optional): length-normalized entropy baseline
      - perplexity (optional): perplexity baseline
      - length (optional): generation length baseline

    Computes AUROC, AUPRC, confusion matrix at thresholds for CES
    and all provided baselines.
    """
    try:
        from sklearn.metrics import (
            roc_auc_score,
            average_precision_score,
            confusion_matrix,
        )
    except ImportError:
        raise click.ClickException(
            "scikit-learn is required for eval.  "
            "Install with: pip install hallucination-sentinel[eval]"
        )

    records = _read_jsonl(input_path)
    artifact = load_calibration(calibration)

    # Compute CES for each record
    ces_scores: list[float] = []
    labels: list[bool] = []
    ln_entropies: list[Optional[float]] = []
    perplexities: list[Optional[float]] = []
    lengths: list[Optional[int]] = []

    for idx, rec in enumerate(records):
        entropy_vals = rec.get("entropy")
        if entropy_vals is None:
            raise click.ClickException(
                f"Record {idx + 1} missing 'entropy' field."
            )
        entropy_arr = np.asarray(entropy_vals, dtype=np.float64)
        if entropy_arr.size == 0:
            raise click.ClickException(
                f"Record {idx + 1} has empty 'entropy' array."
            )

        ces_result = compute_ces(entropy_arr, artifact)
        ces_scores.append(ces_result.ces_score)

        label = rec.get("label")
        if label is None:
            raise click.ClickException(
                f"Record {idx + 1} missing 'label' field."
            )
        labels.append(bool(label))

        ln_entropies.append(rec.get("ln_entropy"))
        perplexities.append(rec.get("perplexity"))
        lengths.append(rec.get("length"))

    y_true = np.array(labels, dtype=int)
    y_ces = np.array(ces_scores)

    # Compute metrics for CES
    report: dict = {"n_samples": len(records), "methods": {}}

    def _compute_method_metrics(name: str, scores: np.ndarray) -> dict:
        """Compute AUROC, AUPRC, confusion matrix for a method."""
        try:
            auroc = float(roc_auc_score(y_true, scores))
        except ValueError:
            auroc = None
        try:
            auprc = float(average_precision_score(y_true, scores))
        except ValueError:
            auprc = None

        # Confusion matrix at default threshold (median of scores)
        threshold = float(np.median(scores))
        y_pred = (scores >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        return {
            "auroc": round(auroc, 6) if auroc is not None else None,
            "auprc": round(auprc, 6) if auprc is not None else None,
            "threshold": round(threshold, 6),
            "confusion_matrix": {
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
            },
        }

    # CES metrics
    report["methods"]["ces"] = _compute_method_metrics("ces", y_ces)

    # LN-Entropy baseline
    ln_vals = [v for v in ln_entropies if v is not None]
    if len(ln_vals) == len(records):
        ln_arr = np.array(ln_vals)
        report["methods"]["ln_entropy"] = _compute_method_metrics("ln_entropy", ln_arr)

    # Perplexity baseline
    perp_vals = [v for v in perplexities if v is not None]
    if len(perp_vals) == len(records):
        perp_arr = np.array(perp_vals)
        report["methods"]["perplexity"] = _compute_method_metrics("perplexity", perp_arr)

    # Length baseline
    len_vals = [v for v in lengths if v is not None]
    if len(len_vals) == len(records):
        len_arr = np.array(len_vals, dtype=float)
        report["methods"]["generation_length"] = _compute_method_metrics(
            "generation_length", len_arr
        )

    _write_json(output_path, report)

    _json_output({
        "status": "ok",
        "output": str(output_path),
        "n_samples": report["n_samples"],
        "methods_evaluated": list(report["methods"].keys()),
    })


# ---------------------------------------------------------------------------
# smoke-provider
# ---------------------------------------------------------------------------

@main.command("smoke-provider")
@click.option("--provider", required=True, help="Provider preset name.")
@click.option("--base-url", default=None, help="Override provider base URL.")
@click.option("--model", default=None, help="Override model name.")
def smoke_provider(provider: str, base_url: Optional[str], model: Optional[str]):
    """Test a provider's logprob support.

    Makes a minimal API call to verify the provider can return logprobs.
    Reports capabilities: selected-token logprobs, top-k, echo support.
    """
    from .providers.openai_compatible import OpenAICompatibleProvider, PROVIDER_SPECS

    if provider not in PROVIDER_SPECS:
        raise click.ClickException(
            f"Unknown provider '{provider}'.  "
            f"Available: {', '.join(PROVIDER_SPECS)}"
        )

    spec = PROVIDER_SPECS[provider]

    try:
        prov = OpenAICompatibleProvider.from_preset(
            provider,
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

    health = prov.check_health()

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
