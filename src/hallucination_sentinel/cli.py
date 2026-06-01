"""
CLI entry point for Hallucination Sentinel.

Usage:
    sentinel check --provider together --prompt "What is the capital of France?"
    sentinel calibrate --data calibration_data.json --method isotonic
    sentinel version
"""

import sys
from typing import Optional

try:
    import click
except ImportError:
    click = None  # type: ignore[assignment]

try:
    from rich.console import Console
except ImportError:
    Console = None  # type: ignore[assignment]


def _get_console():
    """Get a Rich console, or fall back to plain print."""
    if Console is not None:
        return Console()
    return None


@click.group() if click else lambda f: f
@click.version_option(package_name="hallucination-sentinel")
def main():
    """Hallucination Sentinel - Single-pass uncertainty firewall for LLM outputs."""
    pass


@main.command() if click else lambda f: f
@click.option("--provider", default="together", help="LLM provider name")
@click.option("--prompt", required=True, help="Prompt to check")
@click.option("--model", default=None, help="Model override")
@click.option("--top-k", default=None, type=int, help="Top-k logprobs to request")
@click.option("--calibration", default=None, help="Path to calibration artifact JSON")
@click.option("--thresholds", default=None, help="Path to thresholds JSON")
def check(
    provider: str,
    prompt: str,
    model: Optional[str],
    top_k: Optional[int],
    calibration: Optional[str],
    thresholds: Optional[str],
):
    """Check a prompt/response for hallucinations."""
    console = _get_console()
    if console:
        console.print(f"[bold]Checking with provider:[/bold] {provider}")
        console.print(f"[bold]Prompt:[/bold] {prompt}")
    else:
        print(f"Checking with provider: {provider}")
        print(f"Prompt: {prompt}")

    # TODO: Implement actual check pipeline
    # 1. Create provider from preset
    # 2. Generate with logprobs
    # 3. Compute entropy
    # 4. Run CES
    # 5. Calibrate
    # 6. Assign thresholds
    # 7. Display result
    print("Not yet implemented. This is a CLI stub.")
    sys.exit(1)


@main.command() if click else lambda f: f
@click.option("--data", required=True, help="Path to calibration dataset JSON")
@click.option("--method", default="isotonic", type=click.Choice(["isotonic", "platt", "sigmoid"]))
@click.option("--output", default="calibration.json", help="Output path for calibration artifact")
def calibrate(data: str, method: str, output: str):
    """Fit a calibration model from labeled data."""
    console = _get_console()
    if console:
        console.print(f"[bold]Calibrating with method:[/bold] {method}")
        console.print(f"[bold]Data:[/bold] {data}")
    else:
        print(f"Calibrating with method: {method}")
        print(f"Data: {data}")

    # TODO: Implement calibration fitting
    print("Not yet implemented. This is a CLI stub.")
    sys.exit(1)


if __name__ == "__main__":
    main()
