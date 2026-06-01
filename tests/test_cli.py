"""Tests for the CLI module using Click's CliRunner.

Covers all 6 commands:
    - calibrate
    - inspect-calibration
    - score
    - score-provider
    - eval
    - smoke-provider
"""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from click.testing import CliRunner

from hallucination_sentinel.cli import main
from hallucination_sentinel.calibration import build_calibration, save_calibration


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def runner():
    return CliRunner()


def _make_calibration_file(tmp_path: Path) -> Path:
    """Build and save a calibration artifact for use in tests."""
    rng = np.random.RandomState(42)
    seqs = [rng.uniform(0.5, 3.0, size=50) for _ in range(20)]
    artifact = build_calibration(seqs, mode="unsupervised", model="test/model")
    from hallucination_sentinel.thresholds import assign_thresholds
    assign_thresholds(artifact)
    path = tmp_path / "calibration.json"
    save_calibration(artifact, path)
    return path


def _make_calibration_jsonl(tmp_path: Path, n: int = 10, supervised: bool = False) -> Path:
    """Create a calibration JSONL file."""
    rng = np.random.RandomState(42)
    path = tmp_path / "calibration.jsonl"
    lines = []
    for i in range(n):
        entropy = rng.uniform(0.5, 3.0, size=30).tolist()
        rec = {"entropy": entropy}
        if supervised:
            rec["label"] = i % 2 == 0  # alternating true/false
        lines.append(json.dumps(rec))
    path.write_text("\n".join(lines) + "\n")
    return path


def _make_entropy_json(tmp_path: Path, n: int = 30) -> Path:
    """Create an entropy sequence JSON file."""
    rng = np.random.RandomState(99)
    path = tmp_path / "entropy.json"
    data = {"entropy": rng.uniform(0.5, 3.0, size=n).tolist()}
    path.write_text(json.dumps(data))
    return path


def _make_eval_jsonl(
    tmp_path: Path,
    n: int = 20,
    with_baselines: bool = False,
) -> Path:
    """Create an eval JSONL file."""
    rng = np.random.RandomState(42)
    path = tmp_path / "eval.jsonl"
    lines = []
    for i in range(n):
        entropy = rng.uniform(0.5, 3.0, size=30).tolist()
        rec = {
            "entropy": entropy,
            "label": i < n // 2,  # first half faithful, second half hallucinated
        }
        if with_baselines:
            rec["ln_entropy"] = float(rng.uniform(0.5, 3.0))
            rec["perplexity"] = float(rng.uniform(1.0, 10.0))
            rec["length"] = int(rng.randint(10, 100))
        lines.append(json.dumps(rec))
    path.write_text("\n".join(lines) + "\n")
    return path


def _make_text_file(tmp_path: Path, name: str, content: str) -> Path:
    """Create a text file."""
    path = tmp_path / name
    path.write_text(content)
    return path


def _make_topk_response() -> dict:
    """Build a mock OpenAI-style response with top-k logprobs."""
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": "Hello"},
                "logprobs": {
                    "content": [
                        {
                            "token": "Hello",
                            "logprob": -0.1,
                            "top_logprobs": [
                                {"token": "Hello", "logprob": -0.1},
                                {"token": "Hi", "logprob": -1.5},
                                {"token": "Hey", "logprob": -2.0},
                            ],
                        },
                        {
                            "token": " world",
                            "logprob": -0.5,
                            "top_logprobs": [
                                {"token": " world", "logprob": -0.5},
                                {"token": " there", "logprob": -1.0},
                            ],
                        },
                    ]
                },
            }
        ]
    }


# ---------------------------------------------------------------------------
# calibrate
# ---------------------------------------------------------------------------

class TestCalibrateCommand:
    """Tests for `sentinel calibrate`."""

    def test_basic_unsupervised(self, runner, tmp_path):
        input_path = _make_calibration_jsonl(tmp_path)
        output_path = tmp_path / "out.json"

        result = runner.invoke(main, [
            "calibrate",
            "--input", str(input_path),
            "--output", str(output_path),
        ])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["status"] == "ok"
        assert data["calibration_mode"] == "unsupervised"
        assert data["token_count"] == 10 * 30
        assert data["sequence_count"] == 10
        assert output_path.exists()

    def test_supervised_mode(self, runner, tmp_path):
        input_path = _make_calibration_jsonl(tmp_path, supervised=True)
        output_path = tmp_path / "out.json"

        result = runner.invoke(main, [
            "calibrate",
            "--input", str(input_path),
            "--output", str(output_path),
            "--mode", "supervised",
        ])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["calibration_mode"] == "supervised"
        assert data["sequence_count"] == 5  # only truthy-labeled sequences in reference pool

    def test_with_metadata(self, runner, tmp_path):
        input_path = _make_calibration_jsonl(tmp_path)
        output_path = tmp_path / "out.json"

        result = runner.invoke(main, [
            "calibrate",
            "--input", str(input_path),
            "--output", str(output_path),
            "--model", "gpt-4o",
            "--provider", "openai",
            "--task-family", "short_qa",
        ])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["status"] == "ok"

        # Verify metadata was stored
        cal = json.loads(output_path.read_text())
        assert cal["model"] == "gpt-4o"
        assert cal["provider"] == "openai"
        assert cal["task_family"] == "short_qa"

    def test_default_output_path(self, runner, tmp_path):
        """Default output is 'calibration.json' in cwd."""
        input_path = _make_calibration_jsonl(tmp_path)

        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(main, [
                "calibrate",
                "--input", str(input_path),
            ])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["output"] == "calibration.json"

    def test_missing_input_file(self, runner):
        result = runner.invoke(main, [
            "calibrate",
            "--input", "/nonexistent/file.jsonl",
        ])
        assert result.exit_code != 0

    def test_missing_entropy_field(self, runner, tmp_path):
        """Records without 'entropy' field should fail."""
        path = tmp_path / "bad.jsonl"
        path.write_text('{"prompt": "test"}\n')

        result = runner.invoke(main, [
            "calibrate",
            "--input", str(path),
            "--output", str(tmp_path / "out.json"),
        ])
        assert result.exit_code != 0
        assert "entropy" in result.output.lower()

    def test_empty_entropy_array(self, runner, tmp_path):
        """Records with empty entropy should fail."""
        path = tmp_path / "bad.jsonl"
        path.write_text('{"entropy": []}\n')

        result = runner.invoke(main, [
            "calibrate",
            "--input", str(path),
            "--output", str(tmp_path / "out.json"),
        ])
        assert result.exit_code != 0
        assert "empty" in result.output.lower()

    def test_empty_jsonl(self, runner, tmp_path):
        """Empty JSONL file should fail."""
        path = tmp_path / "empty.jsonl"
        path.write_text("")

        result = runner.invoke(main, [
            "calibrate",
            "--input", str(path),
            "--output", str(tmp_path / "out.json"),
        ])
        assert result.exit_code != 0

    def test_invalid_json_line(self, runner, tmp_path):
        """Invalid JSON lines should fail with line number."""
        path = tmp_path / "bad.jsonl"
        path.write_text('{"entropy": [1.0]}\nnot json\n')

        result = runner.invoke(main, [
            "calibrate",
            "--input", str(path),
            "--output", str(tmp_path / "out.json"),
        ])
        assert result.exit_code != 0
        assert "line 2" in result.output.lower() or "invalid json" in result.output.lower()


# ---------------------------------------------------------------------------
# inspect-calibration
# ---------------------------------------------------------------------------

class TestInspectCalibrationCommand:
    """Tests for `sentinel inspect-calibration`."""

    def test_basic_inspect(self, runner, tmp_path):
        cal_path = _make_calibration_file(tmp_path)

        result = runner.invoke(main, [
            "inspect-calibration",
            "--calibration", str(cal_path),
        ])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["schema_version"] == "0.1"
        assert data["model"] == "test/model"
        assert data["token_count"] > 0
        assert data["sequence_count"] > 0
        assert "length_summary" in data
        assert "dkw" in data
        assert "thresholds" in data
        assert "ecdf_size" in data
        assert "known_limitations" in data

    def test_length_summary_present(self, runner, tmp_path):
        cal_path = _make_calibration_file(tmp_path)

        result = runner.invoke(main, [
            "inspect-calibration",
            "--calibration", str(cal_path),
        ])

        data = json.loads(result.output)
        ls = data["length_summary"]
        assert "min" in ls
        assert "p50" in ls
        assert "p90" in ls
        assert "max" in ls

    def test_thresholds_present(self, runner, tmp_path):
        cal_path = _make_calibration_file(tmp_path)

        result = runner.invoke(main, [
            "inspect-calibration",
            "--calibration", str(cal_path),
        ])

        data = json.loads(result.output)
        thresholds = data["thresholds"]
        assert "low" in thresholds
        assert "medium" in thresholds
        assert "high" in thresholds

    def test_missing_calibration_file(self, runner):
        result = runner.invoke(main, [
            "inspect-calibration",
            "--calibration", "/nonexistent/cal.json",
        ])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------

class TestScoreCommand:
    """Tests for `sentinel score`."""

    def test_basic_score(self, runner, tmp_path):
        cal_path = _make_calibration_file(tmp_path)
        entropy_path = _make_entropy_json(tmp_path)

        result = runner.invoke(main, [
            "score",
            "--entropy-json", str(entropy_path),
            "--calibration", str(cal_path),
        ])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "ces_score" in data
        assert "risk_level" in data
        assert "cdf_mean" in data
        assert "cdf_max" in data
        assert "mean_entropy" in data
        assert "max_entropy" in data
        assert "token_count" in data
        assert "warnings" in data
        assert isinstance(data["warnings"], list)

    def test_score_values_in_range(self, runner, tmp_path):
        cal_path = _make_calibration_file(tmp_path)
        entropy_path = _make_entropy_json(tmp_path)

        result = runner.invoke(main, [
            "score",
            "--entropy-json", str(entropy_path),
            "--calibration", str(cal_path),
        ])

        data = json.loads(result.output)
        assert 0.0 <= data["ces_score"] <= 1.0
        assert 0.0 <= data["cdf_mean"] <= 1.0
        assert 0.0 <= data["cdf_max"] <= 1.0
        assert data["mean_entropy"] > 0
        assert data["max_entropy"] > 0
        assert data["token_count"] > 0

    def test_missing_entropy_field(self, runner, tmp_path):
        cal_path = _make_calibration_file(tmp_path)
        bad_path = tmp_path / "bad.json"
        bad_path.write_text('{"not_entropy": [1.0, 2.0]}')

        result = runner.invoke(main, [
            "score",
            "--entropy-json", str(bad_path),
            "--calibration", str(cal_path),
        ])
        assert result.exit_code != 0
        assert "entropy" in result.output.lower()

    def test_empty_entropy_sequence(self, runner, tmp_path):
        cal_path = _make_calibration_file(tmp_path)
        empty_path = tmp_path / "empty.json"
        empty_path.write_text('{"entropy": []}')

        result = runner.invoke(main, [
            "score",
            "--entropy-json", str(empty_path),
            "--calibration", str(cal_path),
        ])
        assert result.exit_code != 0

    def test_missing_calibration(self, runner, tmp_path):
        entropy_path = _make_entropy_json(tmp_path)

        result = runner.invoke(main, [
            "score",
            "--entropy-json", str(entropy_path),
            "--calibration", "/nonexistent/cal.json",
        ])
        assert result.exit_code != 0


class TestCheckCommand:
    """Tests for `sentinel check` (alias for `score`)."""

    def test_basic_check(self, runner, tmp_path):
        cal_path = _make_calibration_file(tmp_path)
        entropy_path = _make_entropy_json(tmp_path)

        result = runner.invoke(main, [
            "check",
            "--entropy-json", str(entropy_path),
            "--calibration", str(cal_path),
        ])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "ces_score" in data
        assert "risk_level" in data
        assert "cdf_mean" in data
        assert "cdf_max" in data
        assert "mean_entropy" in data
        assert "max_entropy" in data
        assert "token_count" in data
        assert "warnings" in data
        assert isinstance(data["warnings"], list)

    def test_check_values_in_range(self, runner, tmp_path):
        cal_path = _make_calibration_file(tmp_path)
        entropy_path = _make_entropy_json(tmp_path)

        result = runner.invoke(main, [
            "check",
            "--entropy-json", str(entropy_path),
            "--calibration", str(cal_path),
        ])

        data = json.loads(result.output)
        assert 0.0 <= data["ces_score"] <= 1.0
        assert 0.0 <= data["cdf_mean"] <= 1.0
        assert 0.0 <= data["cdf_max"] <= 1.0
        assert data["mean_entropy"] > 0
        assert data["max_entropy"] > 0
        assert data["token_count"] > 0

    def test_check_missing_entropy_field(self, runner, tmp_path):
        cal_path = _make_calibration_file(tmp_path)
        bad_path = tmp_path / "bad.json"
        bad_path.write_text('{"not_entropy": [1.0, 2.0]}')

        result = runner.invoke(main, [
            "check",
            "--entropy-json", str(bad_path),
            "--calibration", str(cal_path),
        ])
        assert result.exit_code != 0
        assert "entropy" in result.output.lower()

    def test_check_empty_entropy_sequence(self, runner, tmp_path):
        cal_path = _make_calibration_file(tmp_path)
        empty_path = tmp_path / "empty.json"
        empty_path.write_text('{"entropy": []}')

        result = runner.invoke(main, [
            "check",
            "--entropy-json", str(empty_path),
            "--calibration", str(cal_path),
        ])
        assert result.exit_code != 0

    def test_check_missing_calibration(self, runner, tmp_path):
        entropy_path = _make_entropy_json(tmp_path)

        result = runner.invoke(main, [
            "check",
            "--entropy-json", str(entropy_path),
            "--calibration", "/nonexistent/cal.json",
        ])
        assert result.exit_code != 0

    def test_score_and_check_produce_same_output(self, runner, tmp_path):
        """Both commands must produce identical JSON output."""
        cal_path = _make_calibration_file(tmp_path)
        entropy_path = _make_entropy_json(tmp_path)

        args = [
            "--entropy-json", str(entropy_path),
            "--calibration", str(cal_path),
        ]

        score_result = runner.invoke(main, ["score", *args])
        check_result = runner.invoke(main, ["check", *args])

        assert score_result.exit_code == 0
        assert check_result.exit_code == 0
        assert score_result.output == check_result.output


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------

def _make_prescored_eval_jsonl(
    tmp_path: Path,
    n: int = 20,
) -> Path:
    """Create a pre-scored eval JSONL file (ces_score + token_entropies)."""
    rng = np.random.RandomState(42)
    path = tmp_path / "prescored_eval.jsonl"
    lines = []
    for i in range(n):
        n_tokens = rng.randint(10, 40)
        entropies = rng.uniform(0.5, 3.0, size=n_tokens).tolist()
        ces = float(rng.uniform(0.1, 0.9))
        rec = {
            "ces_score": ces,
            "label": i < n // 2,  # first half faithful, second half hallucinated
            "token_count": n_tokens,
            "token_entropies": entropies,
        }
        lines.append(json.dumps(rec))
    path.write_text("\n".join(lines) + "\n")
    return path


class TestEvalCommand:
    """Tests for `sentinel eval`."""

    def test_basic_eval(self, runner, tmp_path):
        cal_path = _make_calibration_file(tmp_path)
        eval_path = _make_eval_jsonl(tmp_path)
        output_path = tmp_path / "report.json"

        result = runner.invoke(main, [
            "eval",
            "--input", str(eval_path),
            "--calibration", str(cal_path),
            "--output", str(output_path),
        ])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["status"] == "ok"
        assert data["n_samples"] == 20
        assert data["format"] == "json"
        assert output_path.exists()

    def test_eval_report_has_full_structure(self, runner, tmp_path):
        """Report from eval.py has primary_metrics, diagnostics, baselines."""
        cal_path = _make_calibration_file(tmp_path)
        eval_path = _make_eval_jsonl(tmp_path)
        output_path = tmp_path / "report.json"

        runner.invoke(main, [
            "eval",
            "--input", str(eval_path),
            "--calibration", str(cal_path),
            "--output", str(output_path),
        ])

        report = json.loads(output_path.read_text())
        # Primary metrics with bootstrap CIs
        assert "primary_metrics" in report
        pm = report["primary_metrics"]
        assert "auroc" in pm
        assert "auprc" in pm
        assert "bootstrap_auroc" in pm
        assert "bootstrap_auprc" in pm

        # Confusion matrices
        assert "confusion_matrices" in report
        assert len(report["confusion_matrices"]) > 0
        cm = report["confusion_matrices"][0]
        assert "tp" in cm
        assert "fpr" in cm
        assert "f1" in cm

        # Diagnostics
        assert "diagnostics" in report
        assert "lag1_autocorrelation" in report["diagnostics"]
        assert "calibration_coverage" in report["diagnostics"]

        # Baselines
        assert "baselines" in report

    def test_eval_prescored_input(self, runner, tmp_path):
        """Pre-scored rows work without --calibration."""
        eval_path = _make_prescored_eval_jsonl(tmp_path)
        output_path = tmp_path / "report.json"

        result = runner.invoke(main, [
            "eval",
            "--input", str(eval_path),
            "--output", str(output_path),
        ])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["status"] == "ok"
        assert data["n_samples"] == 20
        report = json.loads(output_path.read_text())
        assert "primary_metrics" in report

    def test_eval_markdown_flag(self, runner, tmp_path):
        """--markdown produces a Markdown file."""
        cal_path = _make_calibration_file(tmp_path)
        eval_path = _make_eval_jsonl(tmp_path)
        output_path = tmp_path / "report.md"

        result = runner.invoke(main, [
            "eval",
            "--input", str(eval_path),
            "--calibration", str(cal_path),
            "--output", str(output_path),
            "--markdown",
        ])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["format"] == "markdown"
        content = output_path.read_text()
        assert "# Evaluation Report" in content
        assert "AUROC" in content

    def test_eval_markdown_by_extension(self, runner, tmp_path):
        """Output with .md extension auto-selects Markdown format."""
        cal_path = _make_calibration_file(tmp_path)
        eval_path = _make_eval_jsonl(tmp_path)
        output_path = tmp_path / "report.md"

        result = runner.invoke(main, [
            "eval",
            "--input", str(eval_path),
            "--calibration", str(cal_path),
            "--output", str(output_path),
        ])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["format"] == "markdown"
        assert "# Evaluation Report" in output_path.read_text()

    def test_eval_custom_thresholds(self, runner, tmp_path):
        """--threshold options are passed through to confusion matrices."""
        cal_path = _make_calibration_file(tmp_path)
        eval_path = _make_eval_jsonl(tmp_path)
        output_path = tmp_path / "report.json"

        result = runner.invoke(main, [
            "eval",
            "--input", str(eval_path),
            "--calibration", str(cal_path),
            "--output", str(output_path),
            "--threshold", "0.3",
            "--threshold", "0.6",
        ])

        assert result.exit_code == 0
        report = json.loads(output_path.read_text())
        thresholds = [cm["threshold"] for cm in report["confusion_matrices"]]
        assert 0.3 in thresholds
        assert 0.6 in thresholds
        assert len(report["confusion_matrices"]) == 2

    def test_eval_no_bootstrap(self, runner, tmp_path):
        """--no-bootstrap skips bootstrap CIs."""
        cal_path = _make_calibration_file(tmp_path)
        eval_path = _make_eval_jsonl(tmp_path)
        output_path = tmp_path / "report.json"

        result = runner.invoke(main, [
            "eval",
            "--input", str(eval_path),
            "--calibration", str(cal_path),
            "--output", str(output_path),
            "--no-bootstrap",
        ])

        assert result.exit_code == 0
        report = json.loads(output_path.read_text())
        assert report["primary_metrics"]["bootstrap_auroc"] is None
        assert report["primary_metrics"]["bootstrap_auprc"] is None

    def test_eval_n_bootstrap(self, runner, tmp_path):
        """--n-bootstrap controls resample count."""
        cal_path = _make_calibration_file(tmp_path)
        eval_path = _make_eval_jsonl(tmp_path)
        output_path = tmp_path / "report.json"

        result = runner.invoke(main, [
            "eval",
            "--input", str(eval_path),
            "--calibration", str(cal_path),
            "--output", str(output_path),
            "--n-bootstrap", "50",
        ])

        assert result.exit_code == 0
        report = json.loads(output_path.read_text())
        boot = report["primary_metrics"]["bootstrap_auroc"]
        assert boot is not None
        assert boot["n_bootstrap"] == 50

    def test_eval_missing_calibration_for_raw_entropy(self, runner, tmp_path):
        """Raw entropy rows without --calibration should fail."""
        eval_path = _make_eval_jsonl(tmp_path)

        result = runner.invoke(main, [
            "eval",
            "--input", str(eval_path),
            "--output", str(tmp_path / "out.json"),
        ])
        assert result.exit_code != 0
        assert "calibration" in result.output.lower()

    def test_eval_missing_label(self, runner, tmp_path):
        cal_path = _make_calibration_file(tmp_path)
        path = tmp_path / "bad.jsonl"
        path.write_text('{"entropy": [1.0, 2.0, 3.0]}\n')

        result = runner.invoke(main, [
            "eval",
            "--input", str(path),
            "--calibration", str(cal_path),
            "--output", str(tmp_path / "out.json"),
        ])
        assert result.exit_code != 0
        assert "label" in result.output.lower()

    def test_eval_missing_entropy_and_ces(self, runner, tmp_path):
        """Row with neither 'entropy' nor 'ces_score' should fail."""
        cal_path = _make_calibration_file(tmp_path)
        path = tmp_path / "bad.jsonl"
        path.write_text('{"label": true}\n')

        result = runner.invoke(main, [
            "eval",
            "--input", str(path),
            "--calibration", str(cal_path),
            "--output", str(tmp_path / "out.json"),
        ])
        assert result.exit_code != 0
        assert "ces_score" in result.output.lower() or "entropy" in result.output.lower()


# ---------------------------------------------------------------------------
# score-provider (requires mocking)
# ---------------------------------------------------------------------------

class TestScoreProviderCommand:
    """Tests for `sentinel score-provider`."""

    def test_unknown_provider(self, runner, tmp_path):
        prompt_path = _make_text_file(tmp_path, "prompt.txt", "What is 2+2?")
        output_path = _make_text_file(tmp_path, "output.txt", "4")
        cal_path = _make_calibration_file(tmp_path)

        result = runner.invoke(main, [
            "score-provider",
            "--prompt", str(prompt_path),
            "--output", str(output_path),
            "--provider", "nonexistent_provider",
            "--model", "test-model",
            "--calibration", str(cal_path),
        ])
        assert result.exit_code != 0
        assert "unknown provider" in result.output.lower()

    def test_provider_no_echo_support(self, runner, tmp_path):
        """Providers without echo support should be rejected for score-output mode."""
        prompt_path = _make_text_file(tmp_path, "prompt.txt", "What is 2+2?")
        output_path = _make_text_file(tmp_path, "output.txt", "4")
        cal_path = _make_calibration_file(tmp_path)

        result = runner.invoke(main, [
            "score-provider",
            "--prompt", str(prompt_path),
            "--output", str(output_path),
            "--provider", "openai",  # openai doesn't support echo
            "--model", "gpt-4o-mini",
            "--calibration", str(cal_path),
        ])
        assert result.exit_code != 0
        assert "echo" in result.output.lower()

    def test_missing_prompt_file(self, runner, tmp_path):
        output_path = _make_text_file(tmp_path, "output.txt", "4")
        cal_path = _make_calibration_file(tmp_path)

        result = runner.invoke(main, [
            "score-provider",
            "--prompt", "/nonexistent/prompt.txt",
            "--output", str(output_path),
            "--provider", "together",
            "--model", "test-model",
            "--calibration", str(cal_path),
        ])
        assert result.exit_code != 0

    def test_score_output_mode_requires_prompt(self, runner, tmp_path):
        """--mode score-output without --prompt should fail."""
        output_path = _make_text_file(tmp_path, "output.txt", "4")
        cal_path = _make_calibration_file(tmp_path)

        result = runner.invoke(main, [
            "score-provider",
            "--output", str(output_path),
            "--provider", "together",
            "--model", "test-model",
            "--calibration", str(cal_path),
            "--mode", "score-output",
        ])
        assert result.exit_code != 0
        assert "requires --prompt" in result.output

    def test_score_output_mode_requires_output(self, runner, tmp_path):
        """--mode score-output without --output should fail."""
        prompt_path = _make_text_file(tmp_path, "prompt.txt", "What is 2+2?")
        cal_path = _make_calibration_file(tmp_path)

        result = runner.invoke(main, [
            "score-provider",
            "--prompt", str(prompt_path),
            "--provider", "together",
            "--model", "test-model",
            "--calibration", str(cal_path),
            "--mode", "score-output",
        ])
        assert result.exit_code != 0
        assert "requires --output" in result.output

    def test_score_text_mode_requires_output(self, runner, tmp_path):
        """--mode score-text without --output should fail."""
        cal_path = _make_calibration_file(tmp_path)

        result = runner.invoke(main, [
            "score-provider",
            "--provider", "together",
            "--model", "test-model",
            "--calibration", str(cal_path),
            "--mode", "score-text",
        ])
        assert result.exit_code != 0
        assert "requires --output" in result.output

    def test_generated_mode_requires_prompt(self, runner, tmp_path):
        """--mode generated without --prompt should fail."""
        cal_path = _make_calibration_file(tmp_path)

        result = runner.invoke(main, [
            "score-provider",
            "--provider", "openai",
            "--model", "gpt-4o-mini",
            "--calibration", str(cal_path),
            "--mode", "generated",
        ])
        assert result.exit_code != 0
        assert "requires --prompt" in result.output


class TestScoreProviderModeIntegration:
    """Integration tests for score-provider modes (mocked provider)."""

    @patch("hallucination_sentinel.providers.openai_compatible._call_chat_completions")
    def test_score_provider_passes_prompt_and_output(self, mock_call, runner, tmp_path):
        """--mode score-output must pass both prompt and output to score_output.

        Verifies no code path where --prompt is accepted but ignored.
        """
        mock_call.return_value = _make_topk_response()
        prompt_path = _make_text_file(tmp_path, "prompt.txt", "What is 2+2?")
        output_path = _make_text_file(tmp_path, "output.txt", "4")
        cal_path = _make_calibration_file(tmp_path)

        result = runner.invoke(main, [
            "score-provider",
            "--prompt", str(prompt_path),
            "--output", str(output_path),
            "--provider", "together",
            "--model", "test-model",
            "--calibration", str(cal_path),
            "--mode", "score-output",
        ])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["scoring_mode"] == "score-output"

        # Verify the API was called with messages containing both prompt and output
        call_kwargs = mock_call.call_args[1]
        messages = call_kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert "What is 2+2?" in messages[0]["content"]
        assert messages[1]["role"] == "assistant"
        assert "4" in messages[1]["content"]
        assert call_kwargs.get("echo") is True

    def test_score_output_fails_when_provider_lacks_echo(self, runner, tmp_path):
        """--mode score-output with a non-echo provider must fail clearly."""
        prompt_path = _make_text_file(tmp_path, "prompt.txt", "What is 2+2?")
        output_path = _make_text_file(tmp_path, "output.txt", "4")
        cal_path = _make_calibration_file(tmp_path)

        result = runner.invoke(main, [
            "score-provider",
            "--prompt", str(prompt_path),
            "--output", str(output_path),
            "--provider", "openai",
            "--model", "gpt-4o-mini",
            "--calibration", str(cal_path),
            "--mode", "score-output",
        ])
        assert result.exit_code != 0
        assert "echo" in result.output.lower()

    @patch("hallucination_sentinel.providers.openai_compatible._call_chat_completions")
    def test_generated_mode_does_not_require_output_file(self, mock_call, runner, tmp_path):
        """--mode generated must work with only --prompt (no --output)."""
        mock_call.return_value = _make_topk_response()
        prompt_path = _make_text_file(tmp_path, "prompt.txt", "Say hello")
        cal_path = _make_calibration_file(tmp_path)

        result = runner.invoke(main, [
            "score-provider",
            "--prompt", str(prompt_path),
            "--provider", "openai",
            "--model", "gpt-4o-mini",
            "--calibration", str(cal_path),
            "--mode", "generated",
        ])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["scoring_mode"] == "generated"
        assert "ces_score" in data

    @patch("hallucination_sentinel.providers.openai_compatible._call_chat_completions")
    def test_score_text_mode_does_not_require_prompt(self, mock_call, runner, tmp_path):
        """--mode score-text must work with only --output (no --prompt)."""
        mock_call.return_value = _make_topk_response()
        output_path = _make_text_file(tmp_path, "text.txt", "Some text to score")
        cal_path = _make_calibration_file(tmp_path)

        result = runner.invoke(main, [
            "score-provider",
            "--output", str(output_path),
            "--provider", "together",
            "--model", "test-model",
            "--calibration", str(cal_path),
            "--mode", "score-text",
        ])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["scoring_mode"] == "score-text"

    @patch("hallucination_sentinel.providers.openai_compatible._call_chat_completions")
    def test_scoring_mode_in_json_output(self, mock_call, runner, tmp_path):
        """JSON output must include scoring_mode field."""
        mock_call.return_value = _make_topk_response()
        prompt_path = _make_text_file(tmp_path, "prompt.txt", "Say hello")
        cal_path = _make_calibration_file(tmp_path)

        result = runner.invoke(main, [
            "score-provider",
            "--prompt", str(prompt_path),
            "--provider", "openai",
            "--model", "gpt-4o-mini",
            "--calibration", str(cal_path),
            "--mode", "generated",
        ])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "scoring_mode" in data
        assert data["scoring_mode"] == "generated"


# ---------------------------------------------------------------------------
# smoke-provider
# ---------------------------------------------------------------------------

class TestSmokeProviderCommand:
    """Tests for `sentinel smoke-provider`."""

    def test_unknown_provider(self, runner):
        result = runner.invoke(main, [
            "smoke-provider",
            "--provider", "nonexistent_provider",
        ])
        assert result.exit_code != 0
        assert "unknown provider" in result.output.lower()

    def test_known_provider_returns_json(self, runner):
        """Known provider preset should return valid JSON (even if API fails)."""
        result = runner.invoke(main, [
            "smoke-provider",
            "--provider", "together",
        ])
        # The command should succeed (exit 0) even if the API call fails,
        # because it catches exceptions and reports them.
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "provider" in data
        assert "status" in data
        # When API key is missing, error is returned instead of capabilities
        assert "capabilities" in data or "error" in data

    def test_capabilities_structure(self, runner):
        result = runner.invoke(main, [
            "smoke-provider",
            "--provider", "together",
        ])

        data = json.loads(result.output)
        # Skip if API key is missing (error response)
        if "error" in data:
            pytest.skip("API key not available")
        caps = data["capabilities"]
        assert "echo_supported" in caps
        assert "max_top_k" in caps
        assert "selected_token_logprobs" in caps
        assert "top_k_logprobs" in caps
        assert "arbitrary_text_scoring" in caps

    def test_custom_base_url(self, runner):
        """Custom base URL should be accepted."""
        result = runner.invoke(main, [
            "smoke-provider",
            "--provider", "together",
            "--base-url", "http://localhost:9999/v1",
        ])
        # Should still return valid JSON (will fail health check)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["base_url"] == "http://localhost:9999/v1"


# ---------------------------------------------------------------------------
# main group
# ---------------------------------------------------------------------------

class TestMainGroup:
    """Tests for the main CLI group."""

    def test_help(self, runner):
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Hallucination Sentinel" in result.output

    def test_version(self, runner):
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0

    def test_subcommand_help(self, runner):
        for cmd in ["calibrate", "score", "eval", "smoke-provider"]:
            result = runner.invoke(main, [cmd, "--help"])
            assert result.exit_code == 0


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

class TestEndToEndPipeline:
    """Test the full calibrate -> score -> eval pipeline."""

    def test_calibrate_then_score(self, runner, tmp_path):
        """Calibrate, then score against the calibration."""
        # Step 1: Calibrate
        cal_input = _make_calibration_jsonl(tmp_path)
        cal_output = tmp_path / "cal.json"

        result = runner.invoke(main, [
            "calibrate",
            "--input", str(cal_input),
            "--output", str(cal_output),
        ])
        assert result.exit_code == 0

        # Step 2: Score
        entropy_path = _make_entropy_json(tmp_path)

        result = runner.invoke(main, [
            "score",
            "--entropy-json", str(entropy_path),
            "--calibration", str(cal_output),
        ])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["ces_score"] >= 0
        assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL", "UNKNOWN"]

    def test_calibrate_then_eval(self, runner, tmp_path):
        """Calibrate, then run eval."""
        cal_input = _make_calibration_jsonl(tmp_path)
        cal_output = tmp_path / "cal.json"

        result = runner.invoke(main, [
            "calibrate",
            "--input", str(cal_input),
            "--output", str(cal_output),
        ])
        assert result.exit_code == 0

        eval_input = _make_eval_jsonl(tmp_path)
        eval_output = tmp_path / "report.json"

        result = runner.invoke(main, [
            "eval",
            "--input", str(eval_input),
            "--calibration", str(cal_output),
            "--output", str(eval_output),
        ])
        assert result.exit_code == 0

        report = json.loads(eval_output.read_text())
        assert "primary_metrics" in report
        assert "auroc" in report["primary_metrics"]
        assert "diagnostics" in report
        assert "lag1_autocorrelation" in report["diagnostics"]

    def test_calibrate_then_inspect(self, runner, tmp_path):
        """Calibrate, then inspect the artifact."""
        cal_input = _make_calibration_jsonl(tmp_path)
        cal_output = tmp_path / "cal.json"

        result = runner.invoke(main, [
            "calibrate",
            "--input", str(cal_input),
            "--output", str(cal_output),
        ])
        assert result.exit_code == 0

        result = runner.invoke(main, [
            "inspect-calibration",
            "--calibration", str(cal_output),
        ])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["calibration_mode"] == "unsupervised"
        assert data["token_count"] > 0
