"""Tests for the evaluation harness.

Covers:
- JSONL loading (load_eval_data)
- Calibration / evaluation set splitting
- Metric computation (AUROC, AUPRC, confusion matrices)
- Bootstrap confidence intervals
- Lag-1 autocorrelation diagnostics
- Calibration coverage by token length bucket
- Baseline comparisons
- Report serialization (JSON and Markdown)
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from hallucination_sentinel.eval import (
    BASELINE_COMPUTERS,
    LENGTH_BUCKETS,
    BaselineResult,
    BootstrapCI,
    ConfusionMetrics,
    EvalRecord,
    EvalReport,
    LengthBucketCoverage,
    _bootstrap_ci,
    _calibration_coverage,
    _compute_length_normalized_entropy,
    _compute_max_entropy,
    _compute_mean_entropy,
    _compute_perplexity_proxy,
    _compute_generation_length,
    _compute_raw_geometric_mean,
    _confusion_at_threshold,
    _lag1_autocorrelation,
    build_eval_report,
    compute_baselines,
    compute_metrics,
    load_eval_data,
    report_to_json,
    report_to_markdown,
    save_report_json,
    save_report_markdown,
    split_calibration_eval,
)


# ---------------------------------------------------------------------------
# Fixtures: toy data
# ---------------------------------------------------------------------------

def _make_records(
    n_faithful: int = 50,
    n_halluc: int = 50,
    seed: int = 42,
    faithful_ces_range: tuple[float, float] = (0.1, 0.5),
    halluc_ces_range: tuple[float, float] = (0.6, 0.95),
    faithful_token_range: tuple[int, int] = (10, 60),
    halluc_token_range: tuple[int, int] = (3, 20),
) -> list[EvalRecord]:
    """Create deterministic toy evaluation records with known separation."""
    rng = np.random.RandomState(seed)
    records = []

    for _ in range(n_faithful):
        ces = rng.uniform(*faithful_ces_range)
        n_tokens = rng.randint(*faithful_token_range)
        entropies = rng.uniform(0.5, 2.0, size=n_tokens).tolist()
        records.append(
            EvalRecord(
                ces_score=float(ces),
                label=0,
                token_count=int(n_tokens),
                token_entropies=tuple(entropies),
            )
        )

    for _ in range(n_halluc):
        ces = rng.uniform(*halluc_ces_range)
        n_tokens = rng.randint(*halluc_token_range)
        entropies = rng.uniform(1.5, 4.0, size=n_tokens).tolist()
        records.append(
            EvalRecord(
                ces_score=float(ces),
                label=1,
                token_count=int(n_tokens),
                token_entropies=tuple(entropies),
            )
        )

    return records


def _write_jsonl(records: list[EvalRecord], path: Path) -> Path:
    """Write records to a JSONL file."""
    with open(path, "w") as f:
        for r in records:
            obj = {
                "ces_score": r.ces_score,
                "label": r.label,
                "token_count": r.token_count,
                "token_entropies": list(r.token_entropies),
            }
            if r.calibrated_probability is not None:
                obj["calibrated_probability"] = r.calibrated_probability
            f.write(json.dumps(obj) + "\n")
    return path


# ---------------------------------------------------------------------------
# load_eval_data
# ---------------------------------------------------------------------------

class TestLoadEvalData:
    """Tests for JSONL loading."""

    def test_load_basic(self, tmp_path: Path):
        """Loading a valid JSONL file returns correct records."""
        records = _make_records(n_faithful=5, n_halluc=5)
        path = _write_jsonl(records, tmp_path / "eval.jsonl")
        loaded = load_eval_data(path)
        assert len(loaded) == 10
        assert loaded[0].label in (0, 1)
        assert isinstance(loaded[0].token_entropies, tuple)

    def test_load_preserves_fields(self, tmp_path: Path):
        """All fields round-trip through JSONL."""
        records = _make_records(n_faithful=3, n_halluc=2)
        path = _write_jsonl(records, tmp_path / "eval.jsonl")
        loaded = load_eval_data(path)
        for orig, ld in zip(records, loaded):
            assert ld.ces_score == orig.ces_score
            assert ld.label == orig.label
            assert ld.token_count == orig.token_count
            assert ld.token_entropies == orig.token_entropies

    def test_load_with_calibrated_probability(self, tmp_path: Path):
        """calibrated_probability field is loaded when present."""
        records = _make_records(n_faithful=2, n_halluc=2)
        # Add calibrated_probability to first record
        path = tmp_path / "eval.jsonl"
        with open(path, "w") as f:
            for i, r in enumerate(records):
                obj = {
                    "ces_score": r.ces_score,
                    "label": r.label,
                    "token_count": r.token_count,
                    "token_entropies": list(r.token_entropies),
                }
                if i == 0:
                    obj["calibrated_probability"] = 0.75
                f.write(json.dumps(obj) + "\n")
        loaded = load_eval_data(path)
        assert loaded[0].calibrated_probability == 0.75
        assert loaded[1].calibrated_probability is None

    def test_load_missing_file_raises(self):
        """Loading from nonexistent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_eval_data("/nonexistent/path/eval.jsonl")

    def test_load_missing_field_raises(self, tmp_path: Path):
        """Missing required fields raise ValueError."""
        path = tmp_path / "bad.jsonl"
        path.write_text(json.dumps({"ces_score": 0.5, "label": 1}) + "\n")
        with pytest.raises(ValueError, match="missing required fields"):
            load_eval_data(path)

    def test_load_invalid_json_raises(self, tmp_path: Path):
        """Invalid JSON on a line raises ValueError."""
        path = tmp_path / "bad.jsonl"
        path.write_text("{bad json\n")
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_eval_data(path)

    def test_load_empty_file_raises(self, tmp_path: Path):
        """An empty JSONL file raises ValueError."""
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        with pytest.raises(ValueError, match="No records"):
            load_eval_data(path)

    def test_load_skips_blank_lines(self, tmp_path: Path):
        """Blank lines are silently skipped."""
        records = _make_records(n_faithful=2, n_halluc=1)
        path = tmp_path / "eval.jsonl"
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps({
                    "ces_score": r.ces_score,
                    "label": r.label,
                    "token_count": r.token_count,
                    "token_entropies": list(r.token_entropies),
                }) + "\n")
                f.write("\n")  # blank line
        loaded = load_eval_data(path)
        assert len(loaded) == 3


# ---------------------------------------------------------------------------
# split_calibration_eval
# ---------------------------------------------------------------------------

class TestSplitCalibrationEval:
    """Tests for stratified calibration/eval splitting."""

    def test_split_preserves_total(self):
        """Total records = calibration + eval."""
        records = _make_records(n_faithful=30, n_halluc=30)
        cal, ev = split_calibration_eval(records, calibration_fraction=0.3)
        assert len(cal) + len(ev) == len(records)

    def test_split_stratified(self):
        """Both subsets maintain the hallucination base rate."""
        records = _make_records(n_faithful=40, n_halluc=10)
        cal, ev = split_calibration_eval(records, calibration_fraction=0.3)
        cal_rate = np.mean([r.label for r in cal])
        ev_rate = np.mean([r.label for r in ev])
        # Base rate is 10/50 = 0.2; allow tolerance for small samples
        assert abs(cal_rate - 0.2) < 0.2
        assert abs(ev_rate - 0.2) < 0.2

    def test_split_deterministic(self):
        """Same seed produces same split."""
        records = _make_records(n_faithful=20, n_halluc=20)
        cal1, ev1 = split_calibration_eval(records, seed=99)
        cal2, ev2 = split_calibration_eval(records, seed=99)
        assert [r.ces_score for r in cal1] == [r.ces_score for r in cal2]
        assert [r.ces_score for r in ev1] == [r.ces_score for r in ev2]

    def test_split_fraction_validation(self):
        """Invalid fraction raises ValueError."""
        records = _make_records(n_faithful=5, n_halluc=5)
        with pytest.raises(ValueError):
            split_calibration_eval(records, calibration_fraction=0.0)
        with pytest.raises(ValueError):
            split_calibration_eval(records, calibration_fraction=1.0)

    def test_split_no_disjoint(self):
        """Calibration and eval sets are disjoint."""
        records = _make_records(n_faithful=20, n_halluc=20)
        cal, ev = split_calibration_eval(records, calibration_fraction=0.3)
        cal_ids = {(r.ces_score, r.label) for r in cal}
        ev_ids = {(r.ces_score, r.label) for r in ev}
        # With continuous CES scores, overlap is astronomically unlikely
        assert len(cal_ids & ev_ids) == 0


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    """Tests for the compute_metrics function."""

    def test_perfect_separation(self):
        """Perfectly separated predictions yield AUROC = 1.0."""
        predictions = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 1.0])
        labels = np.array([0, 0, 0, 1, 1, 1])
        result = compute_metrics(predictions, labels, bootstrap=False)
        assert result["auroc"] == pytest.approx(1.0)
        assert result["auprc"] == pytest.approx(1.0)

    def test_random_separation(self):
        """Random predictions yield AUROC near 0.5."""
        rng = np.random.RandomState(42)
        predictions = rng.uniform(0, 1, size=200)
        labels = rng.randint(0, 2, size=200)
        result = compute_metrics(predictions, labels, bootstrap=False)
        # Should be somewhere between 0.3 and 0.7 for random data
        assert 0.3 <= result["auroc"] <= 0.7

    def test_inverted_predictions(self):
        """Predictions inversely correlated with labels yield AUROC < 0.5."""
        predictions = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.0])
        labels = np.array([0, 0, 0, 1, 1, 1])
        result = compute_metrics(predictions, labels, bootstrap=False)
        assert result["auroc"] == pytest.approx(0.0)

    def test_confusion_matrices_at_thresholds(self):
        """Confusion matrices are computed at each threshold."""
        predictions = np.array([0.1, 0.4, 0.6, 0.9])
        labels = np.array([0, 0, 1, 1])
        result = compute_metrics(
            predictions, labels, thresholds=[0.5, 0.7], bootstrap=False
        )
        assert len(result["confusion_matrices"]) == 2
        assert result["confusion_matrices"][0].threshold == 0.5
        assert result["confusion_matrices"][1].threshold == 0.7

    def test_confusion_matrix_values(self):
        """Confusion matrix values are correct for known data."""
        predictions = np.array([0.1, 0.6, 0.8, 0.9])
        labels = np.array([0, 0, 1, 1])
        cm = _confusion_at_threshold(predictions, labels, 0.5)
        # threshold=0.5: pred=[0,1,1,1], true=[0,0,1,1]
        assert cm.tp == 2
        assert cm.fp == 1
        assert cm.tn == 1
        assert cm.fn == 0
        assert cm.tpr == pytest.approx(1.0)
        assert cm.fpr == pytest.approx(0.5)
        assert cm.fnr == pytest.approx(0.0)

    def test_confusion_matrix_all_positive(self):
        """When all predictions are positive, FNR=0, FPR may be >0."""
        predictions = np.array([0.9, 0.8, 0.7])
        labels = np.array([1, 0, 1])
        cm = _confusion_at_threshold(predictions, labels, 0.5)
        assert cm.fn == 0
        assert cm.fnr == pytest.approx(0.0)
        assert cm.fp == 1

    def test_confusion_matrix_all_negative(self):
        """When all predictions are negative, FPR=0."""
        predictions = np.array([0.1, 0.2, 0.3])
        labels = np.array([1, 0, 1])
        cm = _confusion_at_threshold(predictions, labels, 0.5)
        assert cm.fp == 0
        assert cm.fpr == pytest.approx(0.0)
        assert cm.fn == 2

    def test_single_class_labels(self):
        """Single-class labels still produce valid confusion matrices."""
        predictions = np.array([0.1, 0.2, 0.3])
        labels = np.array([0, 0, 0])
        result = compute_metrics(predictions, labels, bootstrap=False)
        # AUROC/AUPRC undefined for single class -> None
        assert result["auroc"] is None
        assert result["auprc"] is None
        # Confusion matrices still computed
        assert len(result["confusion_matrices"]) > 0


# ---------------------------------------------------------------------------
# ConfusionMetrics dataclass
# ---------------------------------------------------------------------------

class TestConfusionMetrics:
    """Tests for the ConfusionMetrics dataclass."""

    def test_f1_score(self):
        """F1 is harmonic mean of precision and recall."""
        cm = ConfusionMetrics(
            threshold=0.5, tp=80, fp=10, tn=90, fn=20,
            tpr=0.8, fpr=0.1, fnr=0.2, precision=0.889, f1=0.0,
        )
        # Manually compute expected F1
        precision = 80 / (80 + 10)
        recall = 80 / (80 + 20)
        expected_f1 = 2 * precision * recall / (precision + recall)
        # The dataclass stores whatever we put in; test the _confusion_at_threshold logic
        predictions = np.array([0.9] * 80 + [0.6] * 10 + [0.1] * 90 + [0.3] * 20)
        labels = np.array([1] * 80 + [0] * 10 + [0] * 90 + [1] * 20)
        result = _confusion_at_threshold(predictions, labels, 0.5)
        assert result.f1 == pytest.approx(expected_f1)


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

class TestBootstrapCI:
    """Tests for bootstrap confidence interval computation."""

    def test_ci_contains_mean(self):
        """The 95% CI should contain the sample mean."""
        rng = np.random.RandomState(42)
        values = rng.normal(5.0, 1.0, size=200)
        ci = _bootstrap_ci(values, np.mean, confidence=0.95, n_bootstrap=2000)
        assert ci.lower <= ci.point_estimate <= ci.upper
        assert ci.lower <= 5.0 + 0.5  # mean should be close to 5.0

    def test_ci_width_decreases_with_n(self):
        """Larger samples produce tighter CIs."""
        rng = np.random.RandomState(42)
        small = rng.normal(0, 1, size=20)
        large = rng.normal(0, 1, size=500)
        ci_small = _bootstrap_ci(small, np.mean, n_bootstrap=1000, seed=42)
        ci_large = _bootstrap_ci(large, np.mean, n_bootstrap=1000, seed=42)
        assert (ci_large.upper - ci_large.lower) < (ci_small.upper - ci_small.lower)

    def test_ci_degenerate_single_value(self):
        """Single-value array returns point estimate as both bounds."""
        ci = _bootstrap_ci(np.array([5.0]), np.mean)
        assert ci.lower == ci.point_estimate == ci.upper == 5.0

    def test_ci_custom_statistic(self):
        """Bootstrap works with custom statistic functions."""
        values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        ci = _bootstrap_ci(values, np.median, n_bootstrap=500, seed=42)
        assert ci.point_estimate == pytest.approx(5.5)

    def test_ci_fields(self):
        """BootstrapCI has correct fields."""
        ci = _bootstrap_ci(np.array([1, 2, 3, 4, 5]), np.mean, confidence=0.90, n_bootstrap=100)
        assert ci.confidence == 0.90
        assert ci.n_bootstrap == 100


# ---------------------------------------------------------------------------
# Lag-1 autocorrelation
# ---------------------------------------------------------------------------

class TestLag1Autocorrelation:
    """Tests for lag-1 autocorrelation diagnostics."""

    def test_uncorrelated_sequence(self):
        """Independent random values have near-zero autocorrelation."""
        rng = np.random.RandomState(42)
        values = rng.uniform(0, 1, size=500)
        ac = _lag1_autocorrelation(values)
        assert abs(ac) < 0.15

    def test_perfectly_correlated(self):
        """Constant sequence has zero autocorrelation (var=0)."""
        ac = _lag1_autocorrelation(np.array([1.0, 1.0, 1.0, 1.0]))
        assert ac == 0.0

    def test_monotone_increasing(self):
        """Strictly increasing sequence has high positive autocorrelation."""
        values = np.arange(100, dtype=float)
        ac = _lag1_autocorrelation(values)
        assert ac > 0.9

    def test_alternating(self):
        """Alternating sequence has negative autocorrelation."""
        values = np.array([0.0, 1.0] * 50)
        ac = _lag1_autocorrelation(values)
        assert ac < -0.5

    def test_short_sequence(self):
        """Sequence shorter than 2 returns 0.0."""
        assert _lag1_autocorrelation(np.array([1.0])) == 0.0
        assert _lag1_autocorrelation(np.array([])) == 0.0


# ---------------------------------------------------------------------------
# Calibration coverage by token length bucket
# ---------------------------------------------------------------------------

class TestCalibrationCoverage:
    """Tests for length-bucket calibration coverage."""

    def test_all_buckets_present(self):
        """Result contains all defined length buckets."""
        records = _make_records(n_faithful=10, n_halluc=10)
        coverage = _calibration_coverage(records)
        assert len(coverage) == len(LENGTH_BUCKETS)

    def test_bucket_names(self):
        """Bucket names match LENGTH_BUCKETS definitions."""
        records = _make_records(n_faithful=5, n_halluc=5)
        coverage = _calibration_coverage(records)
        names = [c.bucket_name for c in coverage]
        expected = [b[0] for b in LENGTH_BUCKETS]
        assert names == expected

    def test_counts_sum_to_total(self):
        """Total count across buckets equals total records."""
        records = _make_records(n_faithful=20, n_halluc=20)
        coverage = _calibration_coverage(records)
        total = sum(c.count for c in coverage)
        assert total == len(records)

    def test_empty_bucket(self):
        """Buckets with no records have count=0 and zero rates."""
        # All records with token_count=50 -> only "50-99" bucket populated
        records = [
            EvalRecord(ces_score=0.5, label=0, token_count=50, token_entropies=(1.0,) * 50),
            EvalRecord(ces_score=0.8, label=1, token_count=75, token_entropies=(2.0,) * 75),
        ]
        coverage = _calibration_coverage(records)
        for c in coverage:
            if c.bucket_name == "50-99":
                assert c.count == 2
            else:
                assert c.count == 0
                assert c.mean_ces == 0.0
                assert c.hallucination_rate == 0.0

    def test_hallucination_rate_in_bucket(self):
        """Hallucination rate reflects label distribution within a bucket."""
        # 2 faithful + 1 hallucination, all in "10-49" bucket
        records = [
            EvalRecord(ces_score=0.3, label=0, token_count=15, token_entropies=(1.0,) * 15),
            EvalRecord(ces_score=0.4, label=0, token_count=20, token_entropies=(1.0,) * 20),
            EvalRecord(ces_score=0.9, label=1, token_count=25, token_entropies=(2.0,) * 25),
        ]
        coverage = _calibration_coverage(records)
        bucket_10_49 = next(c for c in coverage if c.bucket_name == "10-49")
        assert bucket_10_49.count == 3
        assert bucket_10_49.hallucination_rate == pytest.approx(1 / 3)

    def test_short_tokens_bucket(self):
        """Records with <5 tokens fall into the '<5' bucket."""
        records = [
            EvalRecord(ces_score=0.5, label=0, token_count=3, token_entropies=(1.0,) * 3),
        ]
        coverage = _calibration_coverage(records)
        bucket_lt5 = next(c for c in coverage if c.bucket_name == "<5")
        assert bucket_lt5.count == 1

    def test_long_tokens_bucket(self):
        """Records with >=100 tokens fall into the '>=100' bucket."""
        records = [
            EvalRecord(ces_score=0.5, label=0, token_count=150, token_entropies=(1.0,) * 150),
        ]
        coverage = _calibration_coverage(records)
        bucket_gte100 = next(c for c in coverage if c.bucket_name == ">=100")
        assert bucket_gte100.count == 1


# ---------------------------------------------------------------------------
# Baseline comparisons
# ---------------------------------------------------------------------------

class TestBaselines:
    """Tests for baseline method computations."""

    def test_all_baselines_present(self):
        """All expected baseline methods are computed."""
        records = _make_records(n_faithful=20, n_halluc=20)
        labels = np.array([r.label for r in records])
        baselines = compute_baselines(records, labels, bootstrap=False)
        expected_names = set(BASELINE_COMPUTERS.keys())
        assert set(baselines.keys()) == expected_names

    def test_baseline_predictions_shape(self):
        """Each baseline produces one prediction per record."""
        records = _make_records(n_faithful=10, n_halluc=10)
        labels = np.array([r.label for r in records])
        baselines = compute_baselines(records, labels, bootstrap=False)
        for name, bl in baselines.items():
            assert bl.predictions.shape == (20,), f"{name} predictions shape mismatch"

    def test_mean_entropy_baseline(self):
        """Mean entropy baseline equals per-record mean of token entropies."""
        records = _make_records(n_faithful=5, n_halluc=5)
        preds = _compute_mean_entropy(records)
        for i, r in enumerate(records):
            assert preds[i] == pytest.approx(np.mean(r.token_entropies))

    def test_max_entropy_baseline(self):
        """Max entropy baseline equals per-record max of token entropies."""
        records = _make_records(n_faithful=5, n_halluc=5)
        preds = _compute_max_entropy(records)
        for i, r in enumerate(records):
            assert preds[i] == pytest.approx(np.max(r.token_entropies))

    def test_generation_length_baseline(self):
        """Generation length baseline equals token_count."""
        records = _make_records(n_faithful=5, n_halluc=5)
        preds = _compute_generation_length(records)
        for i, r in enumerate(records):
            assert preds[i] == float(r.token_count)

    def test_perplexity_proxy_baseline(self):
        """Perplexity proxy = exp(mean(token_entropies))."""
        records = _make_records(n_faithful=5, n_halluc=5)
        preds = _compute_perplexity_proxy(records)
        for i, r in enumerate(records):
            expected = float(np.exp(np.mean(r.token_entropies)))
            assert preds[i] == pytest.approx(expected)

    def test_length_normalized_entropy_baseline(self):
        """LN-entropy equals mean of token entropies."""
        records = _make_records(n_faithful=5, n_halluc=5)
        preds = _compute_length_normalized_entropy(records)
        mean_preds = _compute_mean_entropy(records)
        np.testing.assert_allclose(preds, mean_preds)

    def test_raw_geometric_mean_baseline(self):
        """Raw geometric mean produces values in [0, 1]."""
        records = _make_records(n_faithful=20, n_halluc=20)
        preds = _compute_raw_geometric_mean(records)
        assert np.all(preds >= 0.0)
        assert np.all(preds <= 1.0)

    def test_baseline_auroc_with_good_separation(self):
        """Mean entropy baseline achieves reasonable AUROC with separated data."""
        records = _make_records(
            n_faithful=50, n_halluc=50,
            faithful_ces_range=(0.1, 0.4),
            halluc_ces_range=(0.7, 0.95),
        )
        labels = np.array([r.label for r in records])
        baselines = compute_baselines(records, labels, bootstrap=False)
        # Mean entropy should have some discriminative power
        assert baselines["mean_entropy"].auroc is not None
        assert baselines["mean_entropy"].auroc > 0.5


# ---------------------------------------------------------------------------
# build_eval_report (full pipeline)
# ---------------------------------------------------------------------------

class TestBuildEvalReport:
    """Tests for the full report building pipeline."""

    def test_report_fields(self):
        """Report contains all expected fields."""
        records = _make_records(n_faithful=30, n_halluc=30)
        report = build_eval_report(records, bootstrap=False)
        assert report.n_samples == 60
        assert report.n_positive == 30
        assert report.n_negative == 30
        assert report.hallucination_rate == pytest.approx(0.5)
        assert report.method == "CES"
        assert report.created_at != ""

    def test_report_auroc_with_separated_data(self):
        """Well-separated data yields high AUROC."""
        records = _make_records(
            n_faithful=50, n_halluc=50,
            faithful_ces_range=(0.05, 0.3),
            halluc_ces_range=(0.7, 0.95),
        )
        report = build_eval_report(records, bootstrap=False)
        assert report.auroc is not None
        assert report.auroc > 0.9

    def test_report_with_bootstrap(self):
        """Report includes bootstrap CIs when enabled."""
        records = _make_records(n_faithful=30, n_halluc=30)
        report = build_eval_report(records, bootstrap=True, n_bootstrap=100)
        assert report.bootstrap_auroc is not None
        assert report.bootstrap_auprc is not None
        assert report.bootstrap_auroc.lower <= report.bootstrap_auroc.upper

    def test_report_confusion_matrices(self):
        """Report includes confusion matrices at default thresholds."""
        records = _make_records(n_faithful=30, n_halluc=30)
        report = build_eval_report(records, bootstrap=False)
        assert len(report.confusion_matrices) == 4  # default thresholds

    def test_report_custom_thresholds(self):
        """Custom thresholds are used for confusion matrices."""
        records = _make_records(n_faithful=20, n_halluc=20)
        report = build_eval_report(records, thresholds=[0.3, 0.6], bootstrap=False)
        assert len(report.confusion_matrices) == 2
        assert report.confusion_matrices[0].threshold == 0.3
        assert report.confusion_matrices[1].threshold == 0.6

    def test_report_lag1_autocorrelation(self):
        """Report includes lag-1 autocorrelation."""
        records = _make_records(n_faithful=30, n_halluc=30)
        report = build_eval_report(records, bootstrap=False)
        assert isinstance(report.lag1_autocorrelation, float)

    def test_report_calibration_coverage(self):
        """Report includes calibration coverage for all buckets."""
        records = _make_records(n_faithful=20, n_halluc=20)
        report = build_eval_report(records, bootstrap=False)
        assert len(report.calibration_coverage) == len(LENGTH_BUCKETS)

    def test_report_baselines(self):
        """Report includes all baseline results."""
        records = _make_records(n_faithful=20, n_halluc=20)
        report = build_eval_report(records, bootstrap=False)
        assert set(report.baselines.keys()) == set(BASELINE_COMPUTERS.keys())


# ---------------------------------------------------------------------------
# Report serialization: JSON
# ---------------------------------------------------------------------------

class TestReportJSON:
    """Tests for JSON report serialization."""

    def test_json_is_valid(self):
        """Serialized JSON is parseable."""
        records = _make_records(n_faithful=10, n_halluc=10)
        report = build_eval_report(records, bootstrap=False)
        json_str = report_to_json(report)
        data = json.loads(json_str)
        assert "method" in data
        assert "dataset" in data
        assert "primary_metrics" in data

    def test_json_dataset_fields(self):
        """JSON dataset section has correct counts."""
        records = _make_records(n_faithful=15, n_halluc=10)
        report = build_eval_report(records, bootstrap=False)
        data = json.loads(report_to_json(report))
        assert data["dataset"]["n_samples"] == 25
        assert data["dataset"]["n_positive"] == 10
        assert data["dataset"]["n_negative"] == 15

    def test_json_confusion_matrices(self):
        """JSON includes confusion matrix data."""
        records = _make_records(n_faithful=10, n_halluc=10)
        report = build_eval_report(records, bootstrap=False)
        data = json.loads(report_to_json(report))
        assert len(data["confusion_matrices"]) > 0
        cm = data["confusion_matrices"][0]
        assert "tp" in cm
        assert "fpr" in cm
        assert "fnr" in cm

    def test_json_baselines(self):
        """JSON includes baseline comparison data."""
        records = _make_records(n_faithful=10, n_halluc=10)
        report = build_eval_report(records, bootstrap=False)
        data = json.loads(report_to_json(report))
        assert "baselines" in data
        assert "mean_entropy" in data["baselines"]

    def test_json_diagnostics(self):
        """JSON includes diagnostics section."""
        records = _make_records(n_faithful=10, n_halluc=10)
        report = build_eval_report(records, bootstrap=False)
        data = json.loads(report_to_json(report))
        assert "diagnostics" in data
        assert "lag1_autocorrelation" in data["diagnostics"]
        assert "calibration_coverage" in data["diagnostics"]

    def test_save_report_json(self, tmp_path: Path):
        """save_report_json creates a valid JSON file."""
        records = _make_records(n_faithful=10, n_halluc=10)
        report = build_eval_report(records, bootstrap=False)
        path = tmp_path / "report.json"
        save_report_json(report, path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["method"] == "CES"


# ---------------------------------------------------------------------------
# Report serialization: Markdown
# ---------------------------------------------------------------------------

class TestReportMarkdown:
    """Tests for Markdown report serialization."""

    def test_markdown_contains_header(self):
        """Markdown contains the method name in the header."""
        records = _make_records(n_faithful=10, n_halluc=10)
        report = build_eval_report(records, bootstrap=False)
        md = report_to_markdown(report)
        assert "# Evaluation Report: CES" in md

    def test_markdown_contains_dataset(self):
        """Markdown contains dataset summary table."""
        records = _make_records(n_faithful=10, n_halluc=10)
        report = build_eval_report(records, bootstrap=False)
        md = report_to_markdown(report)
        assert "## Dataset" in md
        assert "Total samples" in md

    def test_markdown_contains_metrics(self):
        """Markdown contains primary metrics section."""
        records = _make_records(n_faithful=10, n_halluc=10)
        report = build_eval_report(records, bootstrap=False)
        md = report_to_markdown(report)
        assert "## Primary Metrics" in md
        assert "AUROC" in md
        assert "AUPRC" in md

    def test_markdown_contains_confusion(self):
        """Markdown contains confusion matrix table."""
        records = _make_records(n_faithful=10, n_halluc=10)
        report = build_eval_report(records, bootstrap=False)
        md = report_to_markdown(report)
        assert "## Confusion Matrices" in md
        assert "Threshold" in md

    def test_markdown_contains_diagnostics(self):
        """Markdown contains diagnostics section."""
        records = _make_records(n_faithful=10, n_halluc=10)
        report = build_eval_report(records, bootstrap=False)
        md = report_to_markdown(report)
        assert "## Diagnostics" in md
        assert "Lag-1 autocorrelation" in md

    def test_markdown_contains_baselines(self):
        """Markdown contains baseline comparison table."""
        records = _make_records(n_faithful=10, n_halluc=10)
        report = build_eval_report(records, bootstrap=False)
        md = report_to_markdown(report)
        assert "## Baseline Comparisons" in md
        assert "mean_entropy" in md

    def test_markdown_contains_length_buckets(self):
        """Markdown contains calibration coverage by length bucket."""
        records = _make_records(n_faithful=10, n_halluc=10)
        report = build_eval_report(records, bootstrap=False)
        md = report_to_markdown(report)
        assert "Calibration Coverage by Token Length" in md
        for bucket_name, _, _ in LENGTH_BUCKETS:
            assert bucket_name in md

    def test_save_report_markdown(self, tmp_path: Path):
        """save_report_markdown creates a valid Markdown file."""
        records = _make_records(n_faithful=10, n_halluc=10)
        report = build_eval_report(records, bootstrap=False)
        path = tmp_path / "report.md"
        save_report_markdown(report, path)
        assert path.exists()
        content = path.read_text()
        assert "# Evaluation Report" in content


# ---------------------------------------------------------------------------
# Integration: full pipeline from JSONL
# ---------------------------------------------------------------------------

class TestIntegration:
    """End-to-end tests from JSONL file to report."""

    def test_full_pipeline(self, tmp_path: Path):
        """Complete pipeline: load -> split -> eval -> report."""
        # Generate and save records
        records = _make_records(n_faithful=40, n_halluc=40, seed=123)
        jsonl_path = _write_jsonl(records, tmp_path / "eval.jsonl")

        # Load
        loaded = load_eval_data(jsonl_path)
        assert len(loaded) == 80

        # Split
        cal_records, eval_records = split_calibration_eval(loaded, calibration_fraction=0.3, seed=42)
        assert len(cal_records) > 0
        assert len(eval_records) > 0
        assert len(cal_records) + len(eval_records) == len(loaded)

        # Build report on eval set
        report = build_eval_report(eval_records, bootstrap=False)

        # Verify report structure
        assert report.n_samples == len(eval_records)
        assert report.auroc is not None
        assert len(report.confusion_matrices) > 0
        assert len(report.baselines) > 0
        assert len(report.calibration_coverage) == len(LENGTH_BUCKETS)

        # Serialize
        json_str = report_to_json(report)
        md_str = report_to_markdown(report)
        assert len(json_str) > 0
        assert len(md_str) > 0

        # Save
        save_report_json(report, tmp_path / "report.json")
        save_report_markdown(report, tmp_path / "report.md")
        assert (tmp_path / "report.json").exists()
        assert (tmp_path / "report.md").exists()
