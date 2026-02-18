"""Tests for E-FD2 metrics module.

Computes all evaluation metrics for the 9-arm fusion experiment:
    - PR curve data (precision, recall, thresholds) + PR-AUC
    - FPR at fixed TPR levels (0.90, 0.95, 0.99)
    - Escalation analysis (three-way strategies only)
    - Cost-sensitive expected cost
    - McNemar's test for pairwise significance
    - Bootstrap confidence intervals on metric differences
"""

import math

import numpy as np
import pytest

from slfd.decision import Decision


# ===================================================================
# Imports that WILL FAIL until we implement metrics.py
# ===================================================================
from slfd.metrics import (
    # PR curve
    PRCurveData,
    compute_pr_curve,
    # FPR at fixed TPR
    FPRAtTPR,
    compute_fpr_at_tpr,
    # Escalation analysis
    EscalationAnalysis,
    compute_escalation_analysis,
    # Cost-sensitive
    CostConfig,
    compute_expected_cost,
    # McNemar's test
    McNemarResult,
    mcnemar_test,
    # Bootstrap CI
    BootstrapCIResult,
    bootstrap_metric_ci,
    # Convenience: all-in-one arm evaluation
    ArmResult,
    evaluate_arm,
)


# ===================================================================
# Shared fixtures
# ===================================================================

@pytest.fixture
def binary_labels():
    """100 labels with ~30% fraud rate."""
    rng = np.random.default_rng(42)
    return (rng.uniform(size=100) < 0.3).astype(np.int32)


@pytest.fixture
def good_scores(binary_labels):
    """Scores that correlate with true labels (decent model)."""
    rng = np.random.default_rng(123)
    noise = rng.normal(0, 0.15, size=100)
    scores = binary_labels.astype(np.float64) * 0.7 + 0.15 + noise
    return np.clip(scores, 0.0, 1.0)


@pytest.fixture
def random_scores():
    """Uninformative random scores."""
    rng = np.random.default_rng(99)
    return rng.uniform(0.0, 1.0, size=100)


@pytest.fixture
def perfect_scores(binary_labels):
    """Perfect separation — fraud=1.0, legit=0.0."""
    return binary_labels.astype(np.float64)


# ===================================================================
# PR Curve
# ===================================================================

class TestPRCurveData:
    """Container for precision-recall curve data."""

    def test_has_required_fields(self):
        data = PRCurveData(
            precision=np.array([1.0, 0.5]),
            recall=np.array([0.5, 1.0]),
            thresholds=np.array([0.8, 0.2]),
            auc=0.75,
            f1_max=0.67,
            f1_max_threshold=0.5,
        )
        assert data.auc == 0.75
        assert data.f1_max == 0.67
        assert data.f1_max_threshold == 0.5

    def test_precision_recall_same_length(self):
        data = PRCurveData(
            precision=np.array([1.0, 0.8, 0.5]),
            recall=np.array([0.3, 0.6, 1.0]),
            thresholds=np.array([0.9, 0.5]),
            auc=0.7,
            f1_max=0.6,
            f1_max_threshold=0.5,
        )
        assert len(data.precision) == len(data.recall)


class TestComputePRCurve:
    """compute_pr_curve(y_true, scores) → PRCurveData."""

    def test_returns_pr_curve_data(self, binary_labels, good_scores):
        result = compute_pr_curve(binary_labels, good_scores)
        assert isinstance(result, PRCurveData)

    def test_auc_in_valid_range(self, binary_labels, good_scores):
        result = compute_pr_curve(binary_labels, good_scores)
        assert 0.0 <= result.auc <= 1.0

    def test_perfect_scores_high_auc(self, binary_labels, perfect_scores):
        result = compute_pr_curve(binary_labels, perfect_scores)
        assert result.auc > 0.95

    def test_random_scores_lower_auc(self, binary_labels, random_scores, good_scores):
        good = compute_pr_curve(binary_labels, good_scores)
        rand = compute_pr_curve(binary_labels, random_scores)
        assert good.auc > rand.auc

    def test_f1_max_positive(self, binary_labels, good_scores):
        result = compute_pr_curve(binary_labels, good_scores)
        assert result.f1_max > 0.0

    def test_f1_max_threshold_in_zero_one(self, binary_labels, good_scores):
        result = compute_pr_curve(binary_labels, good_scores)
        assert 0.0 <= result.f1_max_threshold <= 1.0

    def test_precision_recall_arrays_non_empty(self, binary_labels, good_scores):
        result = compute_pr_curve(binary_labels, good_scores)
        assert len(result.precision) > 0
        assert len(result.recall) > 0

    def test_deterministic(self, binary_labels, good_scores):
        r1 = compute_pr_curve(binary_labels, good_scores)
        r2 = compute_pr_curve(binary_labels, good_scores)
        assert r1.auc == pytest.approx(r2.auc)


# ===================================================================
# FPR at fixed TPR
# ===================================================================

class TestFPRAtTPR:
    """Container for FPR at fixed TPR results."""

    def test_has_required_fields(self):
        result = FPRAtTPR(
            tpr_targets=np.array([0.90, 0.95, 0.99]),
            fpr_values=np.array([0.10, 0.20, 0.50]),
            thresholds=np.array([0.6, 0.4, 0.1]),
        )
        assert result.tpr_targets.shape == (3,)
        assert result.fpr_values.shape == (3,)
        assert result.thresholds.shape == (3,)


class TestComputeFPRAtTPR:
    """compute_fpr_at_tpr(y_true, scores, tpr_targets) → FPRAtTPR."""

    def test_returns_fpr_at_tpr(self, binary_labels, good_scores):
        result = compute_fpr_at_tpr(
            binary_labels, good_scores,
            tpr_targets=np.array([0.90, 0.95, 0.99]),
        )
        assert isinstance(result, FPRAtTPR)

    def test_default_tpr_targets(self, binary_labels, good_scores):
        """Should default to [0.90, 0.95, 0.99] if not specified."""
        result = compute_fpr_at_tpr(binary_labels, good_scores)
        np.testing.assert_array_almost_equal(
            result.tpr_targets, [0.90, 0.95, 0.99],
        )

    def test_fpr_values_in_valid_range(self, binary_labels, good_scores):
        result = compute_fpr_at_tpr(binary_labels, good_scores)
        assert np.all(result.fpr_values >= 0.0)
        assert np.all(result.fpr_values <= 1.0)

    def test_higher_tpr_requires_higher_fpr(self, binary_labels, good_scores):
        """To catch more fraud (higher TPR), you generally accept more false alarms."""
        result = compute_fpr_at_tpr(
            binary_labels, good_scores,
            tpr_targets=np.array([0.80, 0.95]),
        )
        # FPR at TPR=0.95 should be >= FPR at TPR=0.80
        assert result.fpr_values[1] >= result.fpr_values[0] - 1e-9

    def test_perfect_scores_low_fpr(self, binary_labels, perfect_scores):
        """Perfect model should achieve low FPR even at high TPR."""
        result = compute_fpr_at_tpr(binary_labels, perfect_scores)
        assert result.fpr_values[0] < 0.1  # FPR at TPR=0.90

    def test_deterministic(self, binary_labels, good_scores):
        r1 = compute_fpr_at_tpr(binary_labels, good_scores)
        r2 = compute_fpr_at_tpr(binary_labels, good_scores)
        np.testing.assert_array_equal(r1.fpr_values, r2.fpr_values)


# ===================================================================
# Escalation Analysis
# ===================================================================

class TestEscalationAnalysis:
    """Container for three-way decision escalation metrics."""

    def test_has_required_fields(self):
        ea = EscalationAnalysis(
            escalation_rate=0.2,
            fraud_rate_escalated=0.15,
            fraud_rate_auto_decided=0.03,
            n_escalated=20,
            n_auto_decided=80,
            precision_auto_decided=0.85,
            recall_auto_decided=0.70,
        )
        assert ea.escalation_rate == 0.2
        assert ea.n_escalated == 20

    def test_enrichment_ratio(self):
        """Escalated fraud rate / overall fraud rate — measures signal quality."""
        ea = EscalationAnalysis(
            escalation_rate=0.2,
            fraud_rate_escalated=0.30,
            fraud_rate_auto_decided=0.02,
            n_escalated=20,
            n_auto_decided=80,
            precision_auto_decided=0.85,
            recall_auto_decided=0.70,
        )
        # If overall fraud rate is ~0.08, enrichment = 0.30/0.08 ≈ 3.75
        assert ea.fraud_rate_escalated > ea.fraud_rate_auto_decided


class TestComputeEscalationAnalysis:
    """compute_escalation_analysis(y_true, scores, escalation_mask) → EscalationAnalysis."""

    def _make_escalation_mask(self, n, escalation_rate=0.2, seed=42):
        rng = np.random.default_rng(seed)
        return rng.uniform(size=n) < escalation_rate

    def test_returns_escalation_analysis(self, binary_labels, good_scores):
        mask = self._make_escalation_mask(100)
        result = compute_escalation_analysis(
            binary_labels, good_scores, mask,
        )
        assert isinstance(result, EscalationAnalysis)

    def test_escalation_rate_matches_mask(self, binary_labels, good_scores):
        mask = np.array([True] * 25 + [False] * 75)
        result = compute_escalation_analysis(binary_labels, good_scores, mask)
        assert result.escalation_rate == pytest.approx(0.25)

    def test_counts_sum_to_total(self, binary_labels, good_scores):
        mask = self._make_escalation_mask(100)
        result = compute_escalation_analysis(binary_labels, good_scores, mask)
        assert result.n_escalated + result.n_auto_decided == 100

    def test_fraud_rates_non_negative(self, binary_labels, good_scores):
        mask = self._make_escalation_mask(100)
        result = compute_escalation_analysis(binary_labels, good_scores, mask)
        assert result.fraud_rate_escalated >= 0.0
        assert result.fraud_rate_auto_decided >= 0.0

    def test_all_escalated(self, binary_labels, good_scores):
        """If everything is escalated, auto-decided metrics should be None/NaN or 0."""
        mask = np.ones(100, dtype=bool)
        result = compute_escalation_analysis(binary_labels, good_scores, mask)
        assert result.escalation_rate == pytest.approx(1.0)
        assert result.n_auto_decided == 0

    def test_none_escalated(self, binary_labels, good_scores):
        """If nothing is escalated, escalated fraud rate should be None/NaN or 0."""
        mask = np.zeros(100, dtype=bool)
        result = compute_escalation_analysis(binary_labels, good_scores, mask)
        assert result.escalation_rate == pytest.approx(0.0)
        assert result.n_escalated == 0

    def test_precision_recall_auto_decided_valid(self, binary_labels, good_scores):
        """Precision and recall on auto-decided subset should be in [0, 1]."""
        mask = self._make_escalation_mask(100)
        result = compute_escalation_analysis(binary_labels, good_scores, mask)
        if result.n_auto_decided > 0:
            assert 0.0 <= result.precision_auto_decided <= 1.0
            assert 0.0 <= result.recall_auto_decided <= 1.0


# ===================================================================
# Cost-Sensitive Metric
# ===================================================================

class TestCostConfig:
    """Configuration for cost-sensitive evaluation."""

    def test_default_values(self):
        """Paper specifies: $2/review, $50/missed fraud, $0.50/false block."""
        cfg = CostConfig()
        assert cfg.review_cost == 2.0
        assert cfg.missed_fraud_cost == 50.0
        assert cfg.false_block_cost == 0.50

    def test_custom_values(self):
        cfg = CostConfig(
            review_cost=5.0,
            missed_fraud_cost=100.0,
            false_block_cost=1.0,
        )
        assert cfg.review_cost == 5.0

    def test_negative_cost_raises(self):
        with pytest.raises(ValueError):
            CostConfig(review_cost=-1.0)


class TestComputeExpectedCost:
    """compute_expected_cost(y_true, predictions, escalation_mask, cost_config)."""

    def test_returns_float(self, binary_labels, good_scores):
        cost = compute_expected_cost(
            binary_labels, good_scores,
            threshold=0.5,
        )
        assert isinstance(cost, float)

    def test_cost_non_negative(self, binary_labels, good_scores):
        cost = compute_expected_cost(
            binary_labels, good_scores,
            threshold=0.5,
        )
        assert cost >= 0.0

    def test_perfect_model_zero_cost(self, binary_labels, perfect_scores):
        """Perfect predictions with optimal threshold → zero error cost."""
        cost = compute_expected_cost(
            binary_labels, perfect_scores,
            threshold=0.5,
        )
        assert cost == pytest.approx(0.0, abs=1e-9)

    def test_with_escalation(self, binary_labels, good_scores):
        """Cost with escalation should include review costs."""
        mask = np.array([True] * 20 + [False] * 80)
        cost_no_esc = compute_expected_cost(
            binary_labels, good_scores,
            threshold=0.5,
        )
        cost_with_esc = compute_expected_cost(
            binary_labels, good_scores,
            threshold=0.5,
            escalation_mask=mask,
        )
        # Escalation adds review costs but may reduce error costs
        # We just verify it returns a valid number
        assert isinstance(cost_with_esc, float)
        assert cost_with_esc >= 0.0

    def test_higher_missed_fraud_cost_penalizes_fn(self, binary_labels, good_scores):
        """Higher missed fraud cost → higher total cost (since some fraud is missed)."""
        cfg_low = CostConfig(review_cost=2.0, missed_fraud_cost=10.0, false_block_cost=0.5)
        cfg_high = CostConfig(review_cost=2.0, missed_fraud_cost=500.0, false_block_cost=0.5)
        cost_low = compute_expected_cost(
            binary_labels, good_scores, threshold=0.5, cost_config=cfg_low,
        )
        cost_high = compute_expected_cost(
            binary_labels, good_scores, threshold=0.5, cost_config=cfg_high,
        )
        assert cost_high >= cost_low

    def test_custom_cost_config(self, binary_labels, good_scores):
        cfg = CostConfig(review_cost=10.0, missed_fraud_cost=200.0, false_block_cost=5.0)
        cost = compute_expected_cost(
            binary_labels, good_scores, threshold=0.5, cost_config=cfg,
        )
        assert cost >= 0.0


# ===================================================================
# McNemar's Test
# ===================================================================

class TestMcNemarResult:
    """Container for McNemar's test output."""

    def test_has_required_fields(self):
        result = McNemarResult(
            statistic=3.5,
            p_value=0.06,
            n_discordant_a=15,
            n_discordant_b=25,
        )
        assert result.statistic == 3.5
        assert result.p_value == 0.06


class TestMcNemarTest:
    """mcnemar_test(y_true, scores_a, scores_b, threshold)."""

    def test_returns_mcnemar_result(self, binary_labels, good_scores, random_scores):
        result = mcnemar_test(
            binary_labels, good_scores, random_scores, threshold=0.5,
        )
        assert isinstance(result, McNemarResult)

    def test_p_value_in_valid_range(self, binary_labels, good_scores, random_scores):
        result = mcnemar_test(
            binary_labels, good_scores, random_scores, threshold=0.5,
        )
        assert 0.0 <= result.p_value <= 1.0

    def test_identical_scores_high_p_value(self, binary_labels, good_scores):
        """Same scores should give p ≈ 1 (no difference)."""
        result = mcnemar_test(
            binary_labels, good_scores, good_scores, threshold=0.5,
        )
        assert result.p_value > 0.5

    def test_statistic_non_negative(self, binary_labels, good_scores, random_scores):
        result = mcnemar_test(
            binary_labels, good_scores, random_scores, threshold=0.5,
        )
        assert result.statistic >= 0.0

    def test_discordant_counts_non_negative(self, binary_labels, good_scores, random_scores):
        result = mcnemar_test(
            binary_labels, good_scores, random_scores, threshold=0.5,
        )
        assert result.n_discordant_a >= 0
        assert result.n_discordant_b >= 0

    def test_good_vs_random_likely_significant(self, binary_labels, good_scores, random_scores):
        """A good model vs random should show statistical difference."""
        result = mcnemar_test(
            binary_labels, good_scores, random_scores, threshold=0.5,
        )
        # Not guaranteed, but likely with these fixtures
        assert result.n_discordant_a + result.n_discordant_b > 0


# ===================================================================
# Bootstrap CI
# ===================================================================

class TestBootstrapCIResult:
    """Container for bootstrap confidence interval."""

    def test_has_required_fields(self):
        result = BootstrapCIResult(
            metric_name="f1",
            observed_diff=0.05,
            ci_lower=-0.01,
            ci_upper=0.11,
            p_value=0.04,
            n_bootstrap=1000,
        )
        assert result.ci_lower < result.ci_upper
        assert result.n_bootstrap == 1000

    def test_is_significant_property(self):
        """Significant if CI does not contain zero."""
        sig = BootstrapCIResult(
            metric_name="f1", observed_diff=0.05,
            ci_lower=0.01, ci_upper=0.09, p_value=0.02, n_bootstrap=1000,
        )
        assert sig.is_significant is True

        not_sig = BootstrapCIResult(
            metric_name="f1", observed_diff=0.05,
            ci_lower=-0.02, ci_upper=0.12, p_value=0.15, n_bootstrap=1000,
        )
        assert not_sig.is_significant is False


class TestBootstrapMetricCI:
    """bootstrap_metric_ci(y_true, scores_a, scores_b, metric_fn, ...)."""

    def test_returns_bootstrap_result(self, binary_labels, good_scores, random_scores):
        result = bootstrap_metric_ci(
            binary_labels, good_scores, random_scores,
            metric_name="pr_auc",
            n_bootstrap=200,
            seed=42,
        )
        assert isinstance(result, BootstrapCIResult)

    def test_ci_lower_leq_upper(self, binary_labels, good_scores, random_scores):
        result = bootstrap_metric_ci(
            binary_labels, good_scores, random_scores,
            metric_name="pr_auc",
            n_bootstrap=200,
            seed=42,
        )
        assert result.ci_lower <= result.ci_upper

    def test_observed_diff_between_bounds(self, binary_labels, good_scores, random_scores):
        """Observed difference should typically be within the CI."""
        result = bootstrap_metric_ci(
            binary_labels, good_scores, random_scores,
            metric_name="pr_auc",
            n_bootstrap=500,
            seed=42,
        )
        # Allow slight tolerance for edge cases
        assert result.ci_lower <= result.observed_diff + 0.05
        assert result.ci_upper >= result.observed_diff - 0.05

    def test_identical_scores_diff_near_zero(self, binary_labels, good_scores):
        """Same scores → observed diff ≈ 0, CI contains 0."""
        result = bootstrap_metric_ci(
            binary_labels, good_scores, good_scores,
            metric_name="pr_auc",
            n_bootstrap=200,
            seed=42,
        )
        assert abs(result.observed_diff) < 1e-9

    def test_supports_f1_metric(self, binary_labels, good_scores, random_scores):
        result = bootstrap_metric_ci(
            binary_labels, good_scores, random_scores,
            metric_name="f1",
            n_bootstrap=200,
            seed=42,
        )
        assert result.metric_name == "f1"

    def test_deterministic_with_seed(self, binary_labels, good_scores, random_scores):
        r1 = bootstrap_metric_ci(
            binary_labels, good_scores, random_scores,
            metric_name="pr_auc", n_bootstrap=200, seed=42,
        )
        r2 = bootstrap_metric_ci(
            binary_labels, good_scores, random_scores,
            metric_name="pr_auc", n_bootstrap=200, seed=42,
        )
        assert r1.observed_diff == pytest.approx(r2.observed_diff)
        assert r1.ci_lower == pytest.approx(r2.ci_lower)
        assert r1.ci_upper == pytest.approx(r2.ci_upper)

    def test_n_bootstrap_stored(self, binary_labels, good_scores, random_scores):
        result = bootstrap_metric_ci(
            binary_labels, good_scores, random_scores,
            metric_name="pr_auc", n_bootstrap=300, seed=42,
        )
        assert result.n_bootstrap == 300


# ===================================================================
# ArmResult — convenience all-in-one evaluation
# ===================================================================

class TestArmResult:
    """Container for full per-arm evaluation results."""

    def test_has_required_fields(self):
        result = ArmResult(
            arm_name="A_majority_vote",
            pr_curve=None,  # would be PRCurveData in practice
            fpr_at_tpr=None,
            expected_cost=12.5,
            escalation=None,
        )
        assert result.arm_name == "A_majority_vote"
        assert result.expected_cost == 12.5


class TestEvaluateArm:
    """evaluate_arm(arm_name, y_true, scores, ...) → ArmResult."""

    def test_returns_arm_result(self, binary_labels, good_scores):
        result = evaluate_arm(
            arm_name="test_arm",
            y_true=binary_labels,
            scores=good_scores,
        )
        assert isinstance(result, ArmResult)

    def test_has_pr_curve(self, binary_labels, good_scores):
        result = evaluate_arm("test", binary_labels, good_scores)
        assert isinstance(result.pr_curve, PRCurveData)

    def test_has_fpr_at_tpr(self, binary_labels, good_scores):
        result = evaluate_arm("test", binary_labels, good_scores)
        assert isinstance(result.fpr_at_tpr, FPRAtTPR)

    def test_has_expected_cost(self, binary_labels, good_scores):
        result = evaluate_arm("test", binary_labels, good_scores)
        assert isinstance(result.expected_cost, float)

    def test_no_escalation_by_default(self, binary_labels, good_scores):
        result = evaluate_arm("test", binary_labels, good_scores)
        assert result.escalation is None

    def test_with_escalation_mask(self, binary_labels, good_scores):
        mask = np.array([True] * 20 + [False] * 80)
        result = evaluate_arm(
            "test", binary_labels, good_scores,
            escalation_mask=mask,
        )
        assert isinstance(result.escalation, EscalationAnalysis)

    def test_arm_name_preserved(self, binary_labels, good_scores):
        result = evaluate_arm("B_weighted_avg", binary_labels, good_scores)
        assert result.arm_name == "B_weighted_avg"
