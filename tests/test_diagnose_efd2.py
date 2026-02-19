"""Tests for E-FD2 diagnostic analysis functions.

These tests verify the diagnostic tools work correctly on synthetic data
with known properties, so we can trust the results on real data.
"""

from __future__ import annotations

import numpy as np
import pytest

from slfd.models.ensemble import PredictionSet
from slfd.decision import ThreeWayDecider
from slfd.experiments.diagnose_efd2 import (
    OpinionDistStats,
    FusedOpinionStats,
    DeciderDiagnostics,
    ModelPerformanceStats,
    EFD2Diagnostics,
    diagnose_opinion_distributions,
    diagnose_fused_opinions,
    diagnose_decider,
    diagnose_model_performance,
    run_full_diagnostics,
)


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def simple_pred_set() -> PredictionSet:
    """Prediction set with 100 transactions, 4 models, known properties."""
    rng = np.random.default_rng(42)
    n = 100
    n_models = 4

    # Model 0: good classifier (high prob for fraud, low for legit)
    # Model 1: moderate classifier
    # Model 2: weak classifier (near random)
    # Model 3: anomaly detector (different distribution)
    probs = np.zeros((n, n_models))
    probs[:, 0] = rng.beta(1, 10, n)  # Mostly low (legit)
    probs[:, 1] = rng.beta(2, 8, n)
    probs[:, 2] = rng.beta(3, 4, n)   # Closer to uniform
    probs[:, 3] = rng.beta(1, 5, n)

    # Uncertainties derived from calibration (same as ensemble.py)
    distance = 2.0 * np.abs(probs - 0.5)
    confidence = np.power(distance, 1.0 / 5.0)
    uncerts = np.clip(1.0 - confidence, 0.01, 0.99)

    return PredictionSet(
        probabilities=probs,
        uncertainties=uncerts,
        model_names=["xgboost", "random_forest", "mlp", "isolation_forest"],
    )


@pytest.fixture
def simple_labels() -> np.ndarray:
    """Labels with ~10% fraud rate."""
    rng = np.random.default_rng(42)
    labels = np.zeros(100)
    fraud_idx = rng.choice(100, size=10, replace=False)
    labels[fraud_idx] = 1
    return labels


@pytest.fixture
def default_decider() -> ThreeWayDecider:
    """The same decider used in E-FD2 run 1."""
    return ThreeWayDecider(
        block_threshold=0.6,
        approve_threshold=0.6,
        escalate_uncertainty=0.4,
        escalate_conflict=0.3,
    )


# ===================================================================
# Tests: diagnose_opinion_distributions
# ===================================================================

class TestDiagnoseOpinionDistributions:
    """Test per-model opinion distribution analysis."""

    def test_returns_correct_type(self, simple_pred_set: PredictionSet) -> None:
        result = diagnose_opinion_distributions(simple_pred_set, base_rate=0.035)
        assert isinstance(result, list)
        assert all(isinstance(s, OpinionDistStats) for s in result)

    def test_one_entry_per_model(self, simple_pred_set: PredictionSet) -> None:
        result = diagnose_opinion_distributions(simple_pred_set, base_rate=0.035)
        assert len(result) == 4

    def test_model_names_preserved(self, simple_pred_set: PredictionSet) -> None:
        result = diagnose_opinion_distributions(simple_pred_set, base_rate=0.035)
        names = [s.model_name for s in result]
        assert names == ["xgboost", "random_forest", "mlp", "isolation_forest"]

    def test_stats_are_valid(self, simple_pred_set: PredictionSet) -> None:
        result = diagnose_opinion_distributions(simple_pred_set, base_rate=0.035)
        for s in result:
            # Means should be in [0, 1]
            assert 0.0 <= s.b_mean <= 1.0
            assert 0.0 <= s.d_mean <= 1.0
            assert 0.0 <= s.u_mean <= 1.0
            # Stds should be non-negative
            assert s.b_std >= 0.0
            assert s.d_std >= 0.0
            assert s.u_std >= 0.0
            # Percentiles should be monotonic
            assert s.u_p25 <= s.u_p50 <= s.u_p75

    def test_n_transactions_matches(self, simple_pred_set: PredictionSet) -> None:
        result = diagnose_opinion_distributions(simple_pred_set, base_rate=0.035)
        assert result[0].n_transactions == 100


# ===================================================================
# Tests: diagnose_fused_opinions
# ===================================================================

class TestDiagnoseFusedOpinions:
    """Test fused opinion distribution analysis."""

    def test_returns_correct_type(self, simple_pred_set: PredictionSet) -> None:
        result = diagnose_fused_opinions(simple_pred_set, base_rate=0.035)
        assert isinstance(result, FusedOpinionStats)

    def test_stats_are_valid(self, simple_pred_set: PredictionSet) -> None:
        result = diagnose_fused_opinions(simple_pred_set, base_rate=0.035)
        assert 0.0 <= result.b_mean <= 1.0
        assert 0.0 <= result.d_mean <= 1.0
        assert 0.0 <= result.u_mean <= 1.0
        assert result.n_transactions == 100

    def test_fused_uncertainty_lower_than_individual(
        self, simple_pred_set: PredictionSet
    ) -> None:
        """Cumulative fusion of 4 sources should reduce uncertainty vs individual."""
        indiv = diagnose_opinion_distributions(simple_pred_set, base_rate=0.035)
        fused = diagnose_fused_opinions(simple_pred_set, base_rate=0.035)
        avg_individual_u = np.mean([s.u_mean for s in indiv])
        # Fused uncertainty should generally be lower than average individual
        assert fused.u_mean < avg_individual_u

    def test_conflict_stats_present(self, simple_pred_set: PredictionSet) -> None:
        result = diagnose_fused_opinions(simple_pred_set, base_rate=0.035)
        assert result.conflict_mean >= 0.0
        assert result.conflict_std >= 0.0

    def test_expected_prob_stats_present(self, simple_pred_set: PredictionSet) -> None:
        result = diagnose_fused_opinions(simple_pred_set, base_rate=0.035)
        assert 0.0 <= result.expected_prob_mean <= 1.0


# ===================================================================
# Tests: diagnose_decider
# ===================================================================

class TestDiagnoseDecider:
    """Test decider threshold diagnosis."""

    def test_returns_correct_type(
        self, simple_pred_set: PredictionSet, default_decider: ThreeWayDecider
    ) -> None:
        result = diagnose_decider(simple_pred_set, base_rate=0.035, decider=default_decider)
        assert isinstance(result, DeciderDiagnostics)

    def test_decision_fractions_sum_to_one(
        self, simple_pred_set: PredictionSet, default_decider: ThreeWayDecider
    ) -> None:
        result = diagnose_decider(simple_pred_set, base_rate=0.035, decider=default_decider)
        total = result.frac_block + result.frac_approve + result.frac_escalate
        assert abs(total - 1.0) < 1e-9

    def test_escalation_reasons_present(
        self, simple_pred_set: PredictionSet, default_decider: ThreeWayDecider
    ) -> None:
        result = diagnose_decider(simple_pred_set, base_rate=0.035, decider=default_decider)
        # Should have breakdown of WHY escalation happens
        assert hasattr(result, "escalate_by_conflict")
        assert hasattr(result, "escalate_by_uncertainty")
        assert hasattr(result, "escalate_by_default")

    def test_threshold_comparison_present(
        self, simple_pred_set: PredictionSet, default_decider: ThreeWayDecider
    ) -> None:
        result = diagnose_decider(simple_pred_set, base_rate=0.035, decider=default_decider)
        # What fraction of opinions exceed each threshold?
        assert hasattr(result, "frac_b_above_block")
        assert hasattr(result, "frac_d_above_approve")
        assert hasattr(result, "frac_u_above_escalate")


# ===================================================================
# Tests: diagnose_model_performance
# ===================================================================

class TestDiagnoseModelPerformance:
    """Test per-model individual performance analysis."""

    def test_returns_correct_type(
        self, simple_pred_set: PredictionSet, simple_labels: np.ndarray
    ) -> None:
        result = diagnose_model_performance(simple_pred_set, simple_labels)
        assert isinstance(result, list)
        assert all(isinstance(s, ModelPerformanceStats) for s in result)

    def test_one_entry_per_model(
        self, simple_pred_set: PredictionSet, simple_labels: np.ndarray
    ) -> None:
        result = diagnose_model_performance(simple_pred_set, simple_labels)
        assert len(result) == 4

    def test_metrics_are_valid(
        self, simple_pred_set: PredictionSet, simple_labels: np.ndarray
    ) -> None:
        result = diagnose_model_performance(simple_pred_set, simple_labels)
        for s in result:
            assert 0.0 <= s.accuracy <= 1.0
            assert 0.0 <= s.roc_auc <= 1.0
            assert 0.0 <= s.pr_auc <= 1.0

    def test_model_names_preserved(
        self, simple_pred_set: PredictionSet, simple_labels: np.ndarray
    ) -> None:
        result = diagnose_model_performance(simple_pred_set, simple_labels)
        names = [s.model_name for s in result]
        assert names == ["xgboost", "random_forest", "mlp", "isolation_forest"]


# ===================================================================
# Tests: run_full_diagnostics
# ===================================================================

class TestRunFullDiagnostics:
    """Test the unified diagnostic runner."""

    def test_returns_correct_type(
        self,
        simple_pred_set: PredictionSet,
        simple_labels: np.ndarray,
        default_decider: ThreeWayDecider,
    ) -> None:
        result = run_full_diagnostics(
            pred_set=simple_pred_set,
            labels=simple_labels,
            base_rate=0.035,
            decider=default_decider,
        )
        assert isinstance(result, EFD2Diagnostics)

    def test_all_components_present(
        self,
        simple_pred_set: PredictionSet,
        simple_labels: np.ndarray,
        default_decider: ThreeWayDecider,
    ) -> None:
        result = run_full_diagnostics(
            pred_set=simple_pred_set,
            labels=simple_labels,
            base_rate=0.035,
            decider=default_decider,
        )
        assert len(result.opinion_distributions) == 4
        assert isinstance(result.fused_opinions, FusedOpinionStats)
        assert isinstance(result.decider_diagnostics, DeciderDiagnostics)
        assert len(result.model_performance) == 4

    def test_to_dict_is_serializable(
        self,
        simple_pred_set: PredictionSet,
        simple_labels: np.ndarray,
        default_decider: ThreeWayDecider,
    ) -> None:
        """Ensure diagnostics can be serialized to JSON-compatible dict."""
        import json

        result = run_full_diagnostics(
            pred_set=simple_pred_set,
            labels=simple_labels,
            base_rate=0.035,
            decider=default_decider,
        )
        d = result.to_dict()
        # Should not raise
        json_str = json.dumps(d)
        assert len(json_str) > 0
