"""Tests for trust-weighted fusion strategies.

These strategies apply trust discount to per-model opinions before
fusion, weighting sources by their validated performance. This directly
addresses the Isolation Forest miscalibration problem discovered in
the E-FD2 diagnostics: IF has ROC-AUC=0.754, so its trust should be
much lower than XGBoost (0.919).

Trust weight formula (normalized ROC-AUC, default):
    trust = clamp((roc_auc - 0.5) / 0.5, 0, 1)

This maps chance-level (AUC=0.5) to zero trust and perfect (AUC=1.0)
to full trust. Principled: a random classifier deserves no trust.
"""

from __future__ import annotations

import numpy as np
import pytest

from slfd.models.ensemble import PredictionSet
from slfd.decision import ThreeWayDecider, Decision
from slfd.strategies import (
    compute_trust_from_validation,
    sl_trust_cumulative_scores,
    sl_trust_three_way,
    ThreeWayFusionResult,
)


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def pred_set_4models() -> PredictionSet:
    """4-model prediction set: 3 good models + 1 bad (like Isolation Forest)."""
    rng = np.random.default_rng(99)
    n = 200

    probs = np.zeros((n, 4))
    # Models 0-2: reasonable classifiers (low prob for most, high for some)
    probs[:, 0] = rng.beta(1, 10, n)
    probs[:, 1] = rng.beta(1, 8, n)
    probs[:, 2] = rng.beta(1, 12, n)
    # Model 3: near-constant high output (broken, like IF)
    probs[:, 3] = 0.85 + rng.normal(0, 0.02, n)
    probs = np.clip(probs, 0.001, 0.999)

    # Calibration-based uncertainty
    distance = 2.0 * np.abs(probs - 0.5)
    confidence = np.power(distance, 1.0 / 5.0)
    uncerts = np.clip(1.0 - confidence, 0.01, 0.99)

    return PredictionSet(
        probabilities=probs,
        uncertainties=uncerts,
        model_names=["xgboost", "random_forest", "mlp", "isolation_forest"],
    )


@pytest.fixture
def val_labels() -> np.ndarray:
    """Validation labels with ~5% fraud rate."""
    rng = np.random.default_rng(77)
    labels = np.zeros(500)
    fraud_idx = rng.choice(500, size=25, replace=False)
    labels[fraud_idx] = 1
    return labels


@pytest.fixture
def val_pred_set(val_labels: np.ndarray) -> PredictionSet:
    """Validation predictions correlated with labels.

    Models 0-2 are genuinely discriminative (fraud → higher prob).
    Model 3 is near-constant high output (broken, like IF).
    """
    rng = np.random.default_rng(77)
    n = len(val_labels)
    fraud_mask = val_labels == 1

    probs = np.zeros((n, 4))
    # Model 0: good — fraud gets high prob, legit gets low
    probs[~fraud_mask, 0] = rng.beta(1, 10, int((~fraud_mask).sum()))
    probs[fraud_mask, 0] = rng.beta(6, 2, int(fraud_mask.sum()))
    # Model 1: moderate
    probs[~fraud_mask, 1] = rng.beta(1, 8, int((~fraud_mask).sum()))
    probs[fraud_mask, 1] = rng.beta(4, 2, int(fraud_mask.sum()))
    # Model 2: decent
    probs[~fraud_mask, 2] = rng.beta(1, 12, int((~fraud_mask).sum()))
    probs[fraud_mask, 2] = rng.beta(5, 3, int(fraud_mask.sum()))
    # Model 3: broken — near-constant regardless of label
    probs[:, 3] = 0.85 + rng.normal(0, 0.02, n)
    probs = np.clip(probs, 0.001, 0.999)

    distance = 2.0 * np.abs(probs - 0.5)
    confidence = np.power(distance, 1.0 / 5.0)
    uncerts = np.clip(1.0 - confidence, 0.01, 0.99)

    return PredictionSet(
        probabilities=probs,
        uncertainties=uncerts,
        model_names=["xgboost", "random_forest", "mlp", "isolation_forest"],
    )


@pytest.fixture
def default_decider() -> ThreeWayDecider:
    return ThreeWayDecider(
        block_threshold=0.6,
        approve_threshold=0.6,
        escalate_uncertainty=0.4,
        escalate_conflict=0.3,
    )


# ===================================================================
# Tests: compute_trust_from_validation
# ===================================================================

class TestComputeTrustFromValidation:
    """Test trust weight computation from validation performance."""

    def test_returns_array_of_correct_shape(
        self, val_pred_set: PredictionSet, val_labels: np.ndarray
    ) -> None:
        trust = compute_trust_from_validation(val_pred_set, val_labels)
        assert isinstance(trust, np.ndarray)
        assert trust.shape == (4,)

    def test_values_in_zero_one(
        self, val_pred_set: PredictionSet, val_labels: np.ndarray
    ) -> None:
        trust = compute_trust_from_validation(val_pred_set, val_labels)
        assert np.all(trust >= 0.0)
        assert np.all(trust <= 1.0)

    def test_better_model_gets_higher_trust(
        self, val_pred_set: PredictionSet, val_labels: np.ndarray
    ) -> None:
        """Good classifiers should get higher trust than the broken IF."""
        trust = compute_trust_from_validation(val_pred_set, val_labels)
        # Model 3 (IF-like, near-constant) should have lowest trust
        # At minimum, models 0-2 should all have higher trust than model 3
        assert trust[0] > trust[3], "XGBoost should have more trust than IF"
        assert trust[1] > trust[3], "RF should have more trust than IF"
        assert trust[2] > trust[3], "MLP should have more trust than IF"

    def test_random_model_gets_near_zero_trust(self) -> None:
        """A model with AUC ≈ 0.5 should get near-zero trust."""
        rng = np.random.default_rng(42)
        n = 1000
        # Single model, random predictions
        probs = rng.uniform(0, 1, (n, 1))
        uncerts = np.full((n, 1), 0.5)
        labels = rng.integers(0, 2, n)

        pred_set = PredictionSet(
            probabilities=probs, uncertainties=uncerts, model_names=["random"]
        )
        trust = compute_trust_from_validation(pred_set, labels)
        # AUC should be ~0.5, so normalized trust ≈ 0
        assert trust[0] < 0.15, f"Random model trust should be near 0, got {trust[0]}"

    def test_perfect_model_gets_full_trust(self) -> None:
        """A model with AUC = 1.0 should get trust = 1.0."""
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        probs = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9]).reshape(-1, 1)
        uncerts = np.full((10, 1), 0.1)

        pred_set = PredictionSet(
            probabilities=probs, uncertainties=uncerts, model_names=["perfect"]
        )
        trust = compute_trust_from_validation(pred_set, labels)
        assert trust[0] == pytest.approx(1.0)

    def test_metric_parameter_roc_auc(
        self, val_pred_set: PredictionSet, val_labels: np.ndarray
    ) -> None:
        trust = compute_trust_from_validation(val_pred_set, val_labels, metric="roc_auc")
        assert trust.shape == (4,)
        assert np.all(trust >= 0.0)

    def test_metric_parameter_pr_auc(
        self, val_pred_set: PredictionSet, val_labels: np.ndarray
    ) -> None:
        trust = compute_trust_from_validation(val_pred_set, val_labels, metric="pr_auc")
        assert trust.shape == (4,)
        assert np.all(trust >= 0.0)

    def test_invalid_metric_rejected(
        self, val_pred_set: PredictionSet, val_labels: np.ndarray
    ) -> None:
        with pytest.raises(ValueError, match="metric"):
            compute_trust_from_validation(val_pred_set, val_labels, metric="f1")

    def test_label_length_mismatch_rejected(
        self, val_pred_set: PredictionSet
    ) -> None:
        bad_labels = np.zeros(10)  # wrong length
        with pytest.raises(ValueError, match="length"):
            compute_trust_from_validation(val_pred_set, bad_labels)


# ===================================================================
# Tests: sl_trust_cumulative_scores
# ===================================================================

class TestSLTrustCumulativeScores:
    """Test trust-discounted cumulative fusion."""

    def test_returns_array_of_correct_shape(
        self, pred_set_4models: PredictionSet
    ) -> None:
        trust = np.array([0.9, 0.8, 0.7, 0.1])
        scores = sl_trust_cumulative_scores(pred_set_4models, trust, base_rate=0.035)
        assert isinstance(scores, np.ndarray)
        assert scores.shape == (200,)

    def test_scores_in_zero_one(
        self, pred_set_4models: PredictionSet
    ) -> None:
        trust = np.array([0.9, 0.8, 0.7, 0.1])
        scores = sl_trust_cumulative_scores(pred_set_4models, trust, base_rate=0.035)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_low_trust_on_bad_model_changes_scores(
        self, pred_set_4models: PredictionSet
    ) -> None:
        """Discounting the bad model should change fused scores vs uniform trust."""
        uniform_trust = np.array([1.0, 1.0, 1.0, 1.0])
        discounted_trust = np.array([1.0, 1.0, 1.0, 0.1])

        scores_uniform = sl_trust_cumulative_scores(
            pred_set_4models, uniform_trust, base_rate=0.035
        )
        scores_discounted = sl_trust_cumulative_scores(
            pred_set_4models, discounted_trust, base_rate=0.035
        )

        # Scores should differ
        assert not np.allclose(scores_uniform, scores_discounted)

        # Discounting the always-fraud model should lower the mean score
        assert np.mean(scores_discounted) < np.mean(scores_uniform)

    def test_full_trust_matches_undiscounted(
        self, pred_set_4models: PredictionSet
    ) -> None:
        """Trust=1.0 for all models should match sl_cumulative_scores."""
        from slfd.strategies import sl_cumulative_scores

        full_trust = np.array([1.0, 1.0, 1.0, 1.0])
        scores_trusted = sl_trust_cumulative_scores(
            pred_set_4models, full_trust, base_rate=0.035
        )
        scores_plain = sl_cumulative_scores(pred_set_4models, base_rate=0.035)
        np.testing.assert_allclose(scores_trusted, scores_plain, atol=1e-9)

    def test_wrong_trust_length_rejected(
        self, pred_set_4models: PredictionSet
    ) -> None:
        trust = np.array([0.9, 0.8])  # wrong length
        with pytest.raises(ValueError, match="trust"):
            sl_trust_cumulative_scores(pred_set_4models, trust, base_rate=0.035)


# ===================================================================
# Tests: sl_trust_three_way
# ===================================================================

class TestSLTrustThreeWay:
    """Test trust-discounted three-way decision fusion."""

    def test_returns_correct_type(
        self, pred_set_4models: PredictionSet, default_decider: ThreeWayDecider
    ) -> None:
        trust = np.array([0.9, 0.8, 0.7, 0.1])
        result = sl_trust_three_way(
            pred_set_4models, trust, base_rate=0.035, decider=default_decider
        )
        assert isinstance(result, ThreeWayFusionResult)

    def test_scores_shape(
        self, pred_set_4models: PredictionSet, default_decider: ThreeWayDecider
    ) -> None:
        trust = np.array([0.9, 0.8, 0.7, 0.1])
        result = sl_trust_three_way(
            pred_set_4models, trust, base_rate=0.035, decider=default_decider
        )
        assert result.scores.shape == (200,)
        assert result.decisions.shape == (200,)
        assert result.conflicts.shape == (200,)
        assert result.escalation_mask.shape == (200,)

    def test_discounting_bad_model_reduces_escalation(
        self, pred_set_4models: PredictionSet, default_decider: ThreeWayDecider
    ) -> None:
        """Discounting the broken model should reduce conflict → less escalation."""
        # With uniform trust: the bad model creates high conflict
        uniform = sl_trust_three_way(
            pred_set_4models,
            np.array([1.0, 1.0, 1.0, 1.0]),
            base_rate=0.035,
            decider=default_decider,
        )
        # With discounted trust: bad model pushed toward vacuous, less conflict
        discounted = sl_trust_three_way(
            pred_set_4models,
            np.array([1.0, 1.0, 1.0, 0.1]),
            base_rate=0.035,
            decider=default_decider,
        )
        assert discounted.escalation_rate <= uniform.escalation_rate

    def test_conflict_reduced_by_discounting(
        self, pred_set_4models: PredictionSet, default_decider: ThreeWayDecider
    ) -> None:
        """Trust discounting a disagreeing source should reduce mean conflict."""
        uniform = sl_trust_three_way(
            pred_set_4models,
            np.array([1.0, 1.0, 1.0, 1.0]),
            base_rate=0.035,
            decider=default_decider,
        )
        discounted = sl_trust_three_way(
            pred_set_4models,
            np.array([1.0, 1.0, 1.0, 0.1]),
            base_rate=0.035,
            decider=default_decider,
        )
        assert np.mean(discounted.conflicts) < np.mean(uniform.conflicts)
