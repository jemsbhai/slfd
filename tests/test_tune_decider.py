"""Tests for ThreeWayDecider threshold tuning on validation data.

Threshold tuning is hyperparameter optimization — it uses ONLY the
validation set. The test set is never touched until final evaluation.

The tuner searches over threshold combinations and selects the one
that minimizes expected cost (accounting for escalation costs).
"""

from __future__ import annotations

import numpy as np
import pytest

from slfd.models.ensemble import PredictionSet
from slfd.decision import ThreeWayDecider
from slfd.metrics import CostConfig
from slfd.experiments.tune_decider import (
    TuningResult,
    tune_decider_thresholds,
)


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def val_labels() -> np.ndarray:
    """Validation labels with ~5% fraud rate."""
    rng = np.random.default_rng(55)
    labels = np.zeros(400)
    fraud_idx = rng.choice(400, size=20, replace=False)
    labels[fraud_idx] = 1
    return labels


@pytest.fixture
def val_pred_set(val_labels: np.ndarray) -> PredictionSet:
    """Validation predictions correlated with labels, 4 models."""
    rng = np.random.default_rng(55)
    n = len(val_labels)
    fraud_mask = val_labels == 1

    probs = np.zeros((n, 4))
    probs[~fraud_mask, 0] = rng.beta(1, 10, int((~fraud_mask).sum()))
    probs[fraud_mask, 0] = rng.beta(6, 2, int(fraud_mask.sum()))
    probs[~fraud_mask, 1] = rng.beta(1, 8, int((~fraud_mask).sum()))
    probs[fraud_mask, 1] = rng.beta(4, 2, int(fraud_mask.sum()))
    probs[~fraud_mask, 2] = rng.beta(1, 12, int((~fraud_mask).sum()))
    probs[fraud_mask, 2] = rng.beta(5, 3, int(fraud_mask.sum()))
    probs[~fraud_mask, 3] = rng.beta(1, 6, int((~fraud_mask).sum()))
    probs[fraud_mask, 3] = rng.beta(3, 3, int(fraud_mask.sum()))
    probs = np.clip(probs, 0.001, 0.999)

    distance = 2.0 * np.abs(probs - 0.5)
    confidence = np.power(distance, 1.0 / 5.0)
    uncerts = np.clip(1.0 - confidence, 0.01, 0.99)

    return PredictionSet(
        probabilities=probs,
        uncertainties=uncerts,
        model_names=["m0", "m1", "m2", "m3"],
    )


@pytest.fixture
def cost_config() -> CostConfig:
    return CostConfig(
        review_cost=2.0,
        missed_fraud_cost=50.0,
        false_block_cost=0.50,
    )


# ===================================================================
# Tests: tune_decider_thresholds
# ===================================================================

class TestTuneDeciderThresholds:
    """Test validation-based threshold tuning."""

    def test_returns_tuning_result(
        self, val_pred_set: PredictionSet, val_labels: np.ndarray,
        cost_config: CostConfig,
    ) -> None:
        result = tune_decider_thresholds(
            val_pred_set, val_labels,
            base_rate=0.05, cost_config=cost_config,
        )
        assert isinstance(result, TuningResult)

    def test_best_decider_is_valid(
        self, val_pred_set: PredictionSet, val_labels: np.ndarray,
        cost_config: CostConfig,
    ) -> None:
        result = tune_decider_thresholds(
            val_pred_set, val_labels,
            base_rate=0.05, cost_config=cost_config,
        )
        d = result.best_decider
        assert isinstance(d, ThreeWayDecider)
        assert 0.0 <= d.block_threshold <= 1.0
        assert 0.0 <= d.approve_threshold <= 1.0
        assert 0.0 <= d.escalate_uncertainty <= 1.0
        assert 0.0 <= d.escalate_conflict <= 1.0

    def test_best_cost_is_finite(
        self, val_pred_set: PredictionSet, val_labels: np.ndarray,
        cost_config: CostConfig,
    ) -> None:
        result = tune_decider_thresholds(
            val_pred_set, val_labels,
            base_rate=0.05, cost_config=cost_config,
        )
        assert np.isfinite(result.best_cost)

    def test_escalation_rate_in_bounds(
        self, val_pred_set: PredictionSet, val_labels: np.ndarray,
        cost_config: CostConfig,
    ) -> None:
        result = tune_decider_thresholds(
            val_pred_set, val_labels,
            base_rate=0.05, cost_config=cost_config,
        )
        assert 0.0 <= result.best_escalation_rate <= 1.0

    def test_n_configs_searched_positive(
        self, val_pred_set: PredictionSet, val_labels: np.ndarray,
        cost_config: CostConfig,
    ) -> None:
        result = tune_decider_thresholds(
            val_pred_set, val_labels,
            base_rate=0.05, cost_config=cost_config,
        )
        assert result.n_configs_searched > 0

    def test_does_not_return_degenerate_escalation(
        self, val_pred_set: PredictionSet, val_labels: np.ndarray,
        cost_config: CostConfig,
    ) -> None:
        """Tuner should not return config that escalates >95% of transactions."""
        result = tune_decider_thresholds(
            val_pred_set, val_labels,
            base_rate=0.05, cost_config=cost_config,
        )
        assert result.best_escalation_rate < 0.95, (
            f"Tuned escalation rate too high: {result.best_escalation_rate:.1%}"
        )

    def test_with_trust_weights(
        self, val_pred_set: PredictionSet, val_labels: np.ndarray,
        cost_config: CostConfig,
    ) -> None:
        """Should accept optional trust weights for trust-discounted tuning."""
        trust = np.array([0.9, 0.8, 0.7, 0.5])
        result = tune_decider_thresholds(
            val_pred_set, val_labels,
            base_rate=0.05, cost_config=cost_config,
            trust_weights=trust,
        )
        assert isinstance(result, TuningResult)
        assert result.best_escalation_rate < 0.95

    def test_to_dict_serializable(
        self, val_pred_set: PredictionSet, val_labels: np.ndarray,
        cost_config: CostConfig,
    ) -> None:
        import json
        result = tune_decider_thresholds(
            val_pred_set, val_labels,
            base_rate=0.05, cost_config=cost_config,
        )
        d = result.to_dict()
        json_str = json.dumps(d)
        assert len(json_str) > 0
