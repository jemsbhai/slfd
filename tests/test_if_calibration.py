"""Tests for Isolation Forest Platt-scaling calibration fix.

The original IF mapping (hardcoded sigmoid with scale=5.0) produced
near-constant output (~0.877 for all transactions). This happened
because the sigmoid parameters didn't adapt to the actual anomaly
score distribution.

Fix: Platt scaling — fit logistic regression on anomaly scores vs
training labels. This is standard ML calibration practice.

These tests verify the calibrated IF produces well-distributed,
discriminative probabilities while preserving all FraudModel contracts.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from slfd.models.ensemble import IsolationForestFraudModel, train_model_suite


# ===================================================================
# Fixture: separable dataset
# ===================================================================

@pytest.fixture
def separable_dataset():
    """Dataset where fraud is clearly anomalous (IF should detect it)."""
    rng = np.random.default_rng(42)
    n_train, n_test = 2000, 500
    n_features = 20
    fraud_rate = 0.05

    X_train = rng.normal(0, 1, (n_train, n_features)).astype(np.float32)
    y_train = rng.binomial(1, fraud_rate, n_train).astype(np.int32)
    # Fraud samples are outliers — shifted from normal cluster
    X_train[y_train == 1] += 2.5

    X_test = rng.normal(0, 1, (n_test, n_features)).astype(np.float32)
    y_test = rng.binomial(1, fraud_rate, n_test).astype(np.int32)
    X_test[y_test == 1] += 2.5

    return X_train, X_test, y_train, y_test


# ===================================================================
# Tests: Platt-calibrated IF produces proper probabilities
# ===================================================================

class TestIFPlattCalibration:
    """Verify calibrated IF fixes the near-constant output problem."""

    def test_probabilities_in_unit_interval(self, separable_dataset) -> None:
        X_train, X_test, y_train, y_test = separable_dataset
        model = IsolationForestFraudModel(seed=42)
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)

    def test_probabilities_have_reasonable_spread(self, separable_dataset) -> None:
        """Std should be larger than the old hardcoded sigmoid's ~0.033.

        On the full IEEE-CIS dataset (118K), the old sigmoid produced
        std=0.033. On this smaller synthetic set, Platt scaling should
        still produce a wider distribution.
        """
        X_train, X_test, y_train, y_test = separable_dataset
        model = IsolationForestFraudModel(seed=42)
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)
        assert probs.std() > 0.035, (
            f"IF probabilities still too narrow: std={probs.std():.4f}"
        )

    def test_fraud_gets_higher_probability_than_legit(self, separable_dataset) -> None:
        """On separable data, fraud mean prob should exceed legit mean prob."""
        X_train, X_test, y_train, y_test = separable_dataset
        model = IsolationForestFraudModel(seed=42)
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)
        fraud_mask = y_test == 1
        if fraud_mask.sum() == 0:
            pytest.skip("No fraud in test set")
        mean_fraud = probs[fraud_mask].mean()
        mean_legit = probs[~fraud_mask].mean()
        assert mean_fraud > mean_legit, (
            f"Fraud mean ({mean_fraud:.4f}) should exceed legit mean ({mean_legit:.4f})"
        )

    def test_auc_above_chance(self, separable_dataset) -> None:
        """Calibrated IF should have ROC-AUC well above 0.5."""
        X_train, X_test, y_train, y_test = separable_dataset
        model = IsolationForestFraudModel(seed=42)
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)
        auc = roc_auc_score(y_test, probs)
        assert auc > 0.6, f"Calibrated IF AUC should be > 0.6, got {auc:.4f}"

    def test_uncertainty_varies(self, separable_dataset) -> None:
        """Calibrated probs should produce varying uncertainty."""
        X_train, X_test, y_train, y_test = separable_dataset
        model = IsolationForestFraudModel(seed=42)
        model.fit(X_train, y_train)
        uncert = model.predict_uncertainty(X_test)
        assert uncert.std() > 0.01, (
            f"IF uncertainty too uniform: std={uncert.std():.4f}"
        )

    def test_reproducible(self, separable_dataset) -> None:
        """Same seed produces identical results."""
        X_train, X_test, y_train, y_test = separable_dataset
        model1 = IsolationForestFraudModel(seed=42)
        model1.fit(X_train, y_train)
        model2 = IsolationForestFraudModel(seed=42)
        model2.fit(X_train, y_train)
        np.testing.assert_array_almost_equal(
            model1.predict_proba(X_test),
            model2.predict_proba(X_test),
            decimal=5,
        )

    def test_fit_uses_labels_for_calibration(self, separable_dataset) -> None:
        """Fitting with vs without labels should produce different probabilities.

        This verifies Platt scaling is actually using the labels.
        """
        X_train, X_test, y_train, y_test = separable_dataset

        # With real labels
        model_real = IsolationForestFraudModel(seed=42)
        model_real.fit(X_train, y_train)
        probs_real = model_real.predict_proba(X_test)

        # With shuffled labels (breaks label-score correlation)
        rng = np.random.default_rng(99)
        y_shuffled = y_train.copy()
        rng.shuffle(y_shuffled)
        model_shuffled = IsolationForestFraudModel(seed=42)
        model_shuffled.fit(X_train, y_shuffled)
        probs_shuffled = model_shuffled.predict_proba(X_test)

        # Should not be identical
        assert not np.allclose(probs_real, probs_shuffled, atol=0.01)


class TestIFBackwardCompatibility:
    """Ensure calibrated IF still satisfies all FraudModel contracts."""

    def test_full_suite_still_works(self, separable_dataset) -> None:
        """train_model_suite should work without changes."""
        X_train, X_test, y_train, y_test = separable_dataset
        suite = train_model_suite(X_train, y_train, seed=42)
        assert len(suite.models) == 4
        preds = suite.predict_all(X_test)
        assert preds.probabilities.shape == (len(X_test), 4)

    def test_all_models_beat_random(self, separable_dataset) -> None:
        """All models including calibrated IF should beat AUC 0.5."""
        X_train, X_test, y_train, y_test = separable_dataset
        suite = train_model_suite(X_train, y_train, seed=42)
        for model in suite.models:
            probs = model.predict_proba(X_test)
            auc = roc_auc_score(y_test, probs)
            assert auc > 0.5, f"{model.name} AUC {auc:.3f} <= 0.5"
