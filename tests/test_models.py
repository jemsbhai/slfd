"""Tests for multi-model fraud detection training pipeline.

Trains 4 independent fraud detection models to serve as signal sources
for the fusion experiments (E-FD2):
    1. XGBoost (gradient boosting)
    2. Random Forest
    3. MLP (neural network)
    4. Isolation Forest (anomaly detection)

Each model must produce:
    - Scalar fraud probability ∈ [0, 1]
    - Uncertainty estimate (for Opinion construction)

The ModelSuite wraps all 4 models and provides a unified interface
for per-transaction multi-source predictions.
"""

import numpy as np
import pytest

from slfd.opinion import Opinion
from slfd.models.ensemble import (
    FraudModel,
    ModelSuite,
    train_model_suite,
    PredictionSet,
)


# ===================================================================
# Fixture: small synthetic dataset for fast tests
# ===================================================================

@pytest.fixture
def mock_dataset():
    """Small synthetic dataset mimicking IEEE-CIS structure."""
    rng = np.random.default_rng(42)
    n_train, n_test, n_features = 800, 200, 20
    fraud_rate = 0.05

    X_train = rng.normal(0, 1, size=(n_train, n_features)).astype(np.float32)
    y_train = rng.binomial(1, fraud_rate, size=n_train).astype(np.int32)
    # Make fraud clearly separable for sanity checks
    X_train[y_train == 1] += 1.5

    X_test = rng.normal(0, 1, size=(n_test, n_features)).astype(np.float32)
    y_test = rng.binomial(1, fraud_rate, size=n_test).astype(np.int32)
    X_test[y_test == 1] += 1.5

    return X_train, X_test, y_train, y_test


# ===================================================================
# 1. Individual model contract
# ===================================================================

class TestFraudModelContract:
    """Every FraudModel must satisfy a common interface."""

    def test_train_model_suite_returns_suite(self, mock_dataset):
        X_train, X_test, y_train, y_test = mock_dataset
        suite = train_model_suite(X_train, y_train, seed=42)
        assert isinstance(suite, ModelSuite)

    def test_suite_has_four_models(self, mock_dataset):
        X_train, X_test, y_train, y_test = mock_dataset
        suite = train_model_suite(X_train, y_train, seed=42)
        assert len(suite.models) == 4

    def test_each_model_is_fraud_model(self, mock_dataset):
        X_train, X_test, y_train, y_test = mock_dataset
        suite = train_model_suite(X_train, y_train, seed=42)
        for model in suite.models:
            assert isinstance(model, FraudModel)

    def test_each_model_has_name(self, mock_dataset):
        X_train, X_test, y_train, y_test = mock_dataset
        suite = train_model_suite(X_train, y_train, seed=42)
        names = [m.name for m in suite.models]
        assert "xgboost" in names
        assert "random_forest" in names
        assert "mlp" in names
        assert "isolation_forest" in names

    def test_model_names_unique(self, mock_dataset):
        X_train, X_test, y_train, y_test = mock_dataset
        suite = train_model_suite(X_train, y_train, seed=42)
        names = [m.name for m in suite.models]
        assert len(names) == len(set(names))


# ===================================================================
# 2. Predictions
# ===================================================================

class TestPredictions:
    """Models produce valid scalar probabilities."""

    def test_predict_proba_shape(self, mock_dataset):
        X_train, X_test, y_train, y_test = mock_dataset
        suite = train_model_suite(X_train, y_train, seed=42)
        for model in suite.models:
            probs = model.predict_proba(X_test)
            assert probs.shape == (len(X_test),)

    def test_predict_proba_in_unit_interval(self, mock_dataset):
        X_train, X_test, y_train, y_test = mock_dataset
        suite = train_model_suite(X_train, y_train, seed=42)
        for model in suite.models:
            probs = model.predict_proba(X_test)
            assert np.all(probs >= 0.0), f"{model.name} produced negative probabilities"
            assert np.all(probs <= 1.0), f"{model.name} produced probabilities > 1"

    def test_predict_uncertainty_shape(self, mock_dataset):
        X_train, X_test, y_train, y_test = mock_dataset
        suite = train_model_suite(X_train, y_train, seed=42)
        for model in suite.models:
            uncert = model.predict_uncertainty(X_test)
            assert uncert.shape == (len(X_test),)

    def test_predict_uncertainty_in_unit_interval(self, mock_dataset):
        X_train, X_test, y_train, y_test = mock_dataset
        suite = train_model_suite(X_train, y_train, seed=42)
        for model in suite.models:
            uncert = model.predict_uncertainty(X_test)
            assert np.all(uncert >= 0.0), f"{model.name} produced negative uncertainty"
            assert np.all(uncert <= 1.0), f"{model.name} produced uncertainty > 1"


# ===================================================================
# 3. Suite-level predictions (multi-source)
# ===================================================================

class TestModelSuitePredictions:
    """Suite produces per-transaction, per-source predictions."""

    def test_predict_all_returns_prediction_set(self, mock_dataset):
        X_train, X_test, y_train, y_test = mock_dataset
        suite = train_model_suite(X_train, y_train, seed=42)
        preds = suite.predict_all(X_test)
        assert isinstance(preds, PredictionSet)

    def test_prediction_set_probabilities_shape(self, mock_dataset):
        X_train, X_test, y_train, y_test = mock_dataset
        suite = train_model_suite(X_train, y_train, seed=42)
        preds = suite.predict_all(X_test)
        # (n_transactions, n_models)
        assert preds.probabilities.shape == (len(X_test), 4)

    def test_prediction_set_uncertainties_shape(self, mock_dataset):
        X_train, X_test, y_train, y_test = mock_dataset
        suite = train_model_suite(X_train, y_train, seed=42)
        preds = suite.predict_all(X_test)
        assert preds.uncertainties.shape == (len(X_test), 4)

    def test_prediction_set_has_model_names(self, mock_dataset):
        X_train, X_test, y_train, y_test = mock_dataset
        suite = train_model_suite(X_train, y_train, seed=42)
        preds = suite.predict_all(X_test)
        assert len(preds.model_names) == 4

    def test_prediction_set_to_opinions(self, mock_dataset):
        """Convert predictions to per-source Opinion objects."""
        X_train, X_test, y_train, y_test = mock_dataset
        suite = train_model_suite(X_train, y_train, seed=42)
        preds = suite.predict_all(X_test)
        opinions = preds.to_opinions(base_rate=0.035)
        assert len(opinions) == len(X_test)
        assert len(opinions[0]) == 4
        assert isinstance(opinions[0][0], Opinion)

    def test_opinions_are_valid(self, mock_dataset):
        X_train, X_test, y_train, y_test = mock_dataset
        suite = train_model_suite(X_train, y_train, seed=42)
        preds = suite.predict_all(X_test)
        opinions = preds.to_opinions(base_rate=0.035)
        for txn_opinions in opinions[:50]:  # spot-check first 50
            for o in txn_opinions:
                assert abs(o.b + o.d + o.u - 1.0) < 1e-6
                assert 0.0 <= o.b <= 1.0
                assert 0.0 <= o.d <= 1.0
                assert 0.0 <= o.u <= 1.0

    def test_scalar_average(self, mock_dataset):
        """Suite provides scalar average scores for baselines."""
        X_train, X_test, y_train, y_test = mock_dataset
        suite = train_model_suite(X_train, y_train, seed=42)
        preds = suite.predict_all(X_test)
        avg = preds.scalar_average()
        assert avg.shape == (len(X_test),)
        assert np.all(avg >= 0.0)
        assert np.all(avg <= 1.0)


# ===================================================================
# 4. Model quality (sanity checks, not benchmarks)
# ===================================================================

class TestModelQuality:
    """Basic sanity: models should be better than random on separable data."""

    def test_models_beat_random(self, mock_dataset):
        """Each model's AUC should exceed 0.5 on separable data."""
        from sklearn.metrics import roc_auc_score
        X_train, X_test, y_train, y_test = mock_dataset
        suite = train_model_suite(X_train, y_train, seed=42)
        for model in suite.models:
            probs = model.predict_proba(X_test)
            if len(np.unique(y_test)) < 2:
                pytest.skip("Test set has only one class")
            auc = roc_auc_score(y_test, probs)
            assert auc > 0.5, f"{model.name} AUC {auc:.3f} <= 0.5"


# ===================================================================
# 5. Reproducibility
# ===================================================================

class TestReproducibility:
    """Same seed → same predictions."""

    def test_same_seed_same_predictions(self, mock_dataset):
        X_train, X_test, y_train, y_test = mock_dataset
        suite1 = train_model_suite(X_train, y_train, seed=42)
        suite2 = train_model_suite(X_train, y_train, seed=42)
        preds1 = suite1.predict_all(X_test)
        preds2 = suite2.predict_all(X_test)
        np.testing.assert_array_almost_equal(
            preds1.probabilities, preds2.probabilities, decimal=5,
        )
