"""Multi-model fraud detection ensemble.

Trains 4 independent fraud detection models as signal sources for
the fusion experiments:
    1. XGBoost — gradient boosting
    2. Random Forest — bagging ensemble
    3. MLP — neural network classifier
    4. Isolation Forest — unsupervised anomaly detection

Each model produces:
    - Scalar fraud probability ∈ [0, 1]
    - Uncertainty estimate (for Opinion construction)

Uncertainty estimation:
    - XGBoost / RF / MLP: calibration-based uncertainty using the
      distance from decision boundary. Scores near 0.5 → high uncertainty;
      scores near 0 or 1 → low uncertainty.
    - Isolation Forest: anomaly score mapped to probability; uncertainty
      derived from score spread.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb

    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

from slfd.opinion import Opinion


# ===================================================================
# Uncertainty estimation
# ===================================================================

def _calibration_uncertainty(probs: np.ndarray, steepness: float = 5.0) -> np.ndarray:
    """Estimate uncertainty from predicted probabilities.

    Uses a U-shaped function: scores near 0.5 have high uncertainty,
    scores near 0 or 1 have low uncertainty.

        u(p) = 1 - (2|p - 0.5|)^(1/steepness)

    Clamped to [0.01, 0.99] to avoid degenerate opinions.

    Parameters
    ----------
    probs : np.ndarray
        Predicted probabilities, shape (n,).
    steepness : float
        Controls how quickly uncertainty drops away from 0.5.
        Higher → sharper transition.

    Returns
    -------
    np.ndarray
        Uncertainty estimates ∈ [0.01, 0.99], shape (n,).
    """
    distance = 2.0 * np.abs(probs - 0.5)  # ∈ [0, 1]
    confidence = np.power(distance, 1.0 / steepness)
    uncertainty = 1.0 - confidence
    return np.clip(uncertainty, 0.01, 0.99)


# ===================================================================
# FraudModel base class
# ===================================================================

class FraudModel(ABC):
    """Abstract base for fraud detection models."""

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model."""

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict fraud probability, shape (n,), values ∈ [0, 1]."""

    def predict_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """Estimate prediction uncertainty, shape (n,), values ∈ [0, 1]."""
        probs = self.predict_proba(X)
        return _calibration_uncertainty(probs)


# ===================================================================
# XGBoost model
# ===================================================================

class XGBoostFraudModel(FraudModel):
    """XGBoost gradient boosting classifier."""

    def __init__(self, seed: int = 42) -> None:
        super().__init__(name="xgboost")
        if not _HAS_XGB:
            raise ImportError("xgboost is required. Install with: pip install xgboost")
        self._model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs = self._model.predict_proba(X)
        # Column 1 = fraud probability
        return probs[:, 1].astype(np.float64)


# ===================================================================
# Random Forest model
# ===================================================================

class RandomForestFraudModel(FraudModel):
    """Random Forest classifier."""

    def __init__(self, seed: int = 42) -> None:
        super().__init__(name="random_forest")
        self._model = RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_split=10,
            random_state=seed,
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs = self._model.predict_proba(X)
        return probs[:, 1].astype(np.float64)


# ===================================================================
# MLP model
# ===================================================================

class MLPFraudModel(FraudModel):
    """Multi-layer perceptron classifier with standardized input."""

    def __init__(self, seed: int = 42) -> None:
        super().__init__(name="mlp")
        self._scaler = StandardScaler()
        self._model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=seed,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X_scaled = self._scaler.fit_transform(X)
        self._model.fit(X_scaled, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self._scaler.transform(X)
        probs = self._model.predict_proba(X_scaled)
        return probs[:, 1].astype(np.float64)


# ===================================================================
# Isolation Forest model
# ===================================================================

class IsolationForestFraudModel(FraudModel):
    """Isolation Forest anomaly detector adapted for fraud scoring.

    Isolation Forest is unsupervised — it learns what's 'normal' and
    flags anomalies. We map its anomaly score to a fraud probability.
    """

    def __init__(self, seed: int = 42) -> None:
        super().__init__(name="isolation_forest")
        self._model = IsolationForest(
            n_estimators=100,
            contamination=0.035,  # approximate fraud rate
            random_state=seed,
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # Isolation Forest is unsupervised — ignores y
        self._model.fit(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Map anomaly scores to fraud probabilities.

        score_samples() returns negative anomaly scores where
        more negative = more anomalous. We map to [0, 1] via:
            p = 1 / (1 + exp(score * scale))
        """
        scores = self._model.score_samples(X)
        # Sigmoid mapping: more negative scores → higher fraud probability
        # Scale factor controls the spread
        probs = 1.0 / (1.0 + np.exp(scores * 5.0))
        return np.clip(probs, 0.0, 1.0).astype(np.float64)


# ===================================================================
# PredictionSet — multi-source predictions
# ===================================================================

@dataclass
class PredictionSet:
    """Per-transaction predictions from all models.

    Attributes
    ----------
    probabilities : np.ndarray
        Fraud probabilities, shape (n_transactions, n_models).
    uncertainties : np.ndarray
        Uncertainty estimates, shape (n_transactions, n_models).
    model_names : list[str]
        Names of each model (column order).
    """

    probabilities: np.ndarray
    uncertainties: np.ndarray
    model_names: list[str]

    def to_opinions(self, base_rate: float = 0.035) -> list[list[Opinion]]:
        """Convert to per-transaction, per-source Opinion objects.

        Parameters
        ----------
        base_rate : float
            Prior fraud rate for opinion construction.

        Returns
        -------
        list[list[Opinion]]
            [n_transactions][n_models] Opinion objects.
        """
        n_txns, n_models = self.probabilities.shape
        opinions: list[list[Opinion]] = []

        for i in range(n_txns):
            txn_opinions: list[Opinion] = []
            for j in range(n_models):
                o = Opinion.from_confidence(
                    probability=float(self.probabilities[i, j]),
                    uncertainty=float(self.uncertainties[i, j]),
                    base_rate=base_rate,
                )
                txn_opinions.append(o)
            opinions.append(txn_opinions)

        return opinions

    def scalar_average(self) -> np.ndarray:
        """Simple mean of all model probabilities (baseline fusion).

        Returns
        -------
        np.ndarray
            Average fraud probability, shape (n_transactions,).
        """
        return np.mean(self.probabilities, axis=1)


# ===================================================================
# ModelSuite — unified training and prediction
# ===================================================================

@dataclass
class ModelSuite:
    """Collection of trained fraud detection models.

    Attributes
    ----------
    models : list[FraudModel]
        The trained models.
    """

    models: list[FraudModel]

    def predict_all(self, X: np.ndarray) -> PredictionSet:
        """Get predictions from all models.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix, shape (n_transactions, n_features).

        Returns
        -------
        PredictionSet
        """
        n = len(X)
        n_models = len(self.models)

        probs = np.zeros((n, n_models), dtype=np.float64)
        uncerts = np.zeros((n, n_models), dtype=np.float64)
        names = []

        for j, model in enumerate(self.models):
            probs[:, j] = model.predict_proba(X)
            uncerts[:, j] = model.predict_uncertainty(X)
            names.append(model.name)

        return PredictionSet(
            probabilities=probs,
            uncertainties=uncerts,
            model_names=names,
        )


# ===================================================================
# Training entry point
# ===================================================================

def train_model_suite(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = 42,
) -> ModelSuite:
    """Train all 4 fraud detection models.

    Parameters
    ----------
    X_train : np.ndarray
        Training features, shape (n_train, n_features).
    y_train : np.ndarray
        Training labels (0/1), shape (n_train,).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    ModelSuite
        Suite of 4 trained models.
    """
    models: list[FraudModel] = [
        XGBoostFraudModel(seed=seed),
        RandomForestFraudModel(seed=seed),
        MLPFraudModel(seed=seed),
        IsolationForestFraudModel(seed=seed),
    ]

    for model in models:
        model.fit(X_train, y_train)

    return ModelSuite(models=models)
