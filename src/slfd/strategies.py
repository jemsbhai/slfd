"""E-FD2 fusion strategies — 9 treatment arms for multi-source fraud signal fusion.

Scalar baselines:
    A: majority_vote           — fraction of models predicting fraud (prob > threshold)
    B: weighted_average        — accuracy-weighted mean of probabilities
    C: StackingMetaLearner     — logistic regression on model outputs
    D: bayesian_model_average  — BMA weights from validation log-likelihoods
    E: noisy_or                — P(fraud) = 1 − ∏(1 − pᵢ)

SLFD treatments:
    F: sl_cumulative_scores    — cumulative_fuse → expected_probability
    G: sl_three_way            — F + conflict_metric → ThreeWayDecider
    H: sl_robust_three_way     — robust_fuse → ThreeWayDecider
    I: ConfidenceFeatureLearner — per-source (prob, uncertainty) as meta-features

All strategies consume a PredictionSet from the 4-model ensemble.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from slfd.opinion import Opinion
from slfd.fusion import cumulative_fuse, conflict_metric
from slfd.robust import robust_fuse
from slfd.decision import Decision, ThreeWayDecider
from slfd.models.ensemble import PredictionSet


# ===================================================================
# Result container for three-way strategies (G, H, I)
# ===================================================================

@dataclass
class ThreeWayFusionResult:
    """Result of a three-way fusion strategy.

    Attributes
    ----------
    scores : np.ndarray
        Fused fraud scores (expected probability), shape (n,).
    decisions : np.ndarray
        Per-transaction Decision enum values, shape (n,).
    conflicts : np.ndarray
        Per-transaction conflict metric, shape (n,).
    escalation_mask : np.ndarray
        Boolean mask — True where decision is ESCALATE, shape (n,).
    excluded_counts : np.ndarray or None
        Per-transaction count of excluded sources (robust only).
    """

    scores: np.ndarray
    decisions: np.ndarray
    conflicts: np.ndarray
    escalation_mask: np.ndarray
    excluded_counts: np.ndarray | None = None

    @property
    def escalation_rate(self) -> float:
        """Fraction of transactions routed to escalation."""
        return float(np.mean(self.escalation_mask))

    @property
    def auto_decided_mask(self) -> np.ndarray:
        """Boolean mask — True where transaction was auto-decided (not escalated)."""
        return ~self.escalation_mask


# ===================================================================
# A: Majority Vote
# ===================================================================

def majority_vote(
    pred_set: PredictionSet,
    threshold: float = 0.5,
) -> np.ndarray:
    """Fraction of models predicting fraud (probability > threshold).

    Parameters
    ----------
    pred_set : PredictionSet
        Per-model predictions.
    threshold : float
        Probability threshold for counting a model as "predicting fraud".

    Returns
    -------
    np.ndarray
        Vote fraction per transaction, shape (n,), values in [0, 1].
    """
    votes = (pred_set.probabilities > threshold).astype(np.float64)
    return np.mean(votes, axis=1)


# ===================================================================
# B: Weighted Average
# ===================================================================

def weighted_average(
    pred_set: PredictionSet,
    weights: np.ndarray,
) -> np.ndarray:
    """Accuracy-weighted mean of model probabilities.

    Parameters
    ----------
    pred_set : PredictionSet
        Per-model predictions.
    weights : np.ndarray
        Non-negative weights, shape (n_models,). Normalized internally.

    Returns
    -------
    np.ndarray
        Weighted average fraud score, shape (n,), values in [0, 1].

    Raises
    ------
    ValueError
        If weights shape mismatches, contains negatives, or sums to zero.
    """
    n_models = pred_set.probabilities.shape[1]
    if weights.shape != (n_models,):
        raise ValueError(
            f"weights must have shape ({n_models},), got {weights.shape}"
        )
    if np.any(weights < 0.0):
        raise ValueError("weights must be non-negative")
    total = np.sum(weights)
    if total < 1e-12:
        raise ValueError("weights must not all be zero")

    w = weights / total
    return pred_set.probabilities @ w


# ===================================================================
# Helper: compute_accuracy_weights
# ===================================================================

def compute_accuracy_weights(
    val_pred_set: PredictionSet,
    val_labels: np.ndarray,
) -> np.ndarray:
    """Compute normalized accuracy weights from validation predictions.

    Parameters
    ----------
    val_pred_set : PredictionSet
        Predictions on validation set.
    val_labels : np.ndarray
        True labels, shape (n_val,).

    Returns
    -------
    np.ndarray
        Normalized weights summing to 1, shape (n_models,).

    Raises
    ------
    ValueError
        If label length mismatches prediction count.
    """
    n_val = val_pred_set.probabilities.shape[0]
    if val_labels.shape[0] != n_val:
        raise ValueError(
            f"Label length {val_labels.shape[0]} does not match "
            f"prediction count {n_val}"
        )

    n_models = val_pred_set.probabilities.shape[1]
    accuracies = np.zeros(n_models, dtype=np.float64)

    for j in range(n_models):
        preds = (val_pred_set.probabilities[:, j] > 0.5).astype(np.int32)
        accuracies[j] = np.mean(preds == val_labels)

    total = np.sum(accuracies)
    if total < 1e-12:
        # All models have zero accuracy — fall back to uniform
        return np.ones(n_models, dtype=np.float64) / n_models

    return accuracies / total


# ===================================================================
# C: Stacking Meta-Learner
# ===================================================================

class StackingMetaLearner:
    """Logistic regression trained on model output probabilities.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed
        self._model: LogisticRegression | None = None
        self.n_features_: int = 0

    def fit(self, val_pred_set: PredictionSet, val_labels: np.ndarray) -> StackingMetaLearner:
        """Train on validation model outputs.

        Parameters
        ----------
        val_pred_set : PredictionSet
            Validation predictions — uses probabilities only.
        val_labels : np.ndarray
            True validation labels, shape (n_val,).

        Returns
        -------
        self
        """
        X = val_pred_set.probabilities  # (n_val, n_models)
        self.n_features_ = X.shape[1]
        self._model = LogisticRegression(
            random_state=self._seed,
            max_iter=1000,
            solver="lbfgs",
        )
        self._model.fit(X, val_labels)
        return self

    def predict(self, pred_set: PredictionSet) -> np.ndarray:
        """Predict fraud probabilities via the stacking meta-learner.

        Parameters
        ----------
        pred_set : PredictionSet
            Test predictions.

        Returns
        -------
        np.ndarray
            Fraud probabilities, shape (n,), values in [0, 1].

        Raises
        ------
        RuntimeError
            If called before fit().
        """
        if self._model is None:
            raise RuntimeError("StackingMetaLearner must be fit before predict")
        X = pred_set.probabilities
        return self._model.predict_proba(X)[:, 1].astype(np.float64)


# ===================================================================
# D: Bayesian Model Averaging
# ===================================================================

def compute_bma_weights(
    val_pred_set: PredictionSet,
    val_labels: np.ndarray,
) -> np.ndarray:
    """Compute BMA weights from validation log-likelihoods.

    For each model, computes the mean log-likelihood on the validation
    set, then converts to normalized weights via softmax.

    Parameters
    ----------
    val_pred_set : PredictionSet
        Validation predictions.
    val_labels : np.ndarray
        True validation labels, shape (n_val,).

    Returns
    -------
    np.ndarray
        Normalized BMA weights summing to 1, shape (n_models,).
    """
    n_models = val_pred_set.probabilities.shape[1]
    eps = 1e-15  # Avoid log(0)

    log_liks = np.zeros(n_models, dtype=np.float64)
    for j in range(n_models):
        p = np.clip(val_pred_set.probabilities[:, j], eps, 1.0 - eps)
        # Log-likelihood: sum of log(p) for fraud, log(1-p) for legit
        ll = val_labels * np.log(p) + (1 - val_labels) * np.log(1 - p)
        log_liks[j] = np.mean(ll)

    # Softmax to convert to weights (numerically stable)
    log_liks -= np.max(log_liks)
    weights = np.exp(log_liks)
    return weights / np.sum(weights)


def bayesian_model_average(
    pred_set: PredictionSet,
    val_pred_set: PredictionSet,
    val_labels: np.ndarray,
) -> np.ndarray:
    """BMA fusion using validation log-likelihood weights.

    Parameters
    ----------
    pred_set : PredictionSet
        Test predictions.
    val_pred_set : PredictionSet
        Validation predictions (for weight computation).
    val_labels : np.ndarray
        True validation labels.

    Returns
    -------
    np.ndarray
        BMA-fused fraud scores, shape (n,), values in [0, 1].
    """
    weights = compute_bma_weights(val_pred_set, val_labels)
    return pred_set.probabilities @ weights


# ===================================================================
# E: Noisy-OR
# ===================================================================

def noisy_or(pred_set: PredictionSet) -> np.ndarray:
    """Noisy-OR fusion: P(fraud) = 1 − ∏(1 − pᵢ).

    Common in probabilistic fraud detection systems. Assumes independent
    causal sources — any single source can trigger fraud.

    Parameters
    ----------
    pred_set : PredictionSet
        Per-model predictions.

    Returns
    -------
    np.ndarray
        Fused fraud scores, shape (n,), values in [0, 1].
    """
    complement = 1.0 - pred_set.probabilities  # (n, n_models)
    product = np.prod(complement, axis=1)       # (n,)
    return 1.0 - product


# ===================================================================
# F: SL Cumulative Scores
# ===================================================================

def sl_cumulative_scores(
    pred_set: PredictionSet,
    base_rate: float = 0.035,
) -> np.ndarray:
    """Cumulative SL fusion → expected probability.

    Converts each model's output to an Opinion, cumulatively fuses
    per-transaction, and returns expected_probability.

    Parameters
    ----------
    pred_set : PredictionSet
        Per-model predictions (probabilities and uncertainties).
    base_rate : float
        Prior fraud rate for opinion construction.

    Returns
    -------
    np.ndarray
        Fused fraud scores, shape (n,), values in [0, 1].
    """
    n_txns = pred_set.probabilities.shape[0]
    scores = np.zeros(n_txns, dtype=np.float64)

    for i in range(n_txns):
        opinions = _build_opinions_for_txn(pred_set, i, base_rate)
        fused = cumulative_fuse(opinions)
        scores[i] = fused.expected_probability

    return scores


# ===================================================================
# G: SL Three-Way Decision
# ===================================================================

def sl_three_way(
    pred_set: PredictionSet,
    base_rate: float,
    decider: ThreeWayDecider,
) -> ThreeWayFusionResult:
    """SL cumulative fusion + conflict detection → three-way decision.

    Parameters
    ----------
    pred_set : PredictionSet
        Per-model predictions.
    base_rate : float
        Prior fraud rate for opinion construction.
    decider : ThreeWayDecider
        Decision engine with thresholds.

    Returns
    -------
    ThreeWayFusionResult
    """
    n_txns = pred_set.probabilities.shape[0]
    scores = np.zeros(n_txns, dtype=np.float64)
    decisions = np.empty(n_txns, dtype=object)
    conflicts = np.zeros(n_txns, dtype=np.float64)
    escalation_mask = np.zeros(n_txns, dtype=bool)

    for i in range(n_txns):
        opinions = _build_opinions_for_txn(pred_set, i, base_rate)
        fused = cumulative_fuse(opinions)
        conf = conflict_metric(opinions)

        result = decider.decide(fused, conflict=conf)

        scores[i] = fused.expected_probability
        decisions[i] = result.decision
        conflicts[i] = conf
        escalation_mask[i] = result.decision == Decision.ESCALATE

    return ThreeWayFusionResult(
        scores=scores,
        decisions=decisions,
        conflicts=conflicts,
        escalation_mask=escalation_mask,
    )


# ===================================================================
# H: SL Robust Three-Way
# ===================================================================

def sl_robust_three_way(
    pred_set: PredictionSet,
    base_rate: float,
    decider: ThreeWayDecider,
    robust_threshold: float = 0.15,
) -> ThreeWayFusionResult:
    """Robust SL fusion (Byzantine-resilient) → three-way decision.

    Uses robust_fuse to detect and exclude outlier opinions before
    fusing, providing resilience against compromised models.

    Parameters
    ----------
    pred_set : PredictionSet
        Per-model predictions.
    base_rate : float
        Prior fraud rate for opinion construction.
    decider : ThreeWayDecider
        Decision engine with thresholds.
    robust_threshold : float
        Distance threshold for outlier exclusion in robust_fuse.

    Returns
    -------
    ThreeWayFusionResult
        Includes excluded_counts per transaction.
    """
    n_txns = pred_set.probabilities.shape[0]
    scores = np.zeros(n_txns, dtype=np.float64)
    decisions = np.empty(n_txns, dtype=object)
    conflicts = np.zeros(n_txns, dtype=np.float64)
    escalation_mask = np.zeros(n_txns, dtype=bool)
    excluded_counts = np.zeros(n_txns, dtype=np.int32)

    for i in range(n_txns):
        opinions = _build_opinions_for_txn(pred_set, i, base_rate)

        # Robust fusion — may exclude outlier opinions
        robust_result = robust_fuse(opinions, threshold=robust_threshold)
        fused = robust_result.fused
        excluded_counts[i] = len(robust_result.excluded_indices)

        # Conflict on the retained opinions (if enough remain)
        retained_indices = [
            j for j in range(len(opinions))
            if j not in robust_result.excluded_indices
        ]
        if len(retained_indices) >= 2:
            retained_opinions = [opinions[j] for j in retained_indices]
            conf = conflict_metric(retained_opinions)
        else:
            conf = 0.0

        result = decider.decide(fused, conflict=conf)

        scores[i] = fused.expected_probability
        decisions[i] = result.decision
        conflicts[i] = conf
        escalation_mask[i] = result.decision == Decision.ESCALATE

    return ThreeWayFusionResult(
        scores=scores,
        decisions=decisions,
        conflicts=conflicts,
        escalation_mask=escalation_mask,
        excluded_counts=excluded_counts,
    )


# ===================================================================
# I: Confidence-as-Feature Meta-Learner
# ===================================================================

class ConfidenceFeatureLearner:
    """Meta-learner using per-source probabilities AND uncertainties as features.

    Unlike StackingMetaLearner (which uses only probabilities),
    this learner also feeds uncertainty estimates as features.
    The intuition: the *confidence profile* (high uncertainty on device
    fingerprint, low on transaction monitor) is itself predictive.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed
        self._model: LogisticRegression | None = None
        self.n_features_: int = 0

    @staticmethod
    def _build_features(pred_set: PredictionSet) -> np.ndarray:
        """Build feature matrix: [probabilities | uncertainties].

        Parameters
        ----------
        pred_set : PredictionSet

        Returns
        -------
        np.ndarray
            Shape (n_txns, 2 * n_models).
        """
        return np.hstack([pred_set.probabilities, pred_set.uncertainties])

    def fit(
        self,
        val_pred_set: PredictionSet,
        val_labels: np.ndarray,
    ) -> ConfidenceFeatureLearner:
        """Train on validation model outputs + uncertainties.

        Parameters
        ----------
        val_pred_set : PredictionSet
            Validation predictions.
        val_labels : np.ndarray
            True validation labels.

        Returns
        -------
        self
        """
        X = self._build_features(val_pred_set)
        self.n_features_ = X.shape[1]
        self._model = LogisticRegression(
            random_state=self._seed,
            max_iter=1000,
            solver="lbfgs",
        )
        self._model.fit(X, val_labels)
        return self

    def predict(self, pred_set: PredictionSet) -> np.ndarray:
        """Predict fraud probabilities using confidence features.

        Parameters
        ----------
        pred_set : PredictionSet

        Returns
        -------
        np.ndarray
            Fraud probabilities, shape (n,), values in [0, 1].

        Raises
        ------
        RuntimeError
            If called before fit().
        """
        if self._model is None:
            raise RuntimeError("ConfidenceFeatureLearner must be fit before predict")
        X = self._build_features(pred_set)
        return self._model.predict_proba(X)[:, 1].astype(np.float64)


# ===================================================================
# Internal helper
# ===================================================================

def _build_opinions_for_txn(
    pred_set: PredictionSet,
    txn_idx: int,
    base_rate: float,
) -> list[Opinion]:
    """Build per-source Opinion objects for a single transaction.

    Parameters
    ----------
    pred_set : PredictionSet
    txn_idx : int
        Transaction index.
    base_rate : float
        Prior fraud rate.

    Returns
    -------
    list[Opinion]
        One opinion per model.
    """
    n_models = pred_set.probabilities.shape[1]
    opinions: list[Opinion] = []
    for j in range(n_models):
        o = Opinion.from_confidence(
            probability=float(pred_set.probabilities[txn_idx, j]),
            uncertainty=float(pred_set.uncertainties[txn_idx, j]),
            base_rate=base_rate,
        )
        opinions.append(o)
    return opinions
