"""Tests for E-FD2 fusion strategies (Unit 1).

Nine treatment arms for multi-source fraud signal fusion:

Scalar baselines:
    A: majority_vote           — fraction of models predicting fraud
    B: weighted_average        — accuracy-weighted mean of probabilities
    C: StackingMetaLearner     — logistic regression on model outputs (fit/predict)
    D: bayesian_model_average  — BMA weights from validation log-likelihoods
    E: noisy_or                — P(fraud) = 1 − ∏(1 − pᵢ)

SLFD treatments:
    F: sl_cumulative_scores    — cumulative_fuse → expected_probability
    G: sl_three_way            — F + conflict_metric → ThreeWayDecider
    H: sl_robust_three_way     — robust_fuse → ThreeWayDecider
    I: ConfidenceFeatureLearner — per-source (prob, uncertainty) as meta-features

All strategies consume a PredictionSet from the 4-model ensemble.
Stateless strategies are pure functions; C and I are classes with fit/predict.
"""

import math

import numpy as np
import pytest

from slfd.opinion import Opinion
from slfd.decision import Decision, ThreeWayDecider
from slfd.models.ensemble import PredictionSet


# ===================================================================
# These are the imports that WILL FAIL until we implement strategies.py
# ===================================================================
from slfd.strategies import (
    # Baselines
    majority_vote,          # A
    weighted_average,       # B
    StackingMetaLearner,    # C
    bayesian_model_average, # D
    noisy_or,               # E
    # SL treatments
    sl_cumulative_scores,   # F
    sl_three_way,           # G
    sl_robust_three_way,    # H
    ConfidenceFeatureLearner,  # I
    # Result container for three-way strategies
    ThreeWayFusionResult,
    # Weight computation helpers
    compute_accuracy_weights,
    compute_bma_weights,
)


# ===================================================================
# Shared fixtures
# ===================================================================

def _make_prediction_set(
    n: int = 20,
    n_models: int = 4,
    seed: int = 42,
) -> PredictionSet:
    """Small synthetic PredictionSet for testing."""
    rng = np.random.default_rng(seed)
    probs = rng.uniform(0.0, 1.0, size=(n, n_models))
    uncerts = rng.uniform(0.05, 0.50, size=(n, n_models))
    names = [f"model_{i}" for i in range(n_models)]
    return PredictionSet(
        probabilities=probs,
        uncertainties=uncerts,
        model_names=names,
    )


def _make_labels(n: int = 20, fraud_rate: float = 0.3, seed: int = 42) -> np.ndarray:
    """Synthetic binary labels for validation fitting."""
    rng = np.random.default_rng(seed)
    return (rng.uniform(0.0, 1.0, size=n) < fraud_rate).astype(np.int32)


@pytest.fixture
def pred_set():
    """Standard 20-transaction, 4-model prediction set."""
    return _make_prediction_set(n=20, n_models=4, seed=42)


@pytest.fixture
def val_pred_set():
    """Separate validation prediction set for fitting."""
    return _make_prediction_set(n=50, n_models=4, seed=99)


@pytest.fixture
def val_labels():
    """Validation labels for fitting."""
    return _make_labels(n=50, fraud_rate=0.3, seed=99)


@pytest.fixture
def labels():
    """Test labels (for weight computation tests)."""
    return _make_labels(n=20, fraud_rate=0.3, seed=42)


# ===================================================================
# Unanimous-agreement fixtures (for edge-case tests)
# ===================================================================

@pytest.fixture
def all_fraud_preds():
    """All models predict near-certain fraud."""
    n, m = 10, 4
    probs = np.full((n, m), 0.95)
    uncerts = np.full((n, m), 0.05)
    return PredictionSet(
        probabilities=probs,
        uncertainties=uncerts,
        model_names=[f"model_{i}" for i in range(m)],
    )


@pytest.fixture
def all_legit_preds():
    """All models predict near-certain legitimate."""
    n, m = 10, 4
    probs = np.full((n, m), 0.05)
    uncerts = np.full((n, m), 0.05)
    return PredictionSet(
        probabilities=probs,
        uncertainties=uncerts,
        model_names=[f"model_{i}" for i in range(m)],
    )


@pytest.fixture
def conflicting_preds():
    """Two models say fraud, two say legit — high conflict."""
    n, m = 10, 4
    probs = np.zeros((n, m))
    probs[:, 0] = 0.90
    probs[:, 1] = 0.85
    probs[:, 2] = 0.10
    probs[:, 3] = 0.15
    uncerts = np.full((n, m), 0.10)
    return PredictionSet(
        probabilities=probs,
        uncertainties=uncerts,
        model_names=[f"model_{i}" for i in range(m)],
    )


# ===================================================================
# A: Majority Vote
# ===================================================================

class TestMajorityVote:
    """Arm A — fraction of models predicting fraud (prob > 0.5)."""

    def test_returns_ndarray(self, pred_set):
        result = majority_vote(pred_set)
        assert isinstance(result, np.ndarray)

    def test_shape_matches_transactions(self, pred_set):
        result = majority_vote(pred_set)
        assert result.shape == (20,)

    def test_values_in_valid_range(self, pred_set):
        result = majority_vote(pred_set)
        # Fraction of models: values must be in {0, 0.25, 0.5, 0.75, 1.0}
        # for 4 models. More generally, in [0, 1].
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_unanimous_fraud(self, all_fraud_preds):
        """All models predict fraud → vote fraction = 1.0."""
        result = majority_vote(all_fraud_preds)
        np.testing.assert_array_equal(result, 1.0)

    def test_unanimous_legit(self, all_legit_preds):
        """All models predict legit → vote fraction = 0.0."""
        result = majority_vote(all_legit_preds)
        np.testing.assert_array_equal(result, 0.0)

    def test_split_vote(self, conflicting_preds):
        """Half fraud, half legit → 0.5."""
        result = majority_vote(conflicting_preds)
        np.testing.assert_array_almost_equal(result, 0.5)

    def test_custom_threshold(self, pred_set):
        """Majority vote can use a custom probability threshold."""
        result_default = majority_vote(pred_set)
        result_low = majority_vote(pred_set, threshold=0.3)
        # Lower threshold → more models counted as fraud → higher vote
        assert np.mean(result_low) >= np.mean(result_default)

    def test_deterministic(self, pred_set):
        """Same input → same output."""
        r1 = majority_vote(pred_set)
        r2 = majority_vote(pred_set)
        np.testing.assert_array_equal(r1, r2)


# ===================================================================
# B: Weighted Average
# ===================================================================

class TestWeightedAverage:
    """Arm B — accuracy-weighted mean of model probabilities."""

    def test_returns_ndarray(self, pred_set):
        weights = np.array([0.85, 0.90, 0.75, 0.95])
        result = weighted_average(pred_set, weights)
        assert isinstance(result, np.ndarray)

    def test_shape_matches_transactions(self, pred_set):
        weights = np.array([0.85, 0.90, 0.75, 0.95])
        result = weighted_average(pred_set, weights)
        assert result.shape == (20,)

    def test_values_in_zero_one(self, pred_set):
        weights = np.array([0.85, 0.90, 0.75, 0.95])
        result = weighted_average(pred_set, weights)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_equal_weights_equals_simple_mean(self, pred_set):
        """Equal weights should reduce to simple averaging."""
        weights = np.array([1.0, 1.0, 1.0, 1.0])
        result = weighted_average(pred_set, weights)
        expected = pred_set.scalar_average()
        np.testing.assert_array_almost_equal(result, expected)

    def test_single_model_dominance(self, conflicting_preds):
        """Giving all weight to one model should reproduce that model's output."""
        weights = np.array([1.0, 0.0, 0.0, 0.0])
        result = weighted_average(conflicting_preds, weights)
        np.testing.assert_array_almost_equal(result, conflicting_preds.probabilities[:, 0])

    def test_weight_length_mismatch_raises(self, pred_set):
        """Weights must match number of models."""
        with pytest.raises(ValueError, match="weights"):
            weighted_average(pred_set, np.array([0.5, 0.5]))

    def test_negative_weights_raise(self, pred_set):
        """Weights must be non-negative."""
        with pytest.raises(ValueError, match="non-negative"):
            weighted_average(pred_set, np.array([0.5, -0.1, 0.3, 0.3]))

    def test_zero_total_weight_raises(self, pred_set):
        """All-zero weights are degenerate — should raise."""
        with pytest.raises(ValueError, match="zero"):
            weighted_average(pred_set, np.array([0.0, 0.0, 0.0, 0.0]))


# ===================================================================
# Helper: compute_accuracy_weights
# ===================================================================

class TestComputeAccuracyWeights:
    """Compute per-model accuracy weights from validation set."""

    def test_returns_ndarray(self, val_pred_set, val_labels):
        weights = compute_accuracy_weights(val_pred_set, val_labels)
        assert isinstance(weights, np.ndarray)

    def test_one_weight_per_model(self, val_pred_set, val_labels):
        weights = compute_accuracy_weights(val_pred_set, val_labels)
        assert weights.shape == (4,)

    def test_weights_non_negative(self, val_pred_set, val_labels):
        weights = compute_accuracy_weights(val_pred_set, val_labels)
        assert np.all(weights >= 0.0)

    def test_weights_sum_to_one(self, val_pred_set, val_labels):
        """Weights should be normalized to sum to 1."""
        weights = compute_accuracy_weights(val_pred_set, val_labels)
        assert abs(np.sum(weights) - 1.0) < 1e-9

    def test_label_length_mismatch_raises(self, val_pred_set):
        """Labels must match transaction count."""
        bad_labels = np.array([0, 1, 0])
        with pytest.raises(ValueError, match="length"):
            compute_accuracy_weights(val_pred_set, bad_labels)


# ===================================================================
# C: Stacking Meta-Learner
# ===================================================================

class TestStackingMetaLearner:
    """Arm C — logistic regression trained on model outputs."""

    def test_instantiation(self):
        learner = StackingMetaLearner(seed=42)
        assert learner is not None

    def test_fit_returns_self(self, val_pred_set, val_labels):
        learner = StackingMetaLearner(seed=42)
        result = learner.fit(val_pred_set, val_labels)
        assert result is learner

    def test_predict_before_fit_raises(self, pred_set):
        learner = StackingMetaLearner(seed=42)
        with pytest.raises(RuntimeError, match="[Ff]it"):
            learner.predict(pred_set)

    def test_predict_returns_ndarray(self, pred_set, val_pred_set, val_labels):
        learner = StackingMetaLearner(seed=42)
        learner.fit(val_pred_set, val_labels)
        result = learner.predict(pred_set)
        assert isinstance(result, np.ndarray)

    def test_predict_shape(self, pred_set, val_pred_set, val_labels):
        learner = StackingMetaLearner(seed=42)
        learner.fit(val_pred_set, val_labels)
        result = learner.predict(pred_set)
        assert result.shape == (20,)

    def test_predict_values_in_zero_one(self, pred_set, val_pred_set, val_labels):
        learner = StackingMetaLearner(seed=42)
        learner.fit(val_pred_set, val_labels)
        result = learner.predict(pred_set)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_deterministic(self, pred_set, val_pred_set, val_labels):
        l1 = StackingMetaLearner(seed=42)
        l1.fit(val_pred_set, val_labels)
        r1 = l1.predict(pred_set)

        l2 = StackingMetaLearner(seed=42)
        l2.fit(val_pred_set, val_labels)
        r2 = l2.predict(pred_set)

        np.testing.assert_array_equal(r1, r2)


# ===================================================================
# D: Bayesian Model Average
# ===================================================================

class TestBayesianModelAverage:
    """Arm D — BMA using validation log-likelihood weights."""

    def test_returns_ndarray(self, pred_set, val_pred_set, val_labels):
        result = bayesian_model_average(pred_set, val_pred_set, val_labels)
        assert isinstance(result, np.ndarray)

    def test_shape_matches_transactions(self, pred_set, val_pred_set, val_labels):
        result = bayesian_model_average(pred_set, val_pred_set, val_labels)
        assert result.shape == (20,)

    def test_values_in_zero_one(self, pred_set, val_pred_set, val_labels):
        result = bayesian_model_average(pred_set, val_pred_set, val_labels)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_deterministic(self, pred_set, val_pred_set, val_labels):
        r1 = bayesian_model_average(pred_set, val_pred_set, val_labels)
        r2 = bayesian_model_average(pred_set, val_pred_set, val_labels)
        np.testing.assert_array_equal(r1, r2)


# ===================================================================
# Helper: compute_bma_weights
# ===================================================================

class TestComputeBMAWeights:
    """BMA weights from validation log-likelihoods."""

    def test_returns_ndarray(self, val_pred_set, val_labels):
        weights = compute_bma_weights(val_pred_set, val_labels)
        assert isinstance(weights, np.ndarray)

    def test_one_weight_per_model(self, val_pred_set, val_labels):
        weights = compute_bma_weights(val_pred_set, val_labels)
        assert weights.shape == (4,)

    def test_weights_sum_to_one(self, val_pred_set, val_labels):
        weights = compute_bma_weights(val_pred_set, val_labels)
        assert abs(np.sum(weights) - 1.0) < 1e-9

    def test_weights_non_negative(self, val_pred_set, val_labels):
        weights = compute_bma_weights(val_pred_set, val_labels)
        assert np.all(weights >= 0.0)

    def test_better_model_gets_higher_weight(self):
        """A model with higher log-likelihood should get higher BMA weight."""
        n = 100
        rng = np.random.default_rng(123)
        labels = (rng.uniform(size=n) < 0.3).astype(np.int32)

        # Model 0: good predictions (close to true labels)
        good_probs = labels * 0.9 + (1 - labels) * 0.1
        # Model 1: random predictions
        bad_probs = rng.uniform(0.2, 0.8, size=n)

        probs = np.column_stack([good_probs, bad_probs])
        uncerts = np.full((n, 2), 0.1)
        pset = PredictionSet(
            probabilities=probs,
            uncertainties=uncerts,
            model_names=["good", "bad"],
        )
        weights = compute_bma_weights(pset, labels)
        assert weights[0] > weights[1]


# ===================================================================
# E: Noisy-OR
# ===================================================================

class TestNoisyOr:
    """Arm E — P(fraud) = 1 − ∏(1 − pᵢ)."""

    def test_returns_ndarray(self, pred_set):
        result = noisy_or(pred_set)
        assert isinstance(result, np.ndarray)

    def test_shape_matches_transactions(self, pred_set):
        result = noisy_or(pred_set)
        assert result.shape == (20,)

    def test_values_in_zero_one(self, pred_set):
        result = noisy_or(pred_set)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_all_zero_gives_zero(self):
        """If all models predict 0, noisy-OR is 0."""
        n, m = 5, 4
        probs = np.zeros((n, m))
        uncerts = np.full((n, m), 0.1)
        pset = PredictionSet(probabilities=probs, uncertainties=uncerts,
                             model_names=[f"m{i}" for i in range(m)])
        result = noisy_or(pset)
        np.testing.assert_array_almost_equal(result, 0.0)

    def test_one_certain_gives_one(self):
        """If any model predicts 1.0, noisy-OR is 1.0."""
        n, m = 5, 4
        probs = np.zeros((n, m))
        probs[:, 0] = 1.0
        uncerts = np.full((n, m), 0.1)
        pset = PredictionSet(probabilities=probs, uncertainties=uncerts,
                             model_names=[f"m{i}" for i in range(m)])
        result = noisy_or(pset)
        np.testing.assert_array_almost_equal(result, 1.0)

    def test_known_calculation(self):
        """Verify against hand-calculated value: 1 − (1−0.3)(1−0.4) = 0.58."""
        probs = np.array([[0.3, 0.4]])
        uncerts = np.array([[0.1, 0.1]])
        pset = PredictionSet(probabilities=probs, uncertainties=uncerts,
                             model_names=["a", "b"])
        result = noisy_or(pset)
        expected = 1.0 - (1.0 - 0.3) * (1.0 - 0.4)
        assert result[0] == pytest.approx(expected, abs=1e-9)

    def test_noisy_or_higher_than_mean(self, pred_set):
        """Noisy-OR should generally be >= mean for probabilities in [0,1].
        Specifically: 1 − ∏(1−p_i) ≥ mean(p_i) when probabilities are
        moderate, as noisy-OR is an optimistic combiner."""
        nor = noisy_or(pred_set)
        avg = pred_set.scalar_average()
        # Not strictly true for all values, but should hold on average
        # for diverse predictions
        assert np.mean(nor) >= np.mean(avg) - 0.1

    def test_deterministic(self, pred_set):
        r1 = noisy_or(pred_set)
        r2 = noisy_or(pred_set)
        np.testing.assert_array_equal(r1, r2)


# ===================================================================
# F: SL Cumulative Scores
# ===================================================================

class TestSLCumulativeScores:
    """Arm F — cumulative_fuse → expected_probability."""

    def test_returns_ndarray(self, pred_set):
        result = sl_cumulative_scores(pred_set, base_rate=0.035)
        assert isinstance(result, np.ndarray)

    def test_shape_matches_transactions(self, pred_set):
        result = sl_cumulative_scores(pred_set, base_rate=0.035)
        assert result.shape == (20,)

    def test_values_in_zero_one(self, pred_set):
        result = sl_cumulative_scores(pred_set, base_rate=0.035)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_unanimous_fraud_high_score(self, all_fraud_preds):
        """All models predicting fraud → high fused score."""
        result = sl_cumulative_scores(all_fraud_preds, base_rate=0.035)
        assert np.all(result > 0.8)

    def test_unanimous_legit_low_score(self, all_legit_preds):
        """All models predicting legit → low fused score."""
        result = sl_cumulative_scores(all_legit_preds, base_rate=0.035)
        assert np.all(result < 0.2)

    def test_base_rate_affects_result(self, pred_set):
        """Different base rates should produce different fused scores."""
        r_low = sl_cumulative_scores(pred_set, base_rate=0.01)
        r_high = sl_cumulative_scores(pred_set, base_rate=0.50)
        # Base rate shifts expected probability upward
        assert np.mean(r_high) > np.mean(r_low)

    def test_deterministic(self, pred_set):
        r1 = sl_cumulative_scores(pred_set, base_rate=0.035)
        r2 = sl_cumulative_scores(pred_set, base_rate=0.035)
        np.testing.assert_array_equal(r1, r2)


# ===================================================================
# ThreeWayFusionResult container
# ===================================================================

class TestThreeWayFusionResult:
    """Validate the result container for three-way strategies (G, H, I)."""

    def test_has_required_fields(self):
        """ThreeWayFusionResult must carry scores, decisions, conflicts, escalation mask."""
        result = ThreeWayFusionResult(
            scores=np.array([0.5, 0.3]),
            decisions=np.array([Decision.BLOCK, Decision.APPROVE]),
            conflicts=np.array([0.1, 0.05]),
            escalation_mask=np.array([False, False]),
        )
        assert result.scores.shape == (2,)
        assert result.decisions.shape == (2,)
        assert result.conflicts.shape == (2,)
        assert result.escalation_mask.shape == (2,)

    def test_escalation_rate_property(self):
        """Should compute fraction of escalated transactions."""
        result = ThreeWayFusionResult(
            scores=np.array([0.5, 0.3, 0.8, 0.1]),
            decisions=np.array([
                Decision.ESCALATE, Decision.APPROVE,
                Decision.BLOCK, Decision.ESCALATE,
            ]),
            conflicts=np.array([0.4, 0.05, 0.1, 0.5]),
            escalation_mask=np.array([True, False, False, True]),
        )
        assert result.escalation_rate == pytest.approx(0.5)

    def test_auto_decided_mask_property(self):
        """Inverse of escalation mask — transactions that got auto-decided."""
        result = ThreeWayFusionResult(
            scores=np.array([0.5, 0.3, 0.8]),
            decisions=np.array([Decision.ESCALATE, Decision.APPROVE, Decision.BLOCK]),
            conflicts=np.array([0.4, 0.05, 0.1]),
            escalation_mask=np.array([True, False, False]),
        )
        auto = result.auto_decided_mask
        np.testing.assert_array_equal(auto, [False, True, True])


# ===================================================================
# G: SL Three-Way Decision
# ===================================================================

class TestSLThreeWay:
    """Arm G — SL cumulative fusion + conflict detection → three-way decision."""

    def _default_decider(self):
        return ThreeWayDecider(
            block_threshold=0.6,
            approve_threshold=0.6,
            escalate_uncertainty=0.4,
            escalate_conflict=0.3,
        )

    def test_returns_three_way_result(self, pred_set):
        result = sl_three_way(pred_set, base_rate=0.035,
                              decider=self._default_decider())
        assert isinstance(result, ThreeWayFusionResult)

    def test_scores_shape(self, pred_set):
        result = sl_three_way(pred_set, base_rate=0.035,
                              decider=self._default_decider())
        assert result.scores.shape == (20,)

    def test_decisions_shape(self, pred_set):
        result = sl_three_way(pred_set, base_rate=0.035,
                              decider=self._default_decider())
        assert result.decisions.shape == (20,)

    def test_decisions_are_valid_enums(self, pred_set):
        result = sl_three_way(pred_set, base_rate=0.035,
                              decider=self._default_decider())
        valid = {Decision.BLOCK, Decision.APPROVE, Decision.ESCALATE}
        for d in result.decisions:
            assert d in valid

    def test_conflicts_computed(self, pred_set):
        result = sl_three_way(pred_set, base_rate=0.035,
                              decider=self._default_decider())
        assert result.conflicts.shape == (20,)
        assert np.all(result.conflicts >= 0.0)
        assert np.all(result.conflicts <= 1.0)

    def test_escalation_mask_matches_decisions(self, pred_set):
        result = sl_three_way(pred_set, base_rate=0.035,
                              decider=self._default_decider())
        for i in range(len(result.decisions)):
            if result.decisions[i] == Decision.ESCALATE:
                assert result.escalation_mask[i] is True or result.escalation_mask[i] == True
            else:
                assert result.escalation_mask[i] is False or result.escalation_mask[i] == False

    def test_conflicting_inputs_cause_escalation(self, conflicting_preds):
        """High-conflict inputs should trigger escalation."""
        decider = ThreeWayDecider(
            block_threshold=0.6,
            approve_threshold=0.6,
            escalate_uncertainty=0.4,
            escalate_conflict=0.2,  # low threshold to catch conflict
        )
        result = sl_three_way(conflicting_preds, base_rate=0.035, decider=decider)
        # Most or all conflicting transactions should be escalated
        assert result.escalation_rate > 0.5

    def test_unanimous_fraud_blocks(self, all_fraud_preds):
        """All models agree on fraud → should BLOCK."""
        decider = ThreeWayDecider(
            block_threshold=0.6,
            approve_threshold=0.6,
            escalate_uncertainty=0.4,
            escalate_conflict=0.3,
        )
        result = sl_three_way(all_fraud_preds, base_rate=0.035, decider=decider)
        block_rate = np.mean([d == Decision.BLOCK for d in result.decisions])
        assert block_rate > 0.5

    def test_unanimous_legit_approves(self, all_legit_preds):
        """All models agree on legit → should APPROVE."""
        decider = ThreeWayDecider(
            block_threshold=0.6,
            approve_threshold=0.6,
            escalate_uncertainty=0.4,
            escalate_conflict=0.3,
        )
        result = sl_three_way(all_legit_preds, base_rate=0.035, decider=decider)
        approve_rate = np.mean([d == Decision.APPROVE for d in result.decisions])
        assert approve_rate > 0.5

    def test_deterministic(self, pred_set):
        decider = self._default_decider()
        r1 = sl_three_way(pred_set, base_rate=0.035, decider=decider)
        r2 = sl_three_way(pred_set, base_rate=0.035, decider=decider)
        np.testing.assert_array_equal(r1.scores, r2.scores)


# ===================================================================
# H: SL Robust Three-Way
# ===================================================================

class TestSLRobustThreeWay:
    """Arm H — robust_fuse (Byzantine-resilient) → three-way decision."""

    def _default_decider(self):
        return ThreeWayDecider(
            block_threshold=0.6,
            approve_threshold=0.6,
            escalate_uncertainty=0.4,
            escalate_conflict=0.3,
        )

    def test_returns_three_way_result(self, pred_set):
        result = sl_robust_three_way(
            pred_set, base_rate=0.035,
            decider=self._default_decider(), robust_threshold=0.15,
        )
        assert isinstance(result, ThreeWayFusionResult)

    def test_scores_shape(self, pred_set):
        result = sl_robust_three_way(
            pred_set, base_rate=0.035,
            decider=self._default_decider(), robust_threshold=0.15,
        )
        assert result.scores.shape == (20,)

    def test_decisions_shape(self, pred_set):
        result = sl_robust_three_way(
            pred_set, base_rate=0.035,
            decider=self._default_decider(), robust_threshold=0.15,
        )
        assert result.decisions.shape == (20,)

    def test_has_excluded_counts(self, pred_set):
        """Result should carry per-transaction count of excluded sources."""
        result = sl_robust_three_way(
            pred_set, base_rate=0.035,
            decider=self._default_decider(), robust_threshold=0.15,
        )
        assert hasattr(result, "excluded_counts")
        assert result.excluded_counts.shape == (20,)
        assert np.all(result.excluded_counts >= 0)

    def test_rogue_model_detected(self):
        """One model gives inverted predictions — robust_fuse should exclude it."""
        n, m = 20, 4
        rng = np.random.default_rng(42)
        # 3 honest models: predict ~0.8 for fraud
        honest_probs = rng.uniform(0.70, 0.90, size=(n, 3))
        # 1 rogue: inverted (predicts ~0.1)
        rogue_probs = rng.uniform(0.05, 0.15, size=(n, 1))
        probs = np.hstack([honest_probs, rogue_probs])
        uncerts = np.full((n, m), 0.10)
        pset = PredictionSet(
            probabilities=probs, uncertainties=uncerts,
            model_names=["honest_0", "honest_1", "honest_2", "rogue"],
        )
        decider = ThreeWayDecider(
            block_threshold=0.5, approve_threshold=0.5,
            escalate_uncertainty=0.4, escalate_conflict=0.3,
        )
        result = sl_robust_three_way(
            pset, base_rate=0.035, decider=decider, robust_threshold=0.15,
        )
        # Rogue model should be excluded for most transactions
        assert np.mean(result.excluded_counts) > 0

    def test_deterministic(self, pred_set):
        decider = self._default_decider()
        r1 = sl_robust_three_way(pred_set, base_rate=0.035,
                                 decider=decider, robust_threshold=0.15)
        r2 = sl_robust_three_way(pred_set, base_rate=0.035,
                                 decider=decider, robust_threshold=0.15)
        np.testing.assert_array_equal(r1.scores, r2.scores)


# ===================================================================
# I: Confidence-as-Feature Meta-Learner
# ===================================================================

class TestConfidenceFeatureLearner:
    """Arm I — per-source (probability, uncertainty) as meta-features."""

    def test_instantiation(self):
        learner = ConfidenceFeatureLearner(seed=42)
        assert learner is not None

    def test_fit_returns_self(self, val_pred_set, val_labels):
        learner = ConfidenceFeatureLearner(seed=42)
        result = learner.fit(val_pred_set, val_labels)
        assert result is learner

    def test_predict_before_fit_raises(self, pred_set):
        learner = ConfidenceFeatureLearner(seed=42)
        with pytest.raises(RuntimeError, match="[Ff]it"):
            learner.predict(pred_set)

    def test_predict_returns_ndarray(self, pred_set, val_pred_set, val_labels):
        learner = ConfidenceFeatureLearner(seed=42)
        learner.fit(val_pred_set, val_labels)
        result = learner.predict(pred_set)
        assert isinstance(result, np.ndarray)

    def test_predict_shape(self, pred_set, val_pred_set, val_labels):
        learner = ConfidenceFeatureLearner(seed=42)
        learner.fit(val_pred_set, val_labels)
        result = learner.predict(pred_set)
        assert result.shape == (20,)

    def test_predict_values_in_zero_one(self, pred_set, val_pred_set, val_labels):
        learner = ConfidenceFeatureLearner(seed=42)
        learner.fit(val_pred_set, val_labels)
        result = learner.predict(pred_set)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_uses_uncertainty_features(self, val_pred_set, val_labels):
        """The learner should use BOTH probabilities AND uncertainties.
        Verify by checking the feature matrix has 2x model count columns."""
        learner = ConfidenceFeatureLearner(seed=42)
        learner.fit(val_pred_set, val_labels)
        # 4 models → 4 probs + 4 uncertainties = 8 features
        assert learner.n_features_ == 8

    def test_more_features_than_stacking(self, val_pred_set, val_labels, pred_set):
        """Confidence learner uses more features than basic stacking."""
        stacker = StackingMetaLearner(seed=42)
        stacker.fit(val_pred_set, val_labels)

        conf_learner = ConfidenceFeatureLearner(seed=42)
        conf_learner.fit(val_pred_set, val_labels)

        # Stacking: n_models features. Confidence: 2 * n_models features.
        assert conf_learner.n_features_ > stacker.n_features_

    def test_deterministic(self, pred_set, val_pred_set, val_labels):
        l1 = ConfidenceFeatureLearner(seed=42)
        l1.fit(val_pred_set, val_labels)
        r1 = l1.predict(pred_set)

        l2 = ConfidenceFeatureLearner(seed=42)
        l2.fit(val_pred_set, val_labels)
        r2 = l2.predict(pred_set)

        np.testing.assert_array_equal(r1, r2)


# ===================================================================
# Cross-strategy properties
# ===================================================================

class TestCrossStrategyProperties:
    """Properties that should hold across all score-producing strategies."""

    def setup_method(self):
        self.pred_set = _make_prediction_set(n=50, n_models=4, seed=42)
        self.val_pred_set = _make_prediction_set(n=100, n_models=4, seed=99)
        self.val_labels = _make_labels(n=100, fraud_rate=0.3, seed=99)

    def _all_score_arrays(self) -> dict[str, np.ndarray]:
        """Compute scores from all 9 arms."""
        ps = self.pred_set
        vps = self.val_pred_set
        vl = self.val_labels

        acc_w = compute_accuracy_weights(vps, vl)

        stacker = StackingMetaLearner(seed=42)
        stacker.fit(vps, vl)

        conf_learner = ConfidenceFeatureLearner(seed=42)
        conf_learner.fit(vps, vl)

        decider = ThreeWayDecider(
            block_threshold=0.6, approve_threshold=0.6,
            escalate_uncertainty=0.4, escalate_conflict=0.3,
        )

        return {
            "A_majority_vote": majority_vote(ps),
            "B_weighted_avg": weighted_average(ps, acc_w),
            "C_stacking": stacker.predict(ps),
            "D_bma": bayesian_model_average(ps, vps, vl),
            "E_noisy_or": noisy_or(ps),
            "F_sl_cumulative": sl_cumulative_scores(ps, base_rate=0.035),
            "G_sl_three_way": sl_three_way(ps, base_rate=0.035, decider=decider).scores,
            "H_sl_robust": sl_robust_three_way(
                ps, base_rate=0.035, decider=decider, robust_threshold=0.15,
            ).scores,
            "I_conf_feature": conf_learner.predict(ps),
        }

    def test_all_produce_correct_length(self):
        """Every strategy produces one score per transaction."""
        arrays = self._all_score_arrays()
        for name, arr in arrays.items():
            assert arr.shape == (50,), f"{name} has wrong shape {arr.shape}"

    def test_all_in_valid_range(self):
        """Every strategy produces scores in [0, 1]."""
        arrays = self._all_score_arrays()
        for name, arr in arrays.items():
            assert np.all(arr >= 0.0), f"{name} has values < 0"
            assert np.all(arr <= 1.0), f"{name} has values > 1"

    def test_strategies_not_all_identical(self):
        """Different strategies should produce meaningfully different scores."""
        arrays = self._all_score_arrays()
        names = list(arrays.keys())
        found_different = False
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if not np.allclose(arrays[names[i]], arrays[names[j]], atol=0.01):
                    found_different = True
                    break
            if found_different:
                break
        assert found_different, "All strategies produced nearly identical scores"
