"""E-FD2: Multi-Source Fraud Signal Fusion experiment runner.

Orchestrates the full 9-arm fusion experiment:
    1. Train 4-model ensemble (or accept pre-computed predictions)
    2. Collect validation + test predictions
    3. Run all 9 fusion strategies (A–I)
    4. Evaluate each arm on PR-AUC, FPR@TPR, cost, escalation
    5. Pairwise McNemar's tests and bootstrap CIs
    6. Package results for serialization

Two entry points:
    run_efd2()                  — full pipeline (train + predict + evaluate)
    run_efd2_from_predictions() — evaluation only (pre-computed predictions)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from slfd.decision import ThreeWayDecider
from slfd.metrics import (
    ArmResult,
    CostConfig,
    McNemarResult,
    BootstrapCIResult,
    evaluate_arm,
    mcnemar_test,
    bootstrap_metric_ci,
)
from slfd.models.ensemble import PredictionSet, train_model_suite
from slfd.strategies import (
    majority_vote,
    weighted_average,
    compute_accuracy_weights,
    StackingMetaLearner,
    bayesian_model_average,
    noisy_or,
    sl_cumulative_scores,
    sl_three_way,
    sl_robust_three_way,
    ConfidenceFeatureLearner,
)


# ===================================================================
# Configuration
# ===================================================================

@dataclass
class EFD2Config:
    """Configuration for the E-FD2 experiment.

    Attributes
    ----------
    seed : int
        Global random seed for reproducibility.
    base_rate : float
        Prior fraud rate for SL opinion construction.
    cost_config : CostConfig
        Cost parameters for cost-sensitive evaluation.
    n_bootstrap : int
        Number of bootstrap resamples for CIs.
    decider : ThreeWayDecider
        Decision engine for three-way arms (G, H).
    robust_threshold : float
        Outlier threshold for robust fusion (H).
    """

    seed: int = 42
    base_rate: float = 0.035
    cost_config: CostConfig = field(default_factory=CostConfig)
    n_bootstrap: int = 1000
    decider: ThreeWayDecider = field(default_factory=lambda: ThreeWayDecider(
        block_threshold=0.6,
        approve_threshold=0.6,
        escalate_uncertainty=0.4,
        escalate_conflict=0.3,
    ))
    robust_threshold: float = 0.15


# ===================================================================
# Results container
# ===================================================================

@dataclass
class EFD2Results:
    """Full results of the E-FD2 experiment.

    Attributes
    ----------
    arm_results : list[ArmResult]
        Per-arm evaluation results (9 arms).
    significance_tests : list[dict]
        Pairwise McNemar's test results.
    bootstrap_cis : list[dict]
        Bootstrap CI results for key comparisons.
    metadata : dict
        Experiment metadata for reproducibility.
    """

    arm_results: list[ArmResult]
    significance_tests: list[dict]
    bootstrap_cis: list[dict]
    metadata: dict


# ===================================================================
# Key comparisons for significance testing
# ===================================================================

# Pairs: (arm_a, arm_b) — compare SLFD treatments against best baselines
_SIGNIFICANCE_PAIRS = [
    # Each SL arm vs best scalar baselines
    ("F_sl_cumulative", "C_stacking"),
    ("F_sl_cumulative", "E_noisy_or"),
    ("G_sl_three_way", "C_stacking"),
    ("G_sl_three_way", "E_noisy_or"),
    ("H_sl_robust_three_way", "C_stacking"),
    ("I_confidence_feature", "C_stacking"),
    # SL arms against each other
    ("G_sl_three_way", "F_sl_cumulative"),
    ("H_sl_robust_three_way", "G_sl_three_way"),
    ("I_confidence_feature", "F_sl_cumulative"),
]


# ===================================================================
# Core runner (from predictions)
# ===================================================================

def run_efd2_from_predictions(
    val_predictions: PredictionSet,
    val_labels: np.ndarray,
    test_predictions: PredictionSet,
    test_labels: np.ndarray,
    config: EFD2Config,
) -> EFD2Results:
    """Run E-FD2 from pre-computed model predictions.

    Parameters
    ----------
    val_predictions : PredictionSet
        Validation set predictions (for fitting meta-learners and weights).
    val_labels : np.ndarray
        True validation labels.
    test_predictions : PredictionSet
        Test set predictions (for evaluation).
    test_labels : np.ndarray
        True test labels.
    config : EFD2Config
        Experiment configuration.

    Returns
    -------
    EFD2Results
    """
    # --- Compute scores for all 9 arms ---
    arm_scores: dict[str, np.ndarray] = {}
    arm_escalation_masks: dict[str, np.ndarray] = {}

    # A: Majority vote
    arm_scores["A_majority_vote"] = majority_vote(test_predictions)

    # B: Weighted average
    acc_weights = compute_accuracy_weights(val_predictions, val_labels)
    arm_scores["B_weighted_average"] = weighted_average(test_predictions, acc_weights)

    # C: Stacking meta-learner
    stacker = StackingMetaLearner(seed=config.seed)
    stacker.fit(val_predictions, val_labels)
    arm_scores["C_stacking"] = stacker.predict(test_predictions)

    # D: Bayesian model averaging
    arm_scores["D_bayesian_model_average"] = bayesian_model_average(
        test_predictions, val_predictions, val_labels,
    )

    # E: Noisy-OR
    arm_scores["E_noisy_or"] = noisy_or(test_predictions)

    # F: SL cumulative fusion
    arm_scores["F_sl_cumulative"] = sl_cumulative_scores(
        test_predictions, base_rate=config.base_rate,
    )

    # G: SL three-way decision
    g_result = sl_three_way(
        test_predictions, base_rate=config.base_rate, decider=config.decider,
    )
    arm_scores["G_sl_three_way"] = g_result.scores
    arm_escalation_masks["G_sl_three_way"] = g_result.escalation_mask

    # H: SL robust three-way
    h_result = sl_robust_three_way(
        test_predictions, base_rate=config.base_rate,
        decider=config.decider, robust_threshold=config.robust_threshold,
    )
    arm_scores["H_sl_robust_three_way"] = h_result.scores
    arm_escalation_masks["H_sl_robust_three_way"] = h_result.escalation_mask

    # I: Confidence-as-feature meta-learner
    conf_learner = ConfidenceFeatureLearner(seed=config.seed)
    conf_learner.fit(val_predictions, val_labels)
    arm_scores["I_confidence_feature"] = conf_learner.predict(test_predictions)

    # --- Evaluate each arm ---
    arm_results: list[ArmResult] = []
    for arm_name, scores in arm_scores.items():
        esc_mask = arm_escalation_masks.get(arm_name)
        result = evaluate_arm(
            arm_name=arm_name,
            y_true=test_labels,
            scores=scores,
            escalation_mask=esc_mask,
            cost_config=config.cost_config,
        )
        arm_results.append(result)

    # --- Pairwise significance tests ---
    significance_tests = _run_significance_tests(
        arm_scores, test_labels, config,
    )

    # --- Bootstrap CIs ---
    bootstrap_cis = _run_bootstrap_cis(
        arm_scores, test_labels, config,
    )

    # --- Metadata ---
    metadata = {
        "seed": config.seed,
        "base_rate": config.base_rate,
        "n_test": len(test_labels),
        "n_val": len(val_labels),
        "n_models": test_predictions.probabilities.shape[1],
        "model_names": test_predictions.model_names,
        "n_bootstrap": config.n_bootstrap,
        "fraud_rate_test": float(np.mean(test_labels)),
        "fraud_rate_val": float(np.mean(val_labels)),
    }

    return EFD2Results(
        arm_results=arm_results,
        significance_tests=significance_tests,
        bootstrap_cis=bootstrap_cis,
        metadata=metadata,
    )


# ===================================================================
# Full pipeline runner
# ===================================================================

def run_efd2(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: EFD2Config,
) -> EFD2Results:
    """Run E-FD2 end-to-end: train models → predict → fuse → evaluate.

    Parameters
    ----------
    X_train, y_train : np.ndarray
        Training data for model fitting.
    X_val, y_val : np.ndarray
        Validation data for meta-learner fitting and weight computation.
    X_test, y_test : np.ndarray
        Test data for final evaluation.
    config : EFD2Config
        Experiment configuration.

    Returns
    -------
    EFD2Results
    """
    # 1. Train ensemble
    suite = train_model_suite(X_train, y_train, seed=config.seed)

    # 2. Get predictions
    val_preds = suite.predict_all(X_val)
    test_preds = suite.predict_all(X_test)

    # 3. Run evaluation
    results = run_efd2_from_predictions(
        val_predictions=val_preds,
        val_labels=y_val,
        test_predictions=test_preds,
        test_labels=y_test,
        config=config,
    )

    # 4. Augment metadata with training info
    results.metadata["n_train"] = len(y_train)
    results.metadata["fraud_rate_train"] = float(np.mean(y_train))

    return results


# ===================================================================
# Significance tests
# ===================================================================

def _run_significance_tests(
    arm_scores: dict[str, np.ndarray],
    test_labels: np.ndarray,
    config: EFD2Config,
) -> list[dict]:
    """Run pairwise McNemar's tests for key comparisons."""
    results = []
    for arm_a, arm_b in _SIGNIFICANCE_PAIRS:
        if arm_a not in arm_scores or arm_b not in arm_scores:
            continue
        mn = mcnemar_test(
            test_labels,
            arm_scores[arm_a],
            arm_scores[arm_b],
            threshold=0.5,
        )
        results.append({
            "arm_a": arm_a,
            "arm_b": arm_b,
            "statistic": mn.statistic,
            "p_value": mn.p_value,
            "n_discordant_a": mn.n_discordant_a,
            "n_discordant_b": mn.n_discordant_b,
        })
    return results


def _run_bootstrap_cis(
    arm_scores: dict[str, np.ndarray],
    test_labels: np.ndarray,
    config: EFD2Config,
) -> list[dict]:
    """Run bootstrap CIs on PR-AUC differences for key comparisons."""
    results = []
    for arm_a, arm_b in _SIGNIFICANCE_PAIRS:
        if arm_a not in arm_scores or arm_b not in arm_scores:
            continue
        ci = bootstrap_metric_ci(
            test_labels,
            arm_scores[arm_a],
            arm_scores[arm_b],
            metric_name="pr_auc",
            n_bootstrap=config.n_bootstrap,
            seed=config.seed,
        )
        results.append({
            "arm_a": arm_a,
            "arm_b": arm_b,
            "metric": ci.metric_name,
            "observed_diff": ci.observed_diff,
            "ci_lower": ci.ci_lower,
            "ci_upper": ci.ci_upper,
            "p_value": ci.p_value,
            "is_significant": ci.is_significant,
        })
    return results


# ===================================================================
# Serialization
# ===================================================================

def serialize_results(results: EFD2Results) -> dict:
    """Convert EFD2Results to a JSON-serializable dict.

    Parameters
    ----------
    results : EFD2Results

    Returns
    -------
    dict
        Fully JSON-serializable dictionary.
    """
    arms = []
    for arm in results.arm_results:
        arm_dict: dict = {
            "arm_name": arm.arm_name,
            "pr_auc": arm.pr_curve.auc if arm.pr_curve else None,
            "f1_max": arm.pr_curve.f1_max if arm.pr_curve else None,
            "f1_max_threshold": arm.pr_curve.f1_max_threshold if arm.pr_curve else None,
            "expected_cost": arm.expected_cost,
        }

        # FPR at TPR
        if arm.fpr_at_tpr is not None:
            arm_dict["fpr_at_tpr"] = {
                f"tpr_{t:.2f}": float(f)
                for t, f in zip(arm.fpr_at_tpr.tpr_targets, arm.fpr_at_tpr.fpr_values)
            }

        # Escalation
        if arm.escalation is not None:
            arm_dict["escalation"] = {
                "escalation_rate": arm.escalation.escalation_rate,
                "fraud_rate_escalated": _safe_float(arm.escalation.fraud_rate_escalated),
                "fraud_rate_auto_decided": _safe_float(arm.escalation.fraud_rate_auto_decided),
                "n_escalated": arm.escalation.n_escalated,
                "n_auto_decided": arm.escalation.n_auto_decided,
                "precision_auto_decided": _safe_float(arm.escalation.precision_auto_decided),
                "recall_auto_decided": _safe_float(arm.escalation.recall_auto_decided),
            }

        arms.append(arm_dict)

    return {
        "experiment": "E-FD2",
        "metadata": results.metadata,
        "arms": arms,
        "significance_tests": results.significance_tests,
        "bootstrap_cis": results.bootstrap_cis,
    }


def _safe_float(value: float) -> float | None:
    """Convert NaN to None for JSON serialization."""
    import math
    if math.isnan(value):
        return None
    return value
