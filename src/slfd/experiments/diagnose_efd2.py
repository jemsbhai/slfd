"""E-FD2 diagnostic analysis — understand WHY the first run produced its results.

This module provides structured analysis of:
    1. Per-model opinion distributions (what do b, d, u look like?)
    2. Fused opinion distributions (what happens after cumulative_fuse?)
    3. Decider threshold analysis (why does G escalate 98.9%?)
    4. Per-model individual performance (accuracy, ROC-AUC, PR-AUC)

These diagnostics guide the E-FD2 redesign by identifying root causes
of the observed results rather than guessing at fixes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
)

from slfd.opinion import Opinion
from slfd.fusion import cumulative_fuse, conflict_metric
from slfd.decision import Decision, ThreeWayDecider
from slfd.models.ensemble import PredictionSet


# ===================================================================
# Data classes for diagnostic results
# ===================================================================

@dataclass
class OpinionDistStats:
    """Per-model opinion distribution statistics.

    Captures the shape of the opinion space for a single model
    across all transactions.
    """

    model_name: str
    n_transactions: int

    # Belief
    b_mean: float
    b_std: float
    b_min: float
    b_max: float

    # Disbelief
    d_mean: float
    d_std: float
    d_min: float
    d_max: float

    # Uncertainty
    u_mean: float
    u_std: float
    u_min: float
    u_max: float
    u_p25: float
    u_p50: float
    u_p75: float

    # Raw probability stats (for reference)
    prob_mean: float
    prob_std: float

    # Raw uncertainty stats
    raw_uncert_mean: float
    raw_uncert_std: float


@dataclass
class FusedOpinionStats:
    """Fused opinion distribution statistics after cumulative fusion."""

    n_transactions: int

    # Fused belief
    b_mean: float
    b_std: float
    b_min: float
    b_max: float
    b_p25: float
    b_p50: float
    b_p75: float

    # Fused disbelief
    d_mean: float
    d_std: float
    d_min: float
    d_max: float
    d_p25: float
    d_p50: float
    d_p75: float

    # Fused uncertainty
    u_mean: float
    u_std: float
    u_min: float
    u_max: float
    u_p25: float
    u_p50: float
    u_p75: float

    # Conflict metric
    conflict_mean: float
    conflict_std: float
    conflict_min: float
    conflict_max: float
    conflict_p25: float
    conflict_p50: float
    conflict_p75: float

    # Expected probability
    expected_prob_mean: float
    expected_prob_std: float
    expected_prob_min: float
    expected_prob_max: float


@dataclass
class DeciderDiagnostics:
    """Analysis of how the ThreeWayDecider interacts with fused opinions."""

    # Decision fractions
    frac_block: float
    frac_approve: float
    frac_escalate: float

    # Escalation reason breakdown
    escalate_by_conflict: float
    escalate_by_uncertainty: float
    escalate_by_default: float

    # Threshold exceedance fractions (how many opinions cross each threshold?)
    frac_b_above_block: float
    frac_d_above_approve: float
    frac_u_above_escalate: float
    frac_conflict_above_threshold: float

    # Decider configuration (for reference)
    block_threshold: float
    approve_threshold: float
    escalate_uncertainty: float
    escalate_conflict: float


@dataclass
class ModelPerformanceStats:
    """Individual model performance metrics."""

    model_name: str
    accuracy: float
    roc_auc: float
    pr_auc: float
    prob_mean: float
    prob_std: float
    prob_mean_fraud: float
    prob_mean_legit: float


@dataclass
class EFD2Diagnostics:
    """Full diagnostic report for E-FD2."""

    opinion_distributions: list[OpinionDistStats]
    fused_opinions: FusedOpinionStats
    decider_diagnostics: DeciderDiagnostics
    model_performance: list[ModelPerformanceStats]

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dict."""
        return {
            "opinion_distributions": [
                _opinion_dist_to_dict(s) for s in self.opinion_distributions
            ],
            "fused_opinions": _fused_stats_to_dict(self.fused_opinions),
            "decider_diagnostics": _decider_diag_to_dict(self.decider_diagnostics),
            "model_performance": [
                _model_perf_to_dict(s) for s in self.model_performance
            ],
        }


# ===================================================================
# Diagnostic functions
# ===================================================================

def diagnose_opinion_distributions(
    pred_set: PredictionSet,
    base_rate: float,
) -> list[OpinionDistStats]:
    """Analyze per-model opinion distributions.

    For each model, converts all predictions to Opinion objects and
    computes distributional statistics on b, d, u components.

    Parameters
    ----------
    pred_set : PredictionSet
        Multi-model predictions.
    base_rate : float
        Prior fraud rate for opinion construction.

    Returns
    -------
    list[OpinionDistStats]
        One entry per model.
    """
    n_txns, n_models = pred_set.probabilities.shape
    results: list[OpinionDistStats] = []

    for j in range(n_models):
        probs = pred_set.probabilities[:, j]
        uncerts = pred_set.uncertainties[:, j]

        # Compute opinion components vectorized
        evidence_mass = 1.0 - uncerts
        b_vals = probs * evidence_mass
        d_vals = (1.0 - probs) * evidence_mass
        u_vals = uncerts

        results.append(OpinionDistStats(
            model_name=pred_set.model_names[j],
            n_transactions=n_txns,
            b_mean=float(np.mean(b_vals)),
            b_std=float(np.std(b_vals)),
            b_min=float(np.min(b_vals)),
            b_max=float(np.max(b_vals)),
            d_mean=float(np.mean(d_vals)),
            d_std=float(np.std(d_vals)),
            d_min=float(np.min(d_vals)),
            d_max=float(np.max(d_vals)),
            u_mean=float(np.mean(u_vals)),
            u_std=float(np.std(u_vals)),
            u_min=float(np.min(u_vals)),
            u_max=float(np.max(u_vals)),
            u_p25=float(np.percentile(u_vals, 25)),
            u_p50=float(np.percentile(u_vals, 50)),
            u_p75=float(np.percentile(u_vals, 75)),
            prob_mean=float(np.mean(probs)),
            prob_std=float(np.std(probs)),
            raw_uncert_mean=float(np.mean(uncerts)),
            raw_uncert_std=float(np.std(uncerts)),
        ))

    return results


def diagnose_fused_opinions(
    pred_set: PredictionSet,
    base_rate: float,
) -> FusedOpinionStats:
    """Analyze fused opinion distributions after cumulative fusion.

    Parameters
    ----------
    pred_set : PredictionSet
        Multi-model predictions.
    base_rate : float
        Prior fraud rate for opinion construction.

    Returns
    -------
    FusedOpinionStats
    """
    n_txns = pred_set.probabilities.shape[0]
    b_vals = np.zeros(n_txns)
    d_vals = np.zeros(n_txns)
    u_vals = np.zeros(n_txns)
    conflict_vals = np.zeros(n_txns)
    expected_probs = np.zeros(n_txns)

    for i in range(n_txns):
        opinions = _build_opinions(pred_set, i, base_rate)
        fused = cumulative_fuse(opinions)
        conf = conflict_metric(opinions)

        b_vals[i] = fused.b
        d_vals[i] = fused.d
        u_vals[i] = fused.u
        conflict_vals[i] = conf
        expected_probs[i] = fused.expected_probability

    return FusedOpinionStats(
        n_transactions=n_txns,
        b_mean=float(np.mean(b_vals)),
        b_std=float(np.std(b_vals)),
        b_min=float(np.min(b_vals)),
        b_max=float(np.max(b_vals)),
        b_p25=float(np.percentile(b_vals, 25)),
        b_p50=float(np.percentile(b_vals, 50)),
        b_p75=float(np.percentile(b_vals, 75)),
        d_mean=float(np.mean(d_vals)),
        d_std=float(np.std(d_vals)),
        d_min=float(np.min(d_vals)),
        d_max=float(np.max(d_vals)),
        d_p25=float(np.percentile(d_vals, 25)),
        d_p50=float(np.percentile(d_vals, 50)),
        d_p75=float(np.percentile(d_vals, 75)),
        u_mean=float(np.mean(u_vals)),
        u_std=float(np.std(u_vals)),
        u_min=float(np.min(u_vals)),
        u_max=float(np.max(u_vals)),
        u_p25=float(np.percentile(u_vals, 25)),
        u_p50=float(np.percentile(u_vals, 50)),
        u_p75=float(np.percentile(u_vals, 75)),
        conflict_mean=float(np.mean(conflict_vals)),
        conflict_std=float(np.std(conflict_vals)),
        conflict_min=float(np.min(conflict_vals)),
        conflict_max=float(np.max(conflict_vals)),
        conflict_p25=float(np.percentile(conflict_vals, 25)),
        conflict_p50=float(np.percentile(conflict_vals, 50)),
        conflict_p75=float(np.percentile(conflict_vals, 75)),
        expected_prob_mean=float(np.mean(expected_probs)),
        expected_prob_std=float(np.std(expected_probs)),
        expected_prob_min=float(np.min(expected_probs)),
        expected_prob_max=float(np.max(expected_probs)),
    )


def diagnose_decider(
    pred_set: PredictionSet,
    base_rate: float,
    decider: ThreeWayDecider,
) -> DeciderDiagnostics:
    """Analyze how the ThreeWayDecider interacts with the actual opinion distribution.

    This diagnoses WHY Arm G escalated 98.9% — by counting how many
    fused opinions cross each threshold and tracking escalation reasons.

    Parameters
    ----------
    pred_set : PredictionSet
        Multi-model predictions.
    base_rate : float
        Prior fraud rate for opinion construction.
    decider : ThreeWayDecider
        The decision engine to diagnose.

    Returns
    -------
    DeciderDiagnostics
    """
    n_txns = pred_set.probabilities.shape[0]

    n_block = 0
    n_approve = 0
    n_escalate = 0
    n_esc_conflict = 0
    n_esc_uncertainty = 0
    n_esc_default = 0

    # Track fused opinion components for threshold analysis
    b_vals = np.zeros(n_txns)
    d_vals = np.zeros(n_txns)
    u_vals = np.zeros(n_txns)
    conflict_vals = np.zeros(n_txns)

    for i in range(n_txns):
        opinions = _build_opinions(pred_set, i, base_rate)
        fused = cumulative_fuse(opinions)
        conf = conflict_metric(opinions)

        b_vals[i] = fused.b
        d_vals[i] = fused.d
        u_vals[i] = fused.u
        conflict_vals[i] = conf

        result = decider.decide(fused, conflict=conf)

        if result.decision == Decision.BLOCK:
            n_block += 1
        elif result.decision == Decision.APPROVE:
            n_approve += 1
        else:  # ESCALATE
            n_escalate += 1
            # Determine reason from the decision logic order
            if conf > decider.escalate_conflict:
                n_esc_conflict += 1
            elif fused.u > decider.escalate_uncertainty:
                n_esc_uncertainty += 1
            else:
                n_esc_default += 1

    total = float(n_txns)

    return DeciderDiagnostics(
        frac_block=n_block / total,
        frac_approve=n_approve / total,
        frac_escalate=n_escalate / total,
        escalate_by_conflict=n_esc_conflict / total,
        escalate_by_uncertainty=n_esc_uncertainty / total,
        escalate_by_default=n_esc_default / total,
        frac_b_above_block=float(np.mean(b_vals > decider.block_threshold)),
        frac_d_above_approve=float(np.mean(d_vals > decider.approve_threshold)),
        frac_u_above_escalate=float(np.mean(u_vals > decider.escalate_uncertainty)),
        frac_conflict_above_threshold=float(
            np.mean(conflict_vals > decider.escalate_conflict)
        ),
        block_threshold=decider.block_threshold,
        approve_threshold=decider.approve_threshold,
        escalate_uncertainty=decider.escalate_uncertainty,
        escalate_conflict=decider.escalate_conflict,
    )


def diagnose_model_performance(
    pred_set: PredictionSet,
    labels: np.ndarray,
) -> list[ModelPerformanceStats]:
    """Analyze individual model performance.

    Parameters
    ----------
    pred_set : PredictionSet
        Multi-model predictions.
    labels : np.ndarray
        True labels (0/1).

    Returns
    -------
    list[ModelPerformanceStats]
        One entry per model.
    """
    n_models = pred_set.probabilities.shape[1]
    fraud_mask = labels == 1
    legit_mask = labels == 0
    results: list[ModelPerformanceStats] = []

    for j in range(n_models):
        probs = pred_set.probabilities[:, j]
        preds = (probs > 0.5).astype(int)

        acc = float(accuracy_score(labels, preds))
        roc = float(roc_auc_score(labels, probs))
        pr = float(average_precision_score(labels, probs))

        prob_fraud = float(np.mean(probs[fraud_mask])) if np.any(fraud_mask) else 0.0
        prob_legit = float(np.mean(probs[legit_mask])) if np.any(legit_mask) else 0.0

        results.append(ModelPerformanceStats(
            model_name=pred_set.model_names[j],
            accuracy=acc,
            roc_auc=roc,
            pr_auc=pr,
            prob_mean=float(np.mean(probs)),
            prob_std=float(np.std(probs)),
            prob_mean_fraud=prob_fraud,
            prob_mean_legit=prob_legit,
        ))

    return results


# ===================================================================
# Unified diagnostic runner
# ===================================================================

def run_full_diagnostics(
    pred_set: PredictionSet,
    labels: np.ndarray,
    base_rate: float,
    decider: ThreeWayDecider,
) -> EFD2Diagnostics:
    """Run all diagnostic analyses.

    Parameters
    ----------
    pred_set : PredictionSet
        Multi-model predictions.
    labels : np.ndarray
        True labels (0/1).
    base_rate : float
        Prior fraud rate for opinion construction.
    decider : ThreeWayDecider
        Decision engine to diagnose.

    Returns
    -------
    EFD2Diagnostics
        Full diagnostic report.
    """
    return EFD2Diagnostics(
        opinion_distributions=diagnose_opinion_distributions(pred_set, base_rate),
        fused_opinions=diagnose_fused_opinions(pred_set, base_rate),
        decider_diagnostics=diagnose_decider(pred_set, base_rate, decider),
        model_performance=diagnose_model_performance(pred_set, labels),
    )


# ===================================================================
# Internal helpers
# ===================================================================

def _build_opinions(
    pred_set: PredictionSet,
    txn_idx: int,
    base_rate: float,
) -> list[Opinion]:
    """Build per-source opinions for a single transaction."""
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


# ===================================================================
# Serialization helpers
# ===================================================================

def _opinion_dist_to_dict(s: OpinionDistStats) -> dict:
    return {
        "model_name": s.model_name,
        "n_transactions": s.n_transactions,
        "belief": {"mean": s.b_mean, "std": s.b_std, "min": s.b_min, "max": s.b_max},
        "disbelief": {"mean": s.d_mean, "std": s.d_std, "min": s.d_min, "max": s.d_max},
        "uncertainty": {
            "mean": s.u_mean, "std": s.u_std,
            "min": s.u_min, "max": s.u_max,
            "p25": s.u_p25, "p50": s.u_p50, "p75": s.u_p75,
        },
        "probability": {"mean": s.prob_mean, "std": s.prob_std},
        "raw_uncertainty": {"mean": s.raw_uncert_mean, "std": s.raw_uncert_std},
    }


def _fused_stats_to_dict(s: FusedOpinionStats) -> dict:
    return {
        "n_transactions": s.n_transactions,
        "belief": {
            "mean": s.b_mean, "std": s.b_std,
            "min": s.b_min, "max": s.b_max,
            "p25": s.b_p25, "p50": s.b_p50, "p75": s.b_p75,
        },
        "disbelief": {
            "mean": s.d_mean, "std": s.d_std,
            "min": s.d_min, "max": s.d_max,
            "p25": s.d_p25, "p50": s.d_p50, "p75": s.d_p75,
        },
        "uncertainty": {
            "mean": s.u_mean, "std": s.u_std,
            "min": s.u_min, "max": s.u_max,
            "p25": s.u_p25, "p50": s.u_p50, "p75": s.u_p75,
        },
        "conflict": {
            "mean": s.conflict_mean, "std": s.conflict_std,
            "min": s.conflict_min, "max": s.conflict_max,
            "p25": s.conflict_p25, "p50": s.conflict_p50, "p75": s.conflict_p75,
        },
        "expected_probability": {
            "mean": s.expected_prob_mean, "std": s.expected_prob_std,
            "min": s.expected_prob_min, "max": s.expected_prob_max,
        },
    }


def _decider_diag_to_dict(s: DeciderDiagnostics) -> dict:
    return {
        "decision_fractions": {
            "block": s.frac_block,
            "approve": s.frac_approve,
            "escalate": s.frac_escalate,
        },
        "escalation_reasons": {
            "conflict": s.escalate_by_conflict,
            "uncertainty": s.escalate_by_uncertainty,
            "default": s.escalate_by_default,
        },
        "threshold_exceedance": {
            "b_above_block": s.frac_b_above_block,
            "d_above_approve": s.frac_d_above_approve,
            "u_above_escalate": s.frac_u_above_escalate,
            "conflict_above_threshold": s.frac_conflict_above_threshold,
        },
        "thresholds": {
            "block": s.block_threshold,
            "approve": s.approve_threshold,
            "escalate_uncertainty": s.escalate_uncertainty,
            "escalate_conflict": s.escalate_conflict,
        },
    }


def _model_perf_to_dict(s: ModelPerformanceStats) -> dict:
    return {
        "model_name": s.model_name,
        "accuracy": s.accuracy,
        "roc_auc": s.roc_auc,
        "pr_auc": s.pr_auc,
        "probability": {
            "mean": s.prob_mean,
            "std": s.prob_std,
            "mean_fraud": s.prob_mean_fraud,
            "mean_legit": s.prob_mean_legit,
        },
    }
