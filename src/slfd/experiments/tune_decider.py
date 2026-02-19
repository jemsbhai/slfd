"""Validation-based threshold tuning for the ThreeWayDecider.

Grid searches over threshold combinations and selects the one that
minimizes expected cost on the validation set. This is standard
hyperparameter optimization — the test set is never touched.

The search space covers:
    - block_threshold:       how high belief must be to auto-block
    - approve_threshold:     how high disbelief must be to auto-approve
    - escalate_uncertainty:  uncertainty level triggering escalation
    - escalate_conflict:     conflict level triggering escalation
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from slfd.decision import Decision, ThreeWayDecider
from slfd.fusion import cumulative_fuse, conflict_metric
from slfd.metrics import CostConfig, compute_expected_cost
from slfd.models.ensemble import PredictionSet
from slfd.opinion import Opinion
from slfd.trust import trust_discount


# ===================================================================
# Result container
# ===================================================================

@dataclass
class TuningResult:
    """Result of threshold tuning.

    Attributes
    ----------
    best_decider : ThreeWayDecider
        Decider with optimal thresholds.
    best_cost : float
        Expected cost per transaction at optimal thresholds.
    best_escalation_rate : float
        Escalation rate at optimal thresholds.
    n_configs_searched : int
        Total number of threshold combinations evaluated.
    all_results : list[dict]
        Per-configuration results (for analysis).
    """

    best_decider: ThreeWayDecider
    best_cost: float
    best_escalation_rate: float
    n_configs_searched: int
    all_results: list[dict]

    def to_dict(self) -> dict:
        d = self.best_decider
        return {
            "best_thresholds": {
                "block_threshold": d.block_threshold,
                "approve_threshold": d.approve_threshold,
                "escalate_uncertainty": d.escalate_uncertainty,
                "escalate_conflict": d.escalate_conflict,
            },
            "best_cost": self.best_cost,
            "best_escalation_rate": self.best_escalation_rate,
            "n_configs_searched": self.n_configs_searched,
        }


# ===================================================================
# Internal helpers
# ===================================================================

def _build_opinions_for_txn(
    pred_set: PredictionSet, i: int, base_rate: float,
) -> list[Opinion]:
    """Build per-model opinions for a single transaction."""
    n_models = pred_set.probabilities.shape[1]
    opinions: list[Opinion] = []
    for j in range(n_models):
        o = Opinion.from_confidence(
            probability=float(pred_set.probabilities[i, j]),
            uncertainty=float(pred_set.uncertainties[i, j]),
            base_rate=base_rate,
        )
        opinions.append(o)
    return opinions


def _precompute_fused_opinions(
    pred_set: PredictionSet,
    base_rate: float,
    trust_weights: np.ndarray | None = None,
) -> tuple[list[Opinion], np.ndarray]:
    """Pre-compute fused opinions and conflicts for all transactions.

    This avoids re-doing the expensive fusion for every threshold combo.

    Returns
    -------
    fused_opinions : list[Opinion]
        One fused opinion per transaction.
    conflicts : np.ndarray
        Conflict metric per transaction, shape (n,).
    """
    n_txns = pred_set.probabilities.shape[0]
    n_models = pred_set.probabilities.shape[1]

    fused_opinions: list[Opinion] = []
    conflicts = np.zeros(n_txns, dtype=np.float64)

    for i in range(n_txns):
        opinions = _build_opinions_for_txn(pred_set, i, base_rate)

        if trust_weights is not None:
            opinions = [
                trust_discount(op, float(trust_weights[j]))
                for j, op in enumerate(opinions)
            ]

        fused = cumulative_fuse(opinions)
        conf = conflict_metric(opinions)

        fused_opinions.append(fused)
        conflicts[i] = conf

    return fused_opinions, conflicts


def _evaluate_thresholds(
    fused_opinions: list[Opinion],
    conflicts: np.ndarray,
    y_true: np.ndarray,
    decider: ThreeWayDecider,
    cost_config: CostConfig,
) -> tuple[float, float]:
    """Evaluate a threshold configuration on pre-computed fused opinions.

    Returns (expected_cost, escalation_rate).
    """
    n = len(y_true)
    scores = np.zeros(n, dtype=np.float64)
    escalation_mask = np.zeros(n, dtype=bool)

    for i in range(n):
        result = decider.decide(fused_opinions[i], conflict=conflicts[i])
        scores[i] = fused_opinions[i].expected_probability
        escalation_mask[i] = result.decision == Decision.ESCALATE

    cost = compute_expected_cost(
        y_true, scores,
        threshold=0.5,
        escalation_mask=escalation_mask,
        cost_config=cost_config,
    )
    esc_rate = float(np.mean(escalation_mask))
    return cost, esc_rate


# ===================================================================
# Main tuning function
# ===================================================================

# Default grid: coarse but covers the space.
_DEFAULT_BLOCK_THRESHOLDS = [0.4, 0.5, 0.6, 0.7, 0.8]
_DEFAULT_APPROVE_THRESHOLDS = [0.4, 0.5, 0.6, 0.7, 0.8]
_DEFAULT_ESC_UNCERTAINTY = [0.1, 0.2, 0.3, 0.4, 0.5]
_DEFAULT_ESC_CONFLICT = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


def tune_decider_thresholds(
    val_pred_set: PredictionSet,
    val_labels: np.ndarray,
    base_rate: float,
    cost_config: CostConfig,
    trust_weights: np.ndarray | None = None,
    block_thresholds: list[float] | None = None,
    approve_thresholds: list[float] | None = None,
    esc_uncertainty_thresholds: list[float] | None = None,
    esc_conflict_thresholds: list[float] | None = None,
    max_escalation_rate: float = 0.50,
) -> TuningResult:
    """Grid search for optimal ThreeWayDecider thresholds.

    Searches over all combinations of threshold parameters and selects
    the configuration that minimizes expected cost on the validation set,
    subject to the constraint that escalation rate <= max_escalation_rate.

    Parameters
    ----------
    val_pred_set : PredictionSet
        Validation predictions.
    val_labels : np.ndarray
        True validation labels (0/1).
    base_rate : float
        Prior fraud rate for opinion construction.
    cost_config : CostConfig
        Cost parameters.
    trust_weights : np.ndarray or None
        If provided, apply trust discount before fusion.
    block_thresholds : list[float] or None
        Grid for block threshold.
    approve_thresholds : list[float] or None
        Grid for approve threshold.
    esc_uncertainty_thresholds : list[float] or None
        Grid for escalation uncertainty threshold.
    esc_conflict_thresholds : list[float] or None
        Grid for escalation conflict threshold.
    max_escalation_rate : float
        Maximum allowed escalation rate (default: 50%).

    Returns
    -------
    TuningResult
    """
    if block_thresholds is None:
        block_thresholds = _DEFAULT_BLOCK_THRESHOLDS
    if approve_thresholds is None:
        approve_thresholds = _DEFAULT_APPROVE_THRESHOLDS
    if esc_uncertainty_thresholds is None:
        esc_uncertainty_thresholds = _DEFAULT_ESC_UNCERTAINTY
    if esc_conflict_thresholds is None:
        esc_conflict_thresholds = _DEFAULT_ESC_CONFLICT

    # Pre-compute fused opinions once (expensive step)
    fused_opinions, conflicts = _precompute_fused_opinions(
        val_pred_set, base_rate, trust_weights,
    )

    best_cost = float("inf")
    best_esc_rate = 1.0
    best_decider = None
    all_results: list[dict] = []

    configs = list(product(
        block_thresholds,
        approve_thresholds,
        esc_uncertainty_thresholds,
        esc_conflict_thresholds,
    ))

    for bt, at, eu, ec in configs:
        decider = ThreeWayDecider(
            block_threshold=bt,
            approve_threshold=at,
            escalate_uncertainty=eu,
            escalate_conflict=ec,
        )

        cost, esc_rate = _evaluate_thresholds(
            fused_opinions, conflicts, val_labels, decider, cost_config,
        )

        all_results.append({
            "block": bt,
            "approve": at,
            "esc_u": eu,
            "esc_c": ec,
            "cost": cost,
            "escalation_rate": esc_rate,
        })

        # Only consider configs within the escalation budget
        if esc_rate <= max_escalation_rate and cost < best_cost:
            best_cost = cost
            best_esc_rate = esc_rate
            best_decider = decider

    # If no config met the escalation constraint, pick the one with
    # the lowest escalation rate (fallback — should not happen with
    # reasonable grids)
    if best_decider is None:
        all_results.sort(key=lambda r: r["escalation_rate"])
        fb = all_results[0]
        best_decider = ThreeWayDecider(
            block_threshold=fb["block"],
            approve_threshold=fb["approve"],
            escalate_uncertainty=fb["esc_u"],
            escalate_conflict=fb["esc_c"],
        )
        best_cost = fb["cost"]
        best_esc_rate = fb["escalation_rate"]

    return TuningResult(
        best_decider=best_decider,
        best_cost=best_cost,
        best_escalation_rate=best_esc_rate,
        n_configs_searched=len(configs),
        all_results=all_results,
    )
