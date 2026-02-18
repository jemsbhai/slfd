"""Three-way decision framework for fraud detection.

Instead of binary fraud/legitimate, a fused opinion supports three outcomes:
    - BLOCK:    strong evidence of fraud
    - APPROVE:  strong evidence of legitimacy
    - ESCALATE: insufficient or conflicting evidence → human review

Connection to Yao's Three-Way Decision Theory (2010):
    - Positive region  (b > α) → accept fraud hypothesis → block
    - Negative region  (d > β) → reject fraud hypothesis → approve
    - Boundary region  (else)  → defer → escalate

References:
    Yao, Y. (2010). Three-Way Decisions with Probabilistic Rough Sets.
    Information Sciences, 180(3), 341-353.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto

from slfd.opinion import Opinion


# ===================================================================
# Decision enum
# ===================================================================

class Decision(Enum):
    """Possible outcomes of the three-way decision framework."""

    BLOCK = auto()
    APPROVE = auto()
    ESCALATE = auto()


# ===================================================================
# Decision result (carries audit metadata)
# ===================================================================

@dataclass(frozen=True, slots=True)
class DecisionResult:
    """Outcome of a three-way decision with audit metadata.

    Parameters
    ----------
    decision : Decision
        The decision taken.
    opinion : Opinion
        The fused opinion that drove the decision.
    reason : str
        Human-readable explanation of why this decision was made.
    conflict : float or None
        Conflict metric if provided, else None.
    """

    decision: Decision
    opinion: Opinion
    reason: str
    conflict: float | None = None


# ===================================================================
# Cost matrix
# ===================================================================

def _validate_non_negative(value: float, name: str) -> None:
    if value < 0.0:
        raise ValueError(f"{name} must be ≥ 0, got {value}")


@dataclass(frozen=True, slots=True)
class CostMatrix:
    """Cost structure for fraud decision optimization.

    Parameters
    ----------
    false_positive : float
        Cost of blocking a legitimate transaction (≥ 0).
    false_negative : float
        Cost of missing a fraudulent transaction (≥ 0).
    review : float
        Cost of human review / escalation (≥ 0).
    """

    false_positive: float
    false_negative: float
    review: float

    def __post_init__(self) -> None:
        _validate_non_negative(self.false_positive, "false_positive")
        _validate_non_negative(self.false_negative, "false_negative")
        _validate_non_negative(self.review, "review")


# ===================================================================
# Three-way decider
# ===================================================================

def _validate_threshold(value: float, name: str) -> None:
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")


@dataclass(frozen=True, slots=True)
class ThreeWayDecider:
    """Decision engine with configurable thresholds.

    Decision logic (evaluated in order):
        1. If conflict > escalate_conflict → ESCALATE
        2. If u > escalate_uncertainty    → ESCALATE
        3. If b > block_threshold         → BLOCK
        4. If d > approve_threshold       → APPROVE
        5. Otherwise                      → ESCALATE

    Block is checked before approve so that in the degenerate case where
    both thresholds are exceeded, the safer action (block) wins.

    Parameters
    ----------
    block_threshold : float
        Minimum belief to auto-block, ∈ [0, 1].
    approve_threshold : float
        Minimum disbelief to auto-approve, ∈ [0, 1].
    escalate_uncertainty : float
        Minimum uncertainty to force escalation, ∈ [0, 1].
    escalate_conflict : float
        Minimum conflict to force escalation, ∈ [0, 1].
    """

    block_threshold: float
    approve_threshold: float
    escalate_uncertainty: float
    escalate_conflict: float

    def __post_init__(self) -> None:
        _validate_threshold(self.block_threshold, "block_threshold")
        _validate_threshold(self.approve_threshold, "approve_threshold")
        _validate_threshold(self.escalate_uncertainty, "escalate_uncertainty")
        _validate_threshold(self.escalate_conflict, "escalate_conflict")

    def decide(
        self,
        opinion: Opinion,
        conflict: float | None = None,
    ) -> DecisionResult:
        """Produce a three-way decision from a fused opinion.

        Parameters
        ----------
        opinion : Opinion
            The (typically fused) opinion to decide on.
        conflict : float or None
            Optional conflict metric from multi-source fusion.

        Returns
        -------
        DecisionResult
            The decision with audit metadata.
        """
        # 1. Conflict override
        if conflict is not None and conflict > self.escalate_conflict:
            return DecisionResult(
                decision=Decision.ESCALATE,
                opinion=opinion,
                reason=(
                    f"Conflict {conflict:.3f} exceeds threshold "
                    f"{self.escalate_conflict:.3f}"
                ),
                conflict=conflict,
            )

        # 2. Uncertainty override
        if opinion.u > self.escalate_uncertainty:
            return DecisionResult(
                decision=Decision.ESCALATE,
                opinion=opinion,
                reason=(
                    f"Uncertainty {opinion.u:.3f} exceeds threshold "
                    f"{self.escalate_uncertainty:.3f}"
                ),
                conflict=conflict,
            )

        # 3. Block (checked before approve — safer default)
        if opinion.b > self.block_threshold:
            return DecisionResult(
                decision=Decision.BLOCK,
                opinion=opinion,
                reason=(
                    f"Belief {opinion.b:.3f} exceeds block threshold "
                    f"{self.block_threshold:.3f}"
                ),
                conflict=conflict,
            )

        # 4. Approve
        if opinion.d > self.approve_threshold:
            return DecisionResult(
                decision=Decision.APPROVE,
                opinion=opinion,
                reason=(
                    f"Disbelief {opinion.d:.3f} exceeds approve threshold "
                    f"{self.approve_threshold:.3f}"
                ),
                conflict=conflict,
            )

        # 5. Default: escalate (boundary region)
        return DecisionResult(
            decision=Decision.ESCALATE,
            opinion=opinion,
            reason=(
                f"No threshold met: b={opinion.b:.3f}, d={opinion.d:.3f}, "
                f"u={opinion.u:.3f}"
            ),
            conflict=conflict,
        )


# ===================================================================
# Cost-sensitive threshold computation
# ===================================================================

def compute_optimal_thresholds(costs: CostMatrix) -> ThreeWayDecider:
    """Compute decision thresholds that minimize expected cost.

    Derives thresholds from the cost structure using the principle that:
    - Block when expected cost of blocking < expected cost of alternatives
    - Approve when expected cost of approving < expected cost of alternatives
    - Escalate otherwise

    The block threshold is derived as:
        θ_block = C_fp / (C_fp + C_fn)

    This is the belief level at which the expected cost of blocking
    equals the expected cost of not blocking. Lower C_fp relative to
    C_fn → lower threshold (more aggressive).

    The approve threshold mirrors this:
        θ_approve = C_fn / (C_fp + C_fn)

    The escalation uncertainty threshold reflects when review is cheaper
    than risking a wrong auto-decision:
        θ_escalate = C_review / max(C_fp, C_fn)

    Conflict threshold is set to a conservative default (0.3) as it
    depends on the fusion context rather than costs alone.

    Parameters
    ----------
    costs : CostMatrix
        The cost structure.

    Returns
    -------
    ThreeWayDecider
        A decider with cost-optimal thresholds.
    """
    total_cost = costs.false_positive + costs.false_negative

    # Avoid division by zero when both costs are zero
    if total_cost < 1e-12:
        return ThreeWayDecider(
            block_threshold=0.5,
            approve_threshold=0.5,
            escalate_uncertainty=0.5,
            escalate_conflict=0.3,
        )

    block_threshold = costs.false_positive / total_cost
    approve_threshold = costs.false_negative / total_cost

    # Escalation: when review is cheap relative to worst-case error
    max_error_cost = max(costs.false_positive, costs.false_negative)
    if max_error_cost < 1e-12:
        escalate_uncertainty = 0.5
    else:
        escalate_uncertainty = costs.review / max_error_cost

    # Clamp to valid range (0, 1) exclusive to avoid degenerate deciders
    block_threshold = max(0.01, min(0.99, block_threshold))
    approve_threshold = max(0.01, min(0.99, approve_threshold))
    escalate_uncertainty = max(0.01, min(0.99, escalate_uncertainty))

    return ThreeWayDecider(
        block_threshold=block_threshold,
        approve_threshold=approve_threshold,
        escalate_uncertainty=escalate_uncertainty,
        escalate_conflict=0.3,  # Conservative default
    )
