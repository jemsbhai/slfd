"""Tests for the three-way decision framework.

Instead of binary fraud/legitimate, the fused opinion supports three outcomes:
    - BLOCK:    strong evidence of fraud (b > θ_block)
    - APPROVE:  strong evidence of legitimacy (d > θ_approve)
    - ESCALATE: insufficient or conflicting evidence (otherwise)

The escalation pathway is where the real operational value lies — routing
uncertain/conflicting cases to humans reduces both false positives AND
false negatives.

Connection to Yao's Three-Way Decision Theory (2010):
    - Positive region  (b > α) → accept fraud hypothesis → block
    - Negative region  (d > β) → reject fraud hypothesis → approve
    - Boundary region  (else)  → defer → escalate

References:
    Yao, Y. (2010). Three-Way Decisions with Probabilistic Rough Sets.
    Information Sciences, 180(3), 341-353.
"""

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from slfd.opinion import Opinion
from slfd.decision import (
    Decision,
    DecisionResult,
    ThreeWayDecider,
    CostMatrix,
    compute_optimal_thresholds,
)


# ===================================================================
# 1. Decision enum
# ===================================================================

class TestDecisionEnum:
    """The three possible outcomes."""

    def test_three_values_exist(self):
        assert Decision.BLOCK is not None
        assert Decision.APPROVE is not None
        assert Decision.ESCALATE is not None

    def test_distinct(self):
        assert Decision.BLOCK != Decision.APPROVE
        assert Decision.BLOCK != Decision.ESCALATE
        assert Decision.APPROVE != Decision.ESCALATE


# ===================================================================
# 2. ThreeWayDecider with explicit thresholds
# ===================================================================

class TestThreeWayDecider:
    """Decision engine with configurable thresholds."""

    def setup_method(self):
        self.decider = ThreeWayDecider(
            block_threshold=0.7,
            approve_threshold=0.7,
            escalate_uncertainty=0.5,
            escalate_conflict=0.3,
        )

    def test_strong_belief_blocks(self):
        """High belief → BLOCK."""
        o = Opinion(b=0.8, d=0.1, u=0.1, a=0.5)
        result = self.decider.decide(o)
        assert result.decision == Decision.BLOCK

    def test_strong_disbelief_approves(self):
        """High disbelief → APPROVE."""
        o = Opinion(b=0.1, d=0.8, u=0.1, a=0.5)
        result = self.decider.decide(o)
        assert result.decision == Decision.APPROVE

    def test_high_uncertainty_escalates(self):
        """High uncertainty → ESCALATE."""
        o = Opinion(b=0.2, d=0.2, u=0.6, a=0.5)
        result = self.decider.decide(o)
        assert result.decision == Decision.ESCALATE

    def test_moderate_everything_escalates(self):
        """Nothing decisive → ESCALATE."""
        o = Opinion(b=0.4, d=0.3, u=0.3, a=0.5)
        result = self.decider.decide(o)
        assert result.decision == Decision.ESCALATE

    def test_conflict_triggers_escalation(self):
        """High conflict score → ESCALATE regardless of belief."""
        o = Opinion(b=0.6, d=0.2, u=0.2, a=0.5)
        result = self.decider.decide(o, conflict=0.5)
        assert result.decision == Decision.ESCALATE

    def test_low_conflict_no_override(self):
        """Low conflict → decision based on opinion alone."""
        o = Opinion(b=0.8, d=0.1, u=0.1, a=0.5)
        result = self.decider.decide(o, conflict=0.1)
        assert result.decision == Decision.BLOCK

    def test_block_takes_priority_over_approve(self):
        """If both b and d exceed thresholds (edge case), block wins."""
        # This shouldn't happen with valid opinions since b+d≤1,
        # but with very low thresholds it could
        decider = ThreeWayDecider(
            block_threshold=0.3,
            approve_threshold=0.3,
            escalate_uncertainty=0.5,
            escalate_conflict=0.3,
        )
        o = Opinion(b=0.5, d=0.4, u=0.1, a=0.5)
        result = decider.decide(o)
        assert result.decision == Decision.BLOCK


# ===================================================================
# 3. DecisionResult metadata
# ===================================================================

class TestDecisionResult:
    """Decision results carry metadata for auditability."""

    def setup_method(self):
        self.decider = ThreeWayDecider(
            block_threshold=0.7,
            approve_threshold=0.7,
            escalate_uncertainty=0.5,
            escalate_conflict=0.3,
        )

    def test_result_contains_opinion(self):
        o = Opinion(b=0.8, d=0.1, u=0.1, a=0.5)
        result = self.decider.decide(o)
        assert result.opinion is o

    def test_result_contains_decision(self):
        o = Opinion(b=0.8, d=0.1, u=0.1, a=0.5)
        result = self.decider.decide(o)
        assert isinstance(result.decision, Decision)

    def test_result_contains_reason(self):
        """Reason string explains why this decision was made."""
        o = Opinion(b=0.8, d=0.1, u=0.1, a=0.5)
        result = self.decider.decide(o)
        assert isinstance(result.reason, str)
        assert len(result.reason) > 0

    def test_escalation_reason_mentions_uncertainty(self):
        o = Opinion(b=0.2, d=0.2, u=0.6, a=0.5)
        result = self.decider.decide(o)
        assert result.decision == Decision.ESCALATE
        assert "uncertainty" in result.reason.lower()

    def test_escalation_reason_mentions_conflict(self):
        o = Opinion(b=0.6, d=0.2, u=0.2, a=0.5)
        result = self.decider.decide(o, conflict=0.5)
        assert result.decision == Decision.ESCALATE
        assert "conflict" in result.reason.lower()

    def test_result_contains_conflict(self):
        o = Opinion(b=0.8, d=0.1, u=0.1, a=0.5)
        result = self.decider.decide(o, conflict=0.15)
        assert result.conflict == pytest.approx(0.15)

    def test_result_conflict_none_when_not_provided(self):
        o = Opinion(b=0.8, d=0.1, u=0.1, a=0.5)
        result = self.decider.decide(o)
        assert result.conflict is None


# ===================================================================
# 4. CostMatrix and cost-sensitive thresholds
# ===================================================================

class TestCostMatrix:
    """Cost structure for decision optimization."""

    def test_create_cost_matrix(self):
        costs = CostMatrix(
            false_positive=0.50,   # block legitimate
            false_negative=50.0,   # miss fraud
            review=2.0,            # human review
        )
        assert costs.false_positive == 0.50
        assert costs.false_negative == 50.0
        assert costs.review == 2.0

    def test_all_costs_must_be_non_negative(self):
        with pytest.raises(ValueError):
            CostMatrix(false_positive=-1.0, false_negative=50.0, review=2.0)

    def test_fn_cost_negative_rejected(self):
        with pytest.raises(ValueError):
            CostMatrix(false_positive=0.50, false_negative=-50.0, review=2.0)

    def test_review_cost_negative_rejected(self):
        with pytest.raises(ValueError):
            CostMatrix(false_positive=0.50, false_negative=50.0, review=-2.0)


class TestOptimalThresholds:
    """Cost-sensitive threshold computation.

    Given costs, compute the thresholds that minimize expected cost.
    When fraud is very expensive to miss (high C_fn), block threshold
    should be lower (more aggressive blocking).
    """

    def test_high_fn_cost_lowers_block_threshold(self):
        """Expensive missed fraud → more aggressive blocking."""
        conservative = compute_optimal_thresholds(
            CostMatrix(false_positive=0.50, false_negative=10.0, review=2.0)
        )
        aggressive = compute_optimal_thresholds(
            CostMatrix(false_positive=0.50, false_negative=100.0, review=2.0)
        )
        assert aggressive.block_threshold < conservative.block_threshold

    def test_high_fp_cost_raises_block_threshold(self):
        """Expensive false positives → more conservative blocking."""
        cheap_fp = compute_optimal_thresholds(
            CostMatrix(false_positive=0.10, false_negative=50.0, review=2.0)
        )
        expensive_fp = compute_optimal_thresholds(
            CostMatrix(false_positive=10.0, false_negative=50.0, review=2.0)
        )
        assert expensive_fp.block_threshold > cheap_fp.block_threshold

    def test_cheap_review_widens_escalation(self):
        """Cheap human review → more escalation (lower uncertainty threshold)."""
        cheap_review = compute_optimal_thresholds(
            CostMatrix(false_positive=0.50, false_negative=50.0, review=0.10)
        )
        expensive_review = compute_optimal_thresholds(
            CostMatrix(false_positive=0.50, false_negative=50.0, review=20.0)
        )
        assert cheap_review.escalate_uncertainty < expensive_review.escalate_uncertainty

    def test_thresholds_in_valid_range(self):
        """All computed thresholds are in (0, 1)."""
        thresholds = compute_optimal_thresholds(
            CostMatrix(false_positive=0.50, false_negative=50.0, review=2.0)
        )
        assert 0.0 < thresholds.block_threshold < 1.0
        assert 0.0 < thresholds.approve_threshold < 1.0
        assert 0.0 < thresholds.escalate_uncertainty < 1.0


# ===================================================================
# 5. Edge cases
# ===================================================================

class TestDecisionEdgeCases:
    """Boundary and degenerate cases."""

    def test_vacuous_opinion_escalates(self):
        """Complete ignorance → always escalate."""
        decider = ThreeWayDecider(
            block_threshold=0.5,
            approve_threshold=0.5,
            escalate_uncertainty=0.5,
            escalate_conflict=0.3,
        )
        v = Opinion(b=0.0, d=0.0, u=1.0, a=0.5)
        result = decider.decide(v)
        assert result.decision == Decision.ESCALATE

    def test_dogmatic_belief_blocks(self):
        """Full belief → block."""
        decider = ThreeWayDecider(
            block_threshold=0.5,
            approve_threshold=0.5,
            escalate_uncertainty=0.5,
            escalate_conflict=0.3,
        )
        o = Opinion(b=1.0, d=0.0, u=0.0, a=0.5)
        result = decider.decide(o)
        assert result.decision == Decision.BLOCK

    def test_dogmatic_disbelief_approves(self):
        """Full disbelief → approve."""
        decider = ThreeWayDecider(
            block_threshold=0.5,
            approve_threshold=0.5,
            escalate_uncertainty=0.5,
            escalate_conflict=0.3,
        )
        o = Opinion(b=0.0, d=1.0, u=0.0, a=0.5)
        result = decider.decide(o)
        assert result.decision == Decision.APPROVE

    def test_threshold_validation_block(self):
        with pytest.raises(ValueError):
            ThreeWayDecider(
                block_threshold=1.5,
                approve_threshold=0.7,
                escalate_uncertainty=0.5,
                escalate_conflict=0.3,
            )

    def test_threshold_validation_approve(self):
        with pytest.raises(ValueError):
            ThreeWayDecider(
                block_threshold=0.7,
                approve_threshold=-0.1,
                escalate_uncertainty=0.5,
                escalate_conflict=0.3,
            )

    def test_threshold_validation_uncertainty(self):
        with pytest.raises(ValueError):
            ThreeWayDecider(
                block_threshold=0.7,
                approve_threshold=0.7,
                escalate_uncertainty=1.5,
                escalate_conflict=0.3,
            )

    def test_threshold_validation_conflict(self):
        with pytest.raises(ValueError):
            ThreeWayDecider(
                block_threshold=0.7,
                approve_threshold=0.7,
                escalate_uncertainty=0.5,
                escalate_conflict=-0.1,
            )
