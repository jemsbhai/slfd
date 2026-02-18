"""Tests for the Opinion dataclass — the fundamental SL building block.

An Opinion ω = (b, d, u, a) represents an epistemic state where:
    b = belief (evidence FOR the proposition)
    d = disbelief (evidence AGAINST the proposition)
    u = uncertainty (lack of evidence)
    a = base rate (prior probability absent evidence)

Constraint: b + d + u = 1, all components ∈ [0, 1], a ∈ [0, 1]
"""

import math

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from slfd.opinion import Opinion


# ---------------------------------------------------------------------------
# Strategy: generate valid SL opinion components
# ---------------------------------------------------------------------------
def _opinion_components():
    """Hypothesis strategy producing valid (b, d, u, a) tuples."""
    return (
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
        .flatmap(lambda b:
            st.floats(min_value=0.0, max_value=1.0 - b, allow_nan=False, allow_infinity=False)
            .flatmap(lambda d:
                st.just((b, d, 1.0 - b - d))
            )
        )
        .flatmap(lambda bdu:
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
            .map(lambda a: (*bdu, a))
        )
    )


# ===================================================================
# 1. Construction & validation
# ===================================================================

class TestOpinionConstruction:
    """Basic creation and constraint enforcement."""

    def test_create_valid_opinion(self):
        """Standard opinion with valid components."""
        o = Opinion(b=0.6, d=0.2, u=0.2, a=0.5)
        assert o.b == 0.6
        assert o.d == 0.2
        assert o.u == 0.2
        assert o.a == 0.5

    def test_components_must_sum_to_one(self):
        """b + d + u must equal 1 (within tolerance)."""
        with pytest.raises(ValueError, match="sum to 1"):
            Opinion(b=0.5, d=0.5, u=0.5, a=0.5)

    def test_negative_belief_rejected(self):
        """Negative components are invalid."""
        with pytest.raises(ValueError):
            Opinion(b=-0.1, d=0.6, u=0.5, a=0.5)

    def test_negative_disbelief_rejected(self):
        with pytest.raises(ValueError):
            Opinion(b=0.6, d=-0.1, u=0.5, a=0.5)

    def test_negative_uncertainty_rejected(self):
        with pytest.raises(ValueError):
            Opinion(b=0.5, d=0.6, u=-0.1, a=0.5)

    def test_base_rate_below_zero_rejected(self):
        with pytest.raises(ValueError):
            Opinion(b=0.5, d=0.3, u=0.2, a=-0.1)

    def test_base_rate_above_one_rejected(self):
        with pytest.raises(ValueError):
            Opinion(b=0.5, d=0.3, u=0.2, a=1.1)

    def test_vacuous_opinion(self):
        """Complete ignorance: no evidence, full uncertainty."""
        o = Opinion(b=0.0, d=0.0, u=1.0, a=0.5)
        assert o.u == 1.0

    def test_dogmatic_belief(self):
        """Full belief, no uncertainty."""
        o = Opinion(b=1.0, d=0.0, u=0.0, a=0.5)
        assert o.b == 1.0
        assert o.u == 0.0

    def test_dogmatic_disbelief(self):
        """Full disbelief, no uncertainty."""
        o = Opinion(b=0.0, d=1.0, u=0.0, a=0.5)
        assert o.d == 1.0

    def test_floating_point_tolerance(self):
        """Components that sum to ~1.0 within float tolerance are accepted."""
        # These sum to 1.0000000000000002 due to float arithmetic
        o = Opinion(b=0.1, d=0.2, u=0.7, a=0.5)
        assert o is not None

    @given(_opinion_components())
    @settings(max_examples=200)
    def test_valid_components_always_accepted(self, components):
        """Property: any (b, d, u, a) where b+d+u≈1 and all ∈ [0,1] is valid."""
        b, d, u, a = components
        assume(abs(b + d + u - 1.0) < 1e-9)
        assume(u >= 0.0)
        o = Opinion(b=b, d=d, u=u, a=a)
        assert abs(o.b + o.d + o.u - 1.0) < 1e-6


# ===================================================================
# 2. Properties
# ===================================================================

class TestOpinionProperties:
    """Derived properties from opinion components."""

    def test_expected_probability(self):
        """P(x) = b + a·u — the projected probability."""
        o = Opinion(b=0.6, d=0.2, u=0.2, a=0.5)
        # P = 0.6 + 0.5 * 0.2 = 0.7
        assert math.isclose(o.expected_probability, 0.7, rel_tol=1e-9)

    def test_expected_probability_vacuous(self):
        """Vacuous opinion → expected probability equals base rate."""
        o = Opinion(b=0.0, d=0.0, u=1.0, a=0.3)
        assert math.isclose(o.expected_probability, 0.3, rel_tol=1e-9)

    def test_expected_probability_dogmatic(self):
        """Dogmatic belief → expected probability equals belief."""
        o = Opinion(b=0.9, d=0.1, u=0.0, a=0.5)
        assert math.isclose(o.expected_probability, 0.9, rel_tol=1e-9)

    @given(_opinion_components())
    @settings(max_examples=200)
    def test_expected_probability_in_unit_interval(self, components):
        """Property: expected probability is always in [0, 1]."""
        b, d, u, a = components
        assume(abs(b + d + u - 1.0) < 1e-9)
        assume(u >= 0.0)
        o = Opinion(b=b, d=d, u=u, a=a)
        assert -1e-9 <= o.expected_probability <= 1.0 + 1e-9

    def test_is_vacuous(self):
        o = Opinion(b=0.0, d=0.0, u=1.0, a=0.5)
        assert o.is_vacuous is True

    def test_is_not_vacuous(self):
        o = Opinion(b=0.3, d=0.3, u=0.4, a=0.5)
        assert o.is_vacuous is False

    def test_is_dogmatic(self):
        o = Opinion(b=0.7, d=0.3, u=0.0, a=0.5)
        assert o.is_dogmatic is True

    def test_is_not_dogmatic(self):
        o = Opinion(b=0.6, d=0.2, u=0.2, a=0.5)
        assert o.is_dogmatic is False


# ===================================================================
# 3. Factory methods
# ===================================================================

class TestOpinionFactories:
    """Constructing opinions from different input formats."""

    # --- from_confidence ---

    def test_from_confidence_certain(self):
        """High confidence, low uncertainty → strong belief."""
        o = Opinion.from_confidence(probability=0.9, uncertainty=0.1, base_rate=0.5)
        assert math.isclose(o.b + o.d + o.u, 1.0, rel_tol=1e-9)
        assert o.u == pytest.approx(0.1)
        # b = p * (1 - u) = 0.9 * 0.9 = 0.81
        assert o.b == pytest.approx(0.81)
        # d = (1 - p) * (1 - u) = 0.1 * 0.9 = 0.09
        assert o.d == pytest.approx(0.09)

    def test_from_confidence_uncertain(self):
        """High uncertainty → most mass on u."""
        o = Opinion.from_confidence(probability=0.5, uncertainty=0.8, base_rate=0.5)
        assert o.u == pytest.approx(0.8)
        assert o.b == pytest.approx(0.1)
        assert o.d == pytest.approx(0.1)

    def test_from_confidence_base_rate_preserved(self):
        """Base rate flows through to the opinion."""
        o = Opinion.from_confidence(probability=0.5, uncertainty=0.5, base_rate=0.02)
        assert o.a == pytest.approx(0.02)

    def test_from_confidence_zero_uncertainty(self):
        """Zero uncertainty → dogmatic opinion."""
        o = Opinion.from_confidence(probability=0.7, uncertainty=0.0, base_rate=0.5)
        assert o.is_dogmatic
        assert o.b == pytest.approx(0.7)
        assert o.d == pytest.approx(0.3)

    def test_from_confidence_full_uncertainty(self):
        """Full uncertainty → vacuous opinion."""
        o = Opinion.from_confidence(probability=0.5, uncertainty=1.0, base_rate=0.5)
        assert o.is_vacuous

    def test_from_confidence_invalid_probability(self):
        with pytest.raises(ValueError):
            Opinion.from_confidence(probability=1.5, uncertainty=0.1, base_rate=0.5)

    def test_from_confidence_invalid_uncertainty(self):
        with pytest.raises(ValueError):
            Opinion.from_confidence(probability=0.5, uncertainty=-0.1, base_rate=0.5)

    # --- from_evidence ---

    def test_from_evidence_balanced(self):
        """Equal positive and negative evidence → belief ≈ disbelief."""
        o = Opinion.from_evidence(positive=5, negative=5, base_rate=0.5)
        assert o.b == pytest.approx(o.d, abs=1e-9)
        assert o.b == pytest.approx(5 / 12)  # r / (r + s + W) with W=2
        assert o.u == pytest.approx(2 / 12)

    def test_from_evidence_no_evidence(self):
        """No evidence at all → vacuous opinion."""
        o = Opinion.from_evidence(positive=0, negative=0, base_rate=0.5)
        assert o.is_vacuous

    def test_from_evidence_strong_positive(self):
        """Lots of positive evidence → high belief, low uncertainty."""
        o = Opinion.from_evidence(positive=100, negative=0, base_rate=0.5)
        assert o.b > 0.95
        assert o.u < 0.05

    def test_from_evidence_strong_negative(self):
        """Lots of negative evidence → high disbelief."""
        o = Opinion.from_evidence(positive=0, negative=100, base_rate=0.5)
        assert o.d > 0.95

    def test_from_evidence_custom_prior_weight(self):
        """Non-default prior weight W changes how quickly uncertainty shrinks."""
        o_default = Opinion.from_evidence(positive=5, negative=5, base_rate=0.5)
        o_heavy = Opinion.from_evidence(positive=5, negative=5, base_rate=0.5, prior_weight=10)
        # Heavier prior → more uncertainty for same evidence
        assert o_heavy.u > o_default.u

    def test_from_evidence_negative_counts_rejected(self):
        with pytest.raises(ValueError):
            Opinion.from_evidence(positive=-1, negative=5, base_rate=0.5)

    def test_from_evidence_base_rate_preserved(self):
        o = Opinion.from_evidence(positive=10, negative=2, base_rate=0.02)
        assert o.a == pytest.approx(0.02)


# ===================================================================
# 4. Equality & representation
# ===================================================================

class TestOpinionEquality:
    """Equality comparison and string representation."""

    def test_equal_opinions(self):
        o1 = Opinion(b=0.6, d=0.2, u=0.2, a=0.5)
        o2 = Opinion(b=0.6, d=0.2, u=0.2, a=0.5)
        assert o1 == o2

    def test_unequal_opinions(self):
        o1 = Opinion(b=0.6, d=0.2, u=0.2, a=0.5)
        o2 = Opinion(b=0.5, d=0.3, u=0.2, a=0.5)
        assert o1 != o2

    def test_repr_contains_components(self):
        o = Opinion(b=0.6, d=0.2, u=0.2, a=0.5)
        r = repr(o)
        assert "0.6" in r
        assert "0.2" in r
        assert "0.5" in r

    def test_not_equal_to_non_opinion(self):
        o = Opinion(b=0.6, d=0.2, u=0.2, a=0.5)
        assert o != "not an opinion"
        assert o != 0.6


# ===================================================================
# 5. Immutability
# ===================================================================

class TestOpinionImmutability:
    """Opinions should be immutable (frozen dataclass)."""

    def test_cannot_modify_belief(self):
        o = Opinion(b=0.6, d=0.2, u=0.2, a=0.5)
        with pytest.raises((AttributeError, TypeError)):
            o.b = 0.9

    def test_cannot_modify_base_rate(self):
        o = Opinion(b=0.6, d=0.2, u=0.2, a=0.5)
        with pytest.raises((AttributeError, TypeError)):
            o.a = 0.1


# ===================================================================
# 6. Numpy interop
# ===================================================================

class TestOpinionNumpyInterop:
    """Converting to/from numpy arrays for vectorized operations."""

    def test_to_array(self):
        o = Opinion(b=0.6, d=0.2, u=0.2, a=0.5)
        arr = o.to_array()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (4,)
        np.testing.assert_array_almost_equal(arr, [0.6, 0.2, 0.2, 0.5])

    def test_from_array(self):
        arr = np.array([0.6, 0.2, 0.2, 0.5])
        o = Opinion.from_array(arr)
        assert o.b == pytest.approx(0.6)
        assert o.d == pytest.approx(0.2)
        assert o.u == pytest.approx(0.2)
        assert o.a == pytest.approx(0.5)

    def test_from_array_wrong_length(self):
        with pytest.raises(ValueError):
            Opinion.from_array(np.array([0.6, 0.2, 0.2]))

    def test_roundtrip_array(self):
        """to_array → from_array preserves the opinion."""
        o = Opinion(b=0.3, d=0.4, u=0.3, a=0.1)
        o2 = Opinion.from_array(o.to_array())
        assert o == o2
