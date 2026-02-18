"""Tests for trust discount operator.

In a fraud detection pipeline, not all signal sources are equally reliable.
A well-calibrated ML model deserves more trust than a simple heuristic.
Trust discounting adjusts an opinion based on how much we trust its source.

Trust discount rule (Jøsang, 2016, Ch. 14):
    Given source opinion ω_S = (b, d, u, a) and trust in source t ∈ [0, 1]:
        b_discounted = t · b
        d_discounted = t · d
        u_discounted = 1 - t·b - t·d
        a_discounted = a  (base rate unchanged)

Effect: lower trust pushes the opinion toward vacuous (more uncertainty).
At t=0, we completely distrust the source → vacuous opinion.
At t=1, we fully trust the source → opinion unchanged.
"""

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from slfd.opinion import Opinion
from slfd.trust import trust_discount


# ---------------------------------------------------------------------------
# Helper strategy
# ---------------------------------------------------------------------------
def _valid_opinion():
    """Strategy producing valid Opinion objects."""
    return (
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
        .flatmap(lambda b:
            st.floats(
                min_value=0.0, max_value=max(1.0 - b, 0.0),
                allow_nan=False, allow_infinity=False,
            ).flatmap(lambda d:
                st.just((b, d, 1.0 - b - d))
            )
        )
        .flatmap(lambda bdu:
            st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False)
            .map(lambda a: Opinion(b=bdu[0], d=bdu[1], u=bdu[2], a=a))
        )
    )


# ===================================================================
# 1. Basic behavior
# ===================================================================

class TestTrustDiscountBasics:
    """Core trust discount mechanics."""

    def test_full_trust_no_change(self):
        """t=1 → opinion unchanged."""
        o = Opinion(b=0.6, d=0.2, u=0.2, a=0.5)
        discounted = trust_discount(o, trust=1.0)
        assert discounted.b == pytest.approx(o.b)
        assert discounted.d == pytest.approx(o.d)
        assert discounted.u == pytest.approx(o.u)

    def test_zero_trust_gives_vacuous(self):
        """t=0 → complete distrust → vacuous opinion."""
        o = Opinion(b=0.8, d=0.1, u=0.1, a=0.5)
        discounted = trust_discount(o, trust=0.0)
        assert discounted.is_vacuous

    def test_half_trust_halves_evidence(self):
        """t=0.5 → belief and disbelief halved, uncertainty fills gap."""
        o = Opinion(b=0.6, d=0.2, u=0.2, a=0.5)
        discounted = trust_discount(o, trust=0.5)
        assert discounted.b == pytest.approx(0.3)
        assert discounted.d == pytest.approx(0.1)
        assert discounted.u == pytest.approx(0.6)

    def test_base_rate_preserved(self):
        """Trust discount does not alter the base rate."""
        o = Opinion(b=0.7, d=0.1, u=0.2, a=0.02)
        discounted = trust_discount(o, trust=0.5)
        assert discounted.a == pytest.approx(0.02)

    def test_vacuous_unaffected(self):
        """Discounting a vacuous opinion has no effect."""
        v = Opinion(b=0.0, d=0.0, u=1.0, a=0.5)
        discounted = trust_discount(v, trust=0.3)
        assert discounted.is_vacuous


# ===================================================================
# 2. Monotonicity
# ===================================================================

class TestTrustDiscountMonotonicity:
    """Lower trust → more uncertainty, less evidence."""

    def test_lower_trust_more_uncertainty(self):
        o = Opinion(b=0.7, d=0.1, u=0.2, a=0.5)
        high = trust_discount(o, trust=0.9)
        low = trust_discount(o, trust=0.3)
        assert low.u > high.u

    def test_lower_trust_less_belief(self):
        o = Opinion(b=0.7, d=0.1, u=0.2, a=0.5)
        high = trust_discount(o, trust=0.9)
        low = trust_discount(o, trust=0.3)
        assert low.b < high.b

    def test_lower_trust_less_disbelief(self):
        o = Opinion(b=0.1, d=0.7, u=0.2, a=0.5)
        high = trust_discount(o, trust=0.9)
        low = trust_discount(o, trust=0.3)
        assert low.d < high.d


# ===================================================================
# 3. Ratio preservation
# ===================================================================

class TestTrustDiscountRatio:
    """Trust discount preserves the b:d ratio."""

    def test_ratio_preserved(self):
        o = Opinion(b=0.6, d=0.3, u=0.1, a=0.5)
        discounted = trust_discount(o, trust=0.4)
        original_ratio = o.b / o.d
        discounted_ratio = discounted.b / discounted.d
        assert discounted_ratio == pytest.approx(original_ratio, rel=1e-9)


# ===================================================================
# 4. Validation
# ===================================================================

class TestTrustDiscountValidation:
    """Input validation."""

    def test_trust_above_one_rejected(self):
        o = Opinion(b=0.5, d=0.3, u=0.2, a=0.5)
        with pytest.raises(ValueError, match="trust"):
            trust_discount(o, trust=1.1)

    def test_trust_below_zero_rejected(self):
        o = Opinion(b=0.5, d=0.3, u=0.2, a=0.5)
        with pytest.raises(ValueError, match="trust"):
            trust_discount(o, trust=-0.1)


# ===================================================================
# 5. Result validity (property-based)
# ===================================================================

class TestTrustDiscountProperties:
    """Property-based tests for correctness guarantees."""

    @given(
        _valid_opinion(),
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200)
    def test_result_always_valid(self, opinion, trust):
        """Property: trust discount always produces a valid opinion."""
        discounted = trust_discount(opinion, trust=trust)
        assert -1e-9 <= discounted.b <= 1.0 + 1e-9
        assert -1e-9 <= discounted.d <= 1.0 + 1e-9
        assert -1e-9 <= discounted.u <= 1.0 + 1e-9
        assert abs(discounted.b + discounted.d + discounted.u - 1.0) < 1e-6

    @given(
        _valid_opinion(),
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200)
    def test_uncertainty_never_decreases(self, opinion, trust):
        """Property: discounting never makes an opinion MORE certain."""
        discounted = trust_discount(opinion, trust=trust)
        assert discounted.u >= opinion.u - 1e-9
