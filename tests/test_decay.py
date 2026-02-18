"""Tests for temporal decay of opinions.

Stale evidence should degrade toward uncertainty rather than retaining
false precision. The decay operator moves an opinion toward vacuous
(b=0, d=0, u=1) as time passes, controlled by a half-life parameter.

Decay model:
    λ(t) = 2^(-Δt / half_life)
    b_decayed = b · λ
    d_decayed = d · λ
    u_decayed = 1 - b_decayed - d_decayed

References:
    Jøsang, A. (2016). Subjective Logic, §3.8 (Opinion Aging).
"""

import math

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from slfd.opinion import Opinion
from slfd.decay import decay_opinion


# ---------------------------------------------------------------------------
# Helper strategy
# ---------------------------------------------------------------------------
def _non_vacuous_opinion():
    """Strategy producing opinions with some evidence (u < 1)."""
    return (
        st.floats(min_value=0.01, max_value=0.98, allow_nan=False, allow_infinity=False)
        .flatmap(lambda b:
            st.floats(
                min_value=0.01, max_value=max(0.98 - b, 0.01),
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
# 1. Basic decay behavior
# ===================================================================

class TestDecayBasics:
    """Core decay mechanics."""

    def test_zero_elapsed_no_change(self):
        """No time elapsed → opinion unchanged."""
        o = Opinion(b=0.6, d=0.2, u=0.2, a=0.5)
        decayed = decay_opinion(o, elapsed=0.0, half_life=1.0)
        assert decayed.b == pytest.approx(o.b)
        assert decayed.d == pytest.approx(o.d)
        assert decayed.u == pytest.approx(o.u)

    def test_one_half_life_halves_evidence(self):
        """After one half-life, belief and disbelief are halved."""
        o = Opinion(b=0.6, d=0.2, u=0.2, a=0.5)
        decayed = decay_opinion(o, elapsed=1.0, half_life=1.0)
        assert decayed.b == pytest.approx(0.3, abs=1e-9)
        assert decayed.d == pytest.approx(0.1, abs=1e-9)
        assert decayed.u == pytest.approx(0.6, abs=1e-9)

    def test_two_half_lives_quarters_evidence(self):
        """After two half-lives, belief and disbelief are quartered."""
        o = Opinion(b=0.8, d=0.0, u=0.2, a=0.5)
        decayed = decay_opinion(o, elapsed=2.0, half_life=1.0)
        assert decayed.b == pytest.approx(0.2, abs=1e-9)
        assert decayed.u == pytest.approx(0.8, abs=1e-9)

    def test_large_elapsed_approaches_vacuous(self):
        """Very long elapsed time → opinion approaches vacuous."""
        o = Opinion(b=0.9, d=0.05, u=0.05, a=0.5)
        decayed = decay_opinion(o, elapsed=100.0, half_life=1.0)
        assert decayed.u > 0.999

    def test_vacuous_unaffected(self):
        """Decaying a vacuous opinion has no effect (nothing to decay)."""
        v = Opinion(b=0.0, d=0.0, u=1.0, a=0.5)
        decayed = decay_opinion(v, elapsed=10.0, half_life=1.0)
        assert decayed.b == pytest.approx(0.0)
        assert decayed.d == pytest.approx(0.0)
        assert decayed.u == pytest.approx(1.0)

    def test_dogmatic_still_decays(self):
        """Even a dogmatic opinion decays with time."""
        o = Opinion(b=1.0, d=0.0, u=0.0, a=0.5)
        decayed = decay_opinion(o, elapsed=1.0, half_life=1.0)
        assert decayed.b == pytest.approx(0.5, abs=1e-9)
        assert decayed.u == pytest.approx(0.5, abs=1e-9)


# ===================================================================
# 2. Belief/disbelief ratio preservation
# ===================================================================

class TestDecayRatioPreservation:
    """Decay shrinks evidence toward zero but preserves the b:d ratio."""

    def test_ratio_preserved_after_decay(self):
        """b/d ratio stays constant as both decay equally."""
        o = Opinion(b=0.6, d=0.3, u=0.1, a=0.5)
        decayed = decay_opinion(o, elapsed=1.0, half_life=1.0)
        original_ratio = o.b / o.d
        decayed_ratio = decayed.b / decayed.d
        assert decayed_ratio == pytest.approx(original_ratio, rel=1e-9)

    def test_ratio_preserved_across_multiple_half_lives(self):
        o = Opinion(b=0.5, d=0.2, u=0.3, a=0.5)
        decayed = decay_opinion(o, elapsed=3.5, half_life=1.0)
        original_ratio = o.b / o.d
        decayed_ratio = decayed.b / decayed.d
        assert decayed_ratio == pytest.approx(original_ratio, rel=1e-9)


# ===================================================================
# 3. Half-life parameter
# ===================================================================

class TestDecayHalfLife:
    """Different half-life values control decay speed."""

    def test_longer_half_life_slower_decay(self):
        """Longer half-life → less decay for same elapsed time."""
        o = Opinion(b=0.6, d=0.2, u=0.2, a=0.5)
        fast = decay_opinion(o, elapsed=1.0, half_life=1.0)
        slow = decay_opinion(o, elapsed=1.0, half_life=10.0)
        assert slow.b > fast.b
        assert slow.u < fast.u

    def test_very_short_half_life_fast_decay(self):
        """Very short half-life → rapid decay to vacuous."""
        o = Opinion(b=0.8, d=0.1, u=0.1, a=0.5)
        decayed = decay_opinion(o, elapsed=1.0, half_life=0.01)
        assert decayed.u > 0.99

    def test_half_life_must_be_positive(self):
        o = Opinion(b=0.5, d=0.3, u=0.2, a=0.5)
        with pytest.raises(ValueError, match="half_life"):
            decay_opinion(o, elapsed=1.0, half_life=0.0)

    def test_half_life_negative_rejected(self):
        o = Opinion(b=0.5, d=0.3, u=0.2, a=0.5)
        with pytest.raises(ValueError, match="half_life"):
            decay_opinion(o, elapsed=1.0, half_life=-1.0)


# ===================================================================
# 4. Elapsed time validation
# ===================================================================

class TestDecayElapsedValidation:
    """Elapsed time must be non-negative."""

    def test_negative_elapsed_rejected(self):
        o = Opinion(b=0.5, d=0.3, u=0.2, a=0.5)
        with pytest.raises(ValueError, match="elapsed"):
            decay_opinion(o, elapsed=-1.0, half_life=1.0)


# ===================================================================
# 5. Result validity
# ===================================================================

class TestDecayResultValidity:
    """Decayed opinions must be valid opinions."""

    def test_result_sums_to_one(self):
        o = Opinion(b=0.7, d=0.2, u=0.1, a=0.5)
        decayed = decay_opinion(o, elapsed=2.5, half_life=1.0)
        assert math.isclose(decayed.b + decayed.d + decayed.u, 1.0, rel_tol=1e-9)

    def test_base_rate_preserved(self):
        """Decay does not alter the base rate."""
        o = Opinion(b=0.6, d=0.2, u=0.2, a=0.02)
        decayed = decay_opinion(o, elapsed=1.0, half_life=1.0)
        assert decayed.a == pytest.approx(0.02)

    @given(
        _non_vacuous_opinion(),
        st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200)
    def test_decayed_always_valid(self, opinion, elapsed, half_life):
        """Property: decay always produces a valid opinion."""
        decayed = decay_opinion(opinion, elapsed=elapsed, half_life=half_life)
        assert -1e-9 <= decayed.b <= 1.0 + 1e-9
        assert -1e-9 <= decayed.d <= 1.0 + 1e-9
        assert -1e-9 <= decayed.u <= 1.0 + 1e-9
        assert abs(decayed.b + decayed.d + decayed.u - 1.0) < 1e-6

    @given(
        _non_vacuous_opinion(),
        st.floats(min_value=0.001, max_value=50.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200)
    def test_uncertainty_never_decreases(self, opinion, elapsed, half_life):
        """Property: decay never makes an opinion MORE certain."""
        decayed = decay_opinion(opinion, elapsed=elapsed, half_life=half_life)
        assert decayed.u >= opinion.u - 1e-9
