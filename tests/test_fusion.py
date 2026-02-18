"""Tests for multi-source opinion fusion operators.

Implements Jøsang's Subjective Logic fusion operators:
    - cumulative_fuse: for independent sources (accumulates evidence)
    - averaging_fuse: for correlated/dependent sources
    - conflict_metric: quantifies disagreement between opinions

References:
    Jøsang, A. (2016). Subjective Logic, Ch. 12 (Multi-Source Fusion).
"""

import math

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from slfd.opinion import Opinion
from slfd.fusion import cumulative_fuse, averaging_fuse, conflict_metric


# ---------------------------------------------------------------------------
# Helper: generate valid opinions via Hypothesis
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


def _non_dogmatic_opinion():
    """Strategy producing opinions with u > 0 (required for cumulative fusion)."""
    return (
        st.floats(min_value=0.0, max_value=0.98, allow_nan=False, allow_infinity=False)
        .flatmap(lambda b:
            st.floats(
                min_value=0.0, max_value=max(0.98 - b, 0.0),
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
# 1. Cumulative fusion (independent sources)
# ===================================================================

class TestCumulativeFusion:
    """Cumulative fusion accumulates evidence from independent sources.

    The result should have LESS uncertainty than either input — more
    evidence means more certainty.
    """

    def test_two_agreeing_sources(self):
        """Two sources that both believe → stronger belief."""
        o1 = Opinion(b=0.6, d=0.1, u=0.3, a=0.5)
        o2 = Opinion(b=0.7, d=0.1, u=0.2, a=0.5)
        fused = cumulative_fuse([o1, o2])
        assert fused.b > max(o1.b, o2.b)
        assert fused.u < min(o1.u, o2.u)
        assert math.isclose(fused.b + fused.d + fused.u, 1.0, rel_tol=1e-9)

    def test_two_opposing_sources(self):
        """One believes, one disbelieves → uncertainty should reflect conflict."""
        o1 = Opinion(b=0.8, d=0.0, u=0.2, a=0.5)
        o2 = Opinion(b=0.0, d=0.8, u=0.2, a=0.5)
        fused = cumulative_fuse([o1, o2])
        assert math.isclose(fused.b + fused.d + fused.u, 1.0, rel_tol=1e-9)

    def test_fuse_with_vacuous_is_identity(self):
        """Fusing with a vacuous opinion (no evidence) preserves the other."""
        o1 = Opinion(b=0.6, d=0.2, u=0.2, a=0.5)
        vacuous = Opinion(b=0.0, d=0.0, u=1.0, a=0.5)
        fused = cumulative_fuse([o1, vacuous])
        assert fused.b == pytest.approx(o1.b, abs=1e-6)
        assert fused.d == pytest.approx(o1.d, abs=1e-6)
        assert fused.u == pytest.approx(o1.u, abs=1e-6)

    def test_uncertainty_decreases_with_more_sources(self):
        """Adding more sources (with evidence) should reduce uncertainty."""
        opinions = [Opinion(b=0.4, d=0.1, u=0.5, a=0.5) for _ in range(5)]
        prev_u = 1.0
        for n in range(1, len(opinions) + 1):
            fused = cumulative_fuse(opinions[:n])
            assert fused.u <= prev_u + 1e-9
            prev_u = fused.u

    def test_single_opinion_returned_as_is(self):
        """Fusing a single opinion returns an equivalent opinion."""
        o = Opinion(b=0.6, d=0.2, u=0.2, a=0.5)
        fused = cumulative_fuse([o])
        assert fused == o

    def test_empty_list_raises(self):
        """Cannot fuse zero opinions."""
        with pytest.raises(ValueError, match="at least one"):
            cumulative_fuse([])

    def test_base_rate_consistency(self):
        """Fused opinion has a valid base rate."""
        o1 = Opinion(b=0.5, d=0.2, u=0.3, a=0.3)
        o2 = Opinion(b=0.3, d=0.4, u=0.3, a=0.3)
        fused = cumulative_fuse([o1, o2])
        assert 0.0 <= fused.a <= 1.0

    def test_result_is_valid_opinion(self):
        """Fused result satisfies all Opinion constraints."""
        o1 = Opinion(b=0.3, d=0.3, u=0.4, a=0.5)
        o2 = Opinion(b=0.5, d=0.1, u=0.4, a=0.5)
        o3 = Opinion(b=0.1, d=0.6, u=0.3, a=0.5)
        fused = cumulative_fuse([o1, o2, o3])
        assert 0.0 <= fused.b <= 1.0
        assert 0.0 <= fused.d <= 1.0
        assert 0.0 <= fused.u <= 1.0
        assert math.isclose(fused.b + fused.d + fused.u, 1.0, rel_tol=1e-9)

    @given(st.lists(_non_dogmatic_opinion(), min_size=2, max_size=5))
    @settings(max_examples=100)
    def test_result_always_valid(self, opinions):
        """Property: cumulative fusion always produces a valid opinion."""
        # Ensure all have same base rate for clean fusion
        a = opinions[0].a
        aligned = [Opinion(b=o.b, d=o.d, u=o.u, a=a) for o in opinions]
        fused = cumulative_fuse(aligned)
        assert -1e-9 <= fused.b <= 1.0 + 1e-9
        assert -1e-9 <= fused.d <= 1.0 + 1e-9
        assert -1e-9 <= fused.u <= 1.0 + 1e-9
        assert abs(fused.b + fused.d + fused.u - 1.0) < 1e-6


# ===================================================================
# 2. Averaging fusion (correlated sources)
# ===================================================================

class TestAveragingFusion:
    """Averaging fusion for when sources are NOT independent.

    Unlike cumulative fusion, averaging does NOT accumulate evidence —
    it computes a weighted average, so uncertainty doesn't vanish.
    """

    def test_two_identical_opinions(self):
        """Averaging identical opinions returns the same opinion."""
        o = Opinion(b=0.6, d=0.2, u=0.2, a=0.5)
        fused = averaging_fuse([o, o])
        assert fused.b == pytest.approx(o.b, abs=1e-6)
        assert fused.d == pytest.approx(o.d, abs=1e-6)
        assert fused.u == pytest.approx(o.u, abs=1e-6)

    def test_averaging_preserves_uncertainty(self):
        """Averaging does NOT reduce uncertainty the way cumulative does."""
        opinions = [Opinion(b=0.4, d=0.1, u=0.5, a=0.5) for _ in range(5)]
        fused = averaging_fuse(opinions)
        # Uncertainty should stay roughly the same, not collapse
        assert fused.u == pytest.approx(0.5, abs=0.05)

    def test_averaging_two_sources(self):
        """Average of two different opinions lands between them."""
        o1 = Opinion(b=0.8, d=0.0, u=0.2, a=0.5)
        o2 = Opinion(b=0.0, d=0.8, u=0.2, a=0.5)
        fused = averaging_fuse([o1, o2])
        assert fused.b == pytest.approx(0.4, abs=1e-6)
        assert fused.d == pytest.approx(0.4, abs=1e-6)
        assert math.isclose(fused.b + fused.d + fused.u, 1.0, rel_tol=1e-9)

    def test_single_opinion(self):
        o = Opinion(b=0.6, d=0.2, u=0.2, a=0.5)
        fused = averaging_fuse([o])
        assert fused == o

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            averaging_fuse([])

    def test_result_is_valid_opinion(self):
        o1 = Opinion(b=0.3, d=0.5, u=0.2, a=0.5)
        o2 = Opinion(b=0.7, d=0.1, u=0.2, a=0.5)
        fused = averaging_fuse([o1, o2])
        assert math.isclose(fused.b + fused.d + fused.u, 1.0, rel_tol=1e-9)

    @given(st.lists(_valid_opinion(), min_size=2, max_size=5))
    @settings(max_examples=100)
    def test_result_always_valid(self, opinions):
        """Property: averaging fusion always produces a valid opinion."""
        a = opinions[0].a
        aligned = [Opinion(b=o.b, d=o.d, u=o.u, a=a) for o in opinions]
        fused = averaging_fuse(aligned)
        assert -1e-9 <= fused.b <= 1.0 + 1e-9
        assert -1e-9 <= fused.d <= 1.0 + 1e-9
        assert -1e-9 <= fused.u <= 1.0 + 1e-9
        assert abs(fused.b + fused.d + fused.u - 1.0) < 1e-6


# ===================================================================
# 3. Conflict metric
# ===================================================================

class TestConflictMetric:
    """Conflict metric quantifies disagreement between opinions.

    Ranges from 0 (perfect agreement) to 1 (maximum conflict).
    """

    def test_identical_opinions_zero_conflict(self):
        """No conflict between identical opinions."""
        o = Opinion(b=0.6, d=0.2, u=0.2, a=0.5)
        assert conflict_metric([o, o]) == pytest.approx(0.0, abs=1e-6)

    def test_opposing_opinions_high_conflict(self):
        """Strong believer vs strong disbeliever → high conflict."""
        o1 = Opinion(b=0.9, d=0.0, u=0.1, a=0.5)
        o2 = Opinion(b=0.0, d=0.9, u=0.1, a=0.5)
        c = conflict_metric([o1, o2])
        assert c > 0.5

    def test_vacuous_opinions_zero_conflict(self):
        """Two vacuous opinions → no conflict (no evidence to disagree on)."""
        v1 = Opinion(b=0.0, d=0.0, u=1.0, a=0.5)
        v2 = Opinion(b=0.0, d=0.0, u=1.0, a=0.5)
        assert conflict_metric([v1, v2]) == pytest.approx(0.0, abs=1e-6)

    def test_conflict_in_unit_interval(self):
        """Conflict is always in [0, 1]."""
        o1 = Opinion(b=0.8, d=0.1, u=0.1, a=0.5)
        o2 = Opinion(b=0.1, d=0.7, u=0.2, a=0.5)
        c = conflict_metric([o1, o2])
        assert 0.0 <= c <= 1.0

    def test_conflict_symmetric(self):
        """conflict([A, B]) == conflict([B, A])."""
        o1 = Opinion(b=0.7, d=0.1, u=0.2, a=0.5)
        o2 = Opinion(b=0.2, d=0.6, u=0.2, a=0.5)
        assert conflict_metric([o1, o2]) == pytest.approx(conflict_metric([o2, o1]))

    def test_conflict_three_sources(self):
        """Conflict among three opinions is computable and in [0, 1]."""
        o1 = Opinion(b=0.8, d=0.1, u=0.1, a=0.5)
        o2 = Opinion(b=0.1, d=0.7, u=0.2, a=0.5)
        o3 = Opinion(b=0.5, d=0.2, u=0.3, a=0.5)
        c = conflict_metric([o1, o2, o3])
        assert 0.0 <= c <= 1.0

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="at least two"):
            conflict_metric([])

    def test_single_opinion_raises(self):
        with pytest.raises(ValueError, match="at least two"):
            conflict_metric([Opinion(b=0.5, d=0.3, u=0.2, a=0.5)])

    @given(st.lists(_valid_opinion(), min_size=2, max_size=5))
    @settings(max_examples=100)
    def test_conflict_always_in_unit_interval(self, opinions):
        """Property: conflict metric is always in [0, 1]."""
        c = conflict_metric(opinions)
        assert -1e-9 <= c <= 1.0 + 1e-9
