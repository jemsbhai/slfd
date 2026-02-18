"""Tests for Byzantine-robust fusion operator.

When one or more signal sources are compromised (adversarial, corrupted,
or malfunctioning), standard cumulative fusion is vulnerable — a single
rogue source can skew the fused opinion dramatically.

robust_fuse detects and excludes outlier opinions before fusing, using
the distance from the centroid in (b, d, u) space as the outlier criterion.

This is the foundation for E-FD5: Byzantine-Robust Fusion Under
Adversarial Signals.

References:
    Jøsang, A. (2016). Subjective Logic, Ch. 12.
    Byzantine fault tolerance literature.
"""

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from slfd.opinion import Opinion
from slfd.fusion import cumulative_fuse, conflict_metric
from slfd.robust import robust_fuse, RobustFuseResult


# ---------------------------------------------------------------------------
# Helper strategy
# ---------------------------------------------------------------------------
def _non_dogmatic_opinion():
    """Strategy producing opinions with u > 0."""
    return (
        st.floats(min_value=0.01, max_value=0.90, allow_nan=False, allow_infinity=False)
        .flatmap(lambda b:
            st.floats(
                min_value=0.01, max_value=max(0.90 - b, 0.01),
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
# 1. Basic behavior (no adversarial)
# ===================================================================

class TestRobustFuseCleanInputs:
    """When all sources are honest, robust_fuse ≈ cumulative_fuse."""

    def test_agreeing_sources_match_cumulative(self):
        """No outliers → result close to cumulative fusion."""
        opinions = [
            Opinion(b=0.6, d=0.1, u=0.3, a=0.5),
            Opinion(b=0.5, d=0.2, u=0.3, a=0.5),
            Opinion(b=0.55, d=0.15, u=0.3, a=0.5),
            Opinion(b=0.58, d=0.12, u=0.3, a=0.5),
        ]
        robust = robust_fuse(opinions)
        cumul = cumulative_fuse(opinions)
        # Should be close since no outliers are excluded
        assert robust.fused.b == pytest.approx(cumul.b, abs=0.15)
        assert robust.fused.d == pytest.approx(cumul.d, abs=0.15)

    def test_returns_result_object(self):
        opinions = [
            Opinion(b=0.5, d=0.2, u=0.3, a=0.5),
            Opinion(b=0.6, d=0.1, u=0.3, a=0.5),
        ]
        result = robust_fuse(opinions)
        assert isinstance(result, RobustFuseResult)

    def test_result_has_fused_opinion(self):
        opinions = [
            Opinion(b=0.5, d=0.2, u=0.3, a=0.5),
            Opinion(b=0.6, d=0.1, u=0.3, a=0.5),
        ]
        result = robust_fuse(opinions)
        assert isinstance(result.fused, Opinion)

    def test_result_has_excluded_indices(self):
        opinions = [
            Opinion(b=0.5, d=0.2, u=0.3, a=0.5),
            Opinion(b=0.6, d=0.1, u=0.3, a=0.5),
        ]
        result = robust_fuse(opinions)
        assert isinstance(result.excluded_indices, list)

    def test_no_exclusions_when_all_agree(self):
        opinions = [
            Opinion(b=0.5, d=0.2, u=0.3, a=0.5),
            Opinion(b=0.52, d=0.18, u=0.3, a=0.5),
            Opinion(b=0.48, d=0.22, u=0.3, a=0.5),
        ]
        result = robust_fuse(opinions)
        assert len(result.excluded_indices) == 0


# ===================================================================
# 2. Adversarial detection
# ===================================================================

class TestRobustFuseAdversarialDetection:
    """Rogue opinions should be detected and excluded."""

    def test_inverted_source_detected(self):
        """One source says opposite of the rest → excluded."""
        honest = [
            Opinion(b=0.7, d=0.1, u=0.2, a=0.5),
            Opinion(b=0.65, d=0.15, u=0.2, a=0.5),
            Opinion(b=0.72, d=0.08, u=0.2, a=0.5),
        ]
        rogue = Opinion(b=0.05, d=0.85, u=0.1, a=0.5)
        all_opinions = honest + [rogue]
        result = robust_fuse(all_opinions, threshold=0.3)
        assert 3 in result.excluded_indices

    def test_fused_result_ignores_rogue(self):
        """After excluding rogue, fused opinion matches honest consensus."""
        honest = [
            Opinion(b=0.7, d=0.1, u=0.2, a=0.5),
            Opinion(b=0.65, d=0.15, u=0.2, a=0.5),
            Opinion(b=0.72, d=0.08, u=0.2, a=0.5),
        ]
        rogue = Opinion(b=0.05, d=0.85, u=0.1, a=0.5)

        robust_result = robust_fuse(honest + [rogue], threshold=0.3)
        honest_fused = cumulative_fuse(honest)

        # Robust result should be close to honest-only fusion
        assert robust_result.fused.b == pytest.approx(honest_fused.b, abs=0.05)
        assert robust_result.fused.d == pytest.approx(honest_fused.d, abs=0.05)

    def test_multiple_rogues_detected(self):
        """Two rogue sources among honest ones."""
        honest = [
            Opinion(b=0.6, d=0.1, u=0.3, a=0.5),
            Opinion(b=0.55, d=0.15, u=0.3, a=0.5),
            Opinion(b=0.58, d=0.12, u=0.3, a=0.5),
        ]
        rogues = [
            Opinion(b=0.05, d=0.85, u=0.1, a=0.5),
            Opinion(b=0.0, d=0.9, u=0.1, a=0.5),
        ]
        result = robust_fuse(honest + rogues, threshold=0.3)
        assert 3 in result.excluded_indices
        assert 4 in result.excluded_indices

    def test_random_noise_source_detected(self):
        """A source with unusual (b,d,u) pattern is flagged."""
        honest = [
            Opinion(b=0.7, d=0.1, u=0.2, a=0.5),
            Opinion(b=0.65, d=0.15, u=0.2, a=0.5),
            Opinion(b=0.72, d=0.08, u=0.2, a=0.5),
            Opinion(b=0.68, d=0.12, u=0.2, a=0.5),
        ]
        # Noisy source — very different profile
        noisy = Opinion(b=0.3, d=0.3, u=0.4, a=0.5)
        result = robust_fuse(honest + [noisy], threshold=0.2)
        assert 4 in result.excluded_indices


# ===================================================================
# 3. Threshold behavior
# ===================================================================

class TestRobustFuseThreshold:
    """The threshold parameter controls sensitivity."""

    def test_higher_threshold_fewer_exclusions(self):
        """Lenient threshold → fewer sources excluded."""
        honest = [
            Opinion(b=0.6, d=0.1, u=0.3, a=0.5),
            Opinion(b=0.55, d=0.15, u=0.3, a=0.5),
            Opinion(b=0.58, d=0.12, u=0.3, a=0.5),
        ]
        outlier = Opinion(b=0.3, d=0.4, u=0.3, a=0.5)
        strict = robust_fuse(honest + [outlier], threshold=0.1)
        lenient = robust_fuse(honest + [outlier], threshold=0.5)
        assert len(lenient.excluded_indices) <= len(strict.excluded_indices)

    def test_zero_threshold_excludes_any_deviation(self):
        """threshold=0 would exclude everything except perfect duplicates."""
        opinions = [
            Opinion(b=0.5, d=0.2, u=0.3, a=0.5),
            Opinion(b=0.5, d=0.2, u=0.3, a=0.5),
            Opinion(b=0.51, d=0.19, u=0.3, a=0.5),
        ]
        result = robust_fuse(opinions, threshold=0.001)
        # The slightly different one may be excluded
        # At minimum, it shouldn't crash
        assert isinstance(result.fused, Opinion)

    def test_threshold_must_be_non_negative(self):
        opinions = [
            Opinion(b=0.5, d=0.2, u=0.3, a=0.5),
            Opinion(b=0.6, d=0.1, u=0.3, a=0.5),
        ]
        with pytest.raises(ValueError, match="threshold"):
            robust_fuse(opinions, threshold=-0.1)

    def test_default_threshold(self):
        """Default threshold should be reasonable (0.15 per research plan)."""
        opinions = [
            Opinion(b=0.6, d=0.1, u=0.3, a=0.5),
            Opinion(b=0.55, d=0.15, u=0.3, a=0.5),
        ]
        # Should not crash with default threshold
        result = robust_fuse(opinions)
        assert isinstance(result.fused, Opinion)


# ===================================================================
# 4. Edge cases
# ===================================================================

class TestRobustFuseEdgeCases:
    """Boundary conditions."""

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="at least"):
            robust_fuse([])

    def test_single_opinion(self):
        o = Opinion(b=0.6, d=0.2, u=0.2, a=0.5)
        result = robust_fuse([o])
        assert result.fused == o
        assert len(result.excluded_indices) == 0

    def test_two_opinions_no_crash(self):
        """With only 2 opinions, can still compute centroid and distances."""
        o1 = Opinion(b=0.6, d=0.2, u=0.2, a=0.5)
        o2 = Opinion(b=0.1, d=0.7, u=0.2, a=0.5)
        result = robust_fuse([o1, o2], threshold=0.3)
        assert isinstance(result.fused, Opinion)

    def test_all_excluded_falls_back(self):
        """If threshold is so strict all are excluded, fall back to
        averaging all opinions (graceful degradation)."""
        opinions = [
            Opinion(b=0.8, d=0.0, u=0.2, a=0.5),
            Opinion(b=0.0, d=0.8, u=0.2, a=0.5),
            Opinion(b=0.4, d=0.4, u=0.2, a=0.5),
        ]
        result = robust_fuse(opinions, threshold=0.01)
        # Should still produce a valid opinion
        assert isinstance(result.fused, Opinion)
        assert math.isclose(result.fused.b + result.fused.d + result.fused.u, 1.0, rel_tol=1e-6)

    def test_result_has_n_retained(self):
        opinions = [
            Opinion(b=0.6, d=0.1, u=0.3, a=0.5),
            Opinion(b=0.55, d=0.15, u=0.3, a=0.5),
            Opinion(b=0.05, d=0.85, u=0.1, a=0.5),
        ]
        result = robust_fuse(opinions, threshold=0.3)
        assert result.n_retained == len(opinions) - len(result.excluded_indices)
        assert result.n_retained >= 1


# ===================================================================
# 5. Result validity (property-based)
# ===================================================================

class TestRobustFuseProperties:
    """Property-based tests for correctness guarantees."""

    @given(st.lists(_non_dogmatic_opinion(), min_size=2, max_size=6))
    @settings(max_examples=100)
    def test_result_always_valid_opinion(self, opinions):
        """Property: robust fusion always produces a valid opinion."""
        a = opinions[0].a
        aligned = [Opinion(b=o.b, d=o.d, u=o.u, a=a) for o in opinions]
        result = robust_fuse(aligned)
        assert -1e-9 <= result.fused.b <= 1.0 + 1e-9
        assert -1e-9 <= result.fused.d <= 1.0 + 1e-9
        assert -1e-9 <= result.fused.u <= 1.0 + 1e-9
        assert abs(result.fused.b + result.fused.d + result.fused.u - 1.0) < 1e-6

    @given(st.lists(_non_dogmatic_opinion(), min_size=2, max_size=6))
    @settings(max_examples=100)
    def test_excluded_indices_valid(self, opinions):
        """Property: excluded indices are valid list positions."""
        a = opinions[0].a
        aligned = [Opinion(b=o.b, d=o.d, u=o.u, a=a) for o in opinions]
        result = robust_fuse(aligned)
        for idx in result.excluded_indices:
            assert 0 <= idx < len(aligned)

    @given(st.lists(_non_dogmatic_opinion(), min_size=2, max_size=6))
    @settings(max_examples=100)
    def test_n_retained_consistent(self, opinions):
        """Property: n_retained + len(excluded) == len(input)."""
        a = opinions[0].a
        aligned = [Opinion(b=o.b, d=o.d, u=o.u, a=a) for o in opinions]
        result = robust_fuse(aligned)
        assert result.n_retained + len(result.excluded_indices) == len(aligned)
