"""Byzantine-robust fusion operator.

Detects and excludes outlier opinions before fusing, providing resilience
against compromised, corrupted, or malfunctioning signal sources.

Algorithm:
    1. Compute centroid of all opinions in (b, d, u) space
    2. Compute normalized distance from each opinion to centroid
    3. Exclude opinions whose distance exceeds the threshold
    4. Cumulatively fuse the retained opinions
    5. If all excluded, fall back to averaging (graceful degradation)

References:
    Jøsang, A. (2016). Subjective Logic, Ch. 12.
    Byzantine fault tolerance literature.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from slfd.opinion import Opinion
from slfd.fusion import cumulative_fuse, averaging_fuse

# Default threshold: normalized distance in (b,d,u) space
# √3 is max possible distance in the unit simplex
_MAX_DIST = math.sqrt(3.0)
_DEFAULT_THRESHOLD = 0.15


@dataclass(frozen=True, slots=True)
class RobustFuseResult:
    """Result of Byzantine-robust fusion.

    Attributes
    ----------
    fused : Opinion
        The fused opinion after excluding outliers.
    excluded_indices : list[int]
        Indices of opinions that were excluded as outliers.
    n_retained : int
        Number of opinions retained for fusion.
    """

    fused: Opinion
    excluded_indices: list[int]
    n_retained: int


def robust_fuse(
    opinions: list[Opinion],
    threshold: float = _DEFAULT_THRESHOLD,
) -> RobustFuseResult:
    """Fuse opinions with Byzantine-robust outlier exclusion.

    Detects and removes outlier opinions before cumulative fusion.
    Uses distance from centroid in (b, d, u) space as the outlier
    criterion.

    Parameters
    ----------
    opinions : list[Opinion]
        Two or more opinions to fuse.
    threshold : float
        Normalized distance threshold for outlier detection, ∈ [0, ∞).
        Lower → more aggressive exclusion. Default 0.15.

    Returns
    -------
    RobustFuseResult
        Fused opinion, excluded indices, and retention count.

    Raises
    ------
    ValueError
        If opinions list is empty or threshold is negative.
    """
    if not opinions:
        raise ValueError("robust_fuse requires at least one opinion")
    if threshold < 0.0:
        raise ValueError(f"threshold must be ≥ 0, got {threshold}")

    if len(opinions) == 1:
        return RobustFuseResult(
            fused=opinions[0],
            excluded_indices=[],
            n_retained=1,
        )

    # --- Compute centroid in (b, d, u) space ---
    points = np.array([[o.b, o.d, o.u] for o in opinions])
    centroid = np.mean(points, axis=0)

    # --- Compute normalized distances ---
    diffs = points - centroid
    distances = np.sqrt(np.sum(diffs ** 2, axis=1)) / _MAX_DIST

    # --- Identify outliers ---
    excluded = [i for i, d in enumerate(distances) if d > threshold]
    retained = [i for i in range(len(opinions)) if i not in excluded]

    # --- Fuse retained opinions ---
    if not retained:
        # All excluded — graceful fallback to averaging everything
        fused = averaging_fuse(opinions)
        return RobustFuseResult(
            fused=fused,
            excluded_indices=excluded,
            n_retained=0,
        )

    retained_opinions = [opinions[i] for i in retained]

    if len(retained_opinions) == 1:
        fused = retained_opinions[0]
    else:
        fused = cumulative_fuse(retained_opinions)

    return RobustFuseResult(
        fused=fused,
        excluded_indices=sorted(excluded),
        n_retained=len(retained_opinions),
    )
