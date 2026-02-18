"""Multi-source opinion fusion operators.

Implements Jøsang's Subjective Logic fusion operators for combining opinions
from multiple signal sources in a fraud detection pipeline.

Operators:
    cumulative_fuse  — for independent sources (accumulates evidence)
    averaging_fuse   — for correlated/dependent sources
    conflict_metric  — quantifies disagreement between opinions

References:
    Jøsang, A. (2016). Subjective Logic, Ch. 12 (Multi-Source Fusion).
"""

from __future__ import annotations

import math
from functools import reduce

from slfd.opinion import Opinion


# ===================================================================
# Cumulative fusion (independent sources)
# ===================================================================

def _cumulative_fuse_pair(a: Opinion, b: Opinion) -> Opinion:
    """Fuse two opinions assuming independent sources.

    Uses Jøsang's cumulative fusion rule:
        κ = u_A + u_B - u_A·u_B
        b = (b_A·u_B + b_B·u_A) / κ
        d = (d_A·u_B + d_B·u_A) / κ
        u = (u_A·u_B) / κ

    When both are dogmatic (κ = 0), falls back to simple averaging.
    """
    kappa = a.u + b.u - a.u * b.u

    if kappa < 1e-12:
        # Both dogmatic — degenerate case, average
        return Opinion(
            b=(a.b + b.b) / 2.0,
            d=(a.d + b.d) / 2.0,
            u=0.0,
            a=(a.a + b.a) / 2.0,
        )

    fused_b = (a.b * b.u + b.b * a.u) / kappa
    fused_d = (a.d * b.u + b.d * a.u) / kappa
    fused_u = (a.u * b.u) / kappa
    fused_a = (a.a + b.a) / 2.0

    return Opinion(b=fused_b, d=fused_d, u=fused_u, a=fused_a)


def cumulative_fuse(opinions: list[Opinion]) -> Opinion:
    """Fuse multiple opinions from independent sources.

    Evidence is accumulated — more sources means less uncertainty.
    Requires at least one opinion. All opinions should have u > 0
    (non-dogmatic) for well-defined results, though dogmatic pairs
    are handled via averaging fallback.

    Parameters
    ----------
    opinions : list[Opinion]
        One or more opinions to fuse.

    Returns
    -------
    Opinion
        The fused opinion.

    Raises
    ------
    ValueError
        If the list is empty.
    """
    if not opinions:
        raise ValueError("cumulative_fuse requires at least one opinion")
    if len(opinions) == 1:
        return opinions[0]
    return reduce(_cumulative_fuse_pair, opinions)


# ===================================================================
# Averaging fusion (correlated sources)
# ===================================================================

def averaging_fuse(opinions: list[Opinion]) -> Opinion:
    """Fuse multiple opinions by simple averaging.

    Appropriate when sources are correlated or dependent — evidence
    is NOT accumulated, so uncertainty does not vanish with more sources.

    Parameters
    ----------
    opinions : list[Opinion]
        One or more opinions to fuse.

    Returns
    -------
    Opinion
        The averaged opinion.

    Raises
    ------
    ValueError
        If the list is empty.
    """
    if not opinions:
        raise ValueError("averaging_fuse requires at least one opinion")
    if len(opinions) == 1:
        return opinions[0]

    n = len(opinions)
    avg_b = sum(o.b for o in opinions) / n
    avg_d = sum(o.d for o in opinions) / n
    avg_u = sum(o.u for o in opinions) / n
    avg_a = sum(o.a for o in opinions) / n

    return Opinion(b=avg_b, d=avg_d, u=avg_u, a=avg_a)


# ===================================================================
# Conflict metric
# ===================================================================

def conflict_metric(opinions: list[Opinion]) -> float:
    """Quantify disagreement between opinions.

    Computes the mean pairwise normalized Euclidean distance in
    (b, d) space. This captures how much opinions point in different
    evidential directions.

    - 0.0 = perfect agreement (all opinions identical)
    - 1.0 = maximum conflict (one fully believes, another fully disbelieves)

    The metric is:
        conflict = mean over all pairs (i, j) of:
            ‖(b_i, d_i) - (b_j, d_j)‖₂ / √2

    where √2 is the maximum possible distance in the (b, d) simplex
    (from (1, 0) to (0, 1)).

    Parameters
    ----------
    opinions : list[Opinion]
        Two or more opinions.

    Returns
    -------
    float
        Conflict in [0, 1].

    Raises
    ------
    ValueError
        If fewer than two opinions are provided.
    """
    if len(opinions) < 2:
        raise ValueError("conflict_metric requires at least two opinions")

    max_dist = math.sqrt(2.0)
    n = len(opinions)
    total = 0.0
    count = 0

    for i in range(n):
        for j in range(i + 1, n):
            db = opinions[i].b - opinions[j].b
            dd = opinions[i].d - opinions[j].d
            dist = math.sqrt(db * db + dd * dd)
            total += dist / max_dist
            count += 1

    return total / count
