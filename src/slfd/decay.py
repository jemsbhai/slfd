"""Temporal decay of opinions.

Stale evidence should degrade toward uncertainty rather than retaining
false precision. The decay operator moves an opinion toward vacuous
(b=0, d=0, u=1) as time passes, controlled by a half-life parameter.

Decay model:
    λ(Δt) = 2^(-Δt / half_life)
    b_decayed = b · λ
    d_decayed = d · λ
    u_decayed = 1 - b_decayed - d_decayed

This preserves the b:d ratio while monotonically increasing uncertainty.

References:
    Jøsang, A. (2016). Subjective Logic, §3.8 (Opinion Aging).
"""

from __future__ import annotations

from slfd.opinion import Opinion


def decay_opinion(
    opinion: Opinion,
    elapsed: float,
    half_life: float,
) -> Opinion:
    """Apply temporal decay to an opinion.

    Evidence (belief and disbelief) decays exponentially toward zero,
    while uncertainty grows to fill the gap. The b:d ratio is preserved.

    Parameters
    ----------
    opinion : Opinion
        The opinion to decay.
    elapsed : float
        Time elapsed since the opinion was formed (≥ 0).
        Units are arbitrary but must match half_life.
    half_life : float
        Time for evidence to halve (> 0). Same units as elapsed.

    Returns
    -------
    Opinion
        The decayed opinion.

    Raises
    ------
    ValueError
        If elapsed < 0 or half_life ≤ 0.
    """
    if elapsed < 0.0:
        raise ValueError(f"elapsed must be ≥ 0, got {elapsed}")
    if half_life <= 0.0:
        raise ValueError(f"half_life must be > 0, got {half_life}")

    lam = 2.0 ** (-elapsed / half_life)

    b_decayed = opinion.b * lam
    d_decayed = opinion.d * lam
    u_decayed = 1.0 - b_decayed - d_decayed

    return Opinion(b=b_decayed, d=d_decayed, u=u_decayed, a=opinion.a)
