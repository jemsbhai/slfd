"""Trust discount operator for source reliability weighting.

Not all signal sources are equally reliable. A well-calibrated ML model
deserves more trust than a simple heuristic. Trust discounting adjusts
an opinion based on how much we trust its source.

Trust discount rule (Jøsang, 2016, Ch. 14):
    Given source opinion ω = (b, d, u, a) and trust t ∈ [0, 1]:
        b_discounted = t · b
        d_discounted = t · d
        u_discounted = 1 - t·b - t·d
        a_discounted = a  (base rate unchanged)

Effect: lower trust pushes the opinion toward vacuous (more uncertainty).
At t=0, we completely distrust the source → vacuous opinion.
At t=1, we fully trust the source → opinion unchanged.

References:
    Jøsang, A. (2016). Subjective Logic, Ch. 14 (Trust Discount).
"""

from __future__ import annotations

from slfd.opinion import Opinion


def trust_discount(opinion: Opinion, trust: float) -> Opinion:
    """Discount an opinion by the trust level in its source.

    Parameters
    ----------
    opinion : Opinion
        The opinion to discount.
    trust : float
        Trust in the source, ∈ [0, 1].
        0 = complete distrust (result is vacuous).
        1 = full trust (result is unchanged).

    Returns
    -------
    Opinion
        The trust-discounted opinion.

    Raises
    ------
    ValueError
        If trust is not in [0, 1].
    """
    if trust < 0.0 or trust > 1.0:
        raise ValueError(f"trust must be in [0, 1], got {trust}")

    b_discounted = trust * opinion.b
    d_discounted = trust * opinion.d
    u_discounted = 1.0 - b_discounted - d_discounted

    return Opinion(b=b_discounted, d=d_discounted, u=u_discounted, a=opinion.a)
