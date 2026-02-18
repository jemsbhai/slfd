"""Subjective Logic Opinion dataclass.

An Opinion ω = (b, d, u, a) represents an epistemic state where:
    b = belief (evidence FOR the proposition)
    d = disbelief (evidence AGAINST the proposition)
    u = uncertainty (lack of evidence)
    a = base rate (prior probability absent evidence)

Constraint: b + d + u = 1, all components ∈ [0, 1], a ∈ [0, 1]

References:
    Jøsang, A. (2016). Subjective Logic: A Formalism for Reasoning Under
    Uncertainty. Springer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Tolerance for floating-point comparison of the sum constraint
_SUM_TOL = 1e-9
_VACUOUS_TOL = 1e-9
_DOGMATIC_TOL = 1e-9

# Default non-informative prior weight (W=2 is standard in SL literature)
_DEFAULT_PRIOR_WEIGHT = 2.0


def _validate_unit(value: float, name: str) -> None:
    """Raise ValueError if value is not in [0, 1]."""
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")


@dataclass(frozen=True, slots=True)
class Opinion:
    """Immutable Subjective Logic opinion.

    Parameters
    ----------
    b : float
        Belief mass — evidence supporting the proposition.
    d : float
        Disbelief mass — evidence against the proposition.
    u : float
        Uncertainty mass — lack of evidence.
    a : float
        Base rate — prior probability in the absence of evidence.
    """

    b: float
    d: float
    u: float
    a: float

    def __post_init__(self) -> None:
        _validate_unit(self.b, "b (belief)")
        _validate_unit(self.d, "d (disbelief)")
        _validate_unit(self.u, "u (uncertainty)")
        _validate_unit(self.a, "a (base rate)")

        total = self.b + self.d + self.u
        if abs(total - 1.0) > _SUM_TOL:
            raise ValueError(
                f"b + d + u must sum to 1 (within tolerance {_SUM_TOL}), "
                f"got {self.b} + {self.d} + {self.u} = {total}"
            )

    # -------------------------------------------------------------------
    # Derived properties
    # -------------------------------------------------------------------

    @property
    def expected_probability(self) -> float:
        """Projected probability: P(x) = b + a·u."""
        return self.b + self.a * self.u

    @property
    def is_vacuous(self) -> bool:
        """True if opinion represents complete ignorance (u ≈ 1)."""
        return self.u >= 1.0 - _VACUOUS_TOL

    @property
    def is_dogmatic(self) -> bool:
        """True if opinion has no uncertainty (u ≈ 0)."""
        return self.u <= _DOGMATIC_TOL

    # -------------------------------------------------------------------
    # Factory methods
    # -------------------------------------------------------------------

    @classmethod
    def from_confidence(
        cls,
        probability: float,
        uncertainty: float,
        base_rate: float,
    ) -> Opinion:
        """Construct an opinion from an ML model's probability + uncertainty estimate.

        Parameters
        ----------
        probability : float
            Model's predicted probability for the proposition, ∈ [0, 1].
        uncertainty : float
            Estimated uncertainty (e.g. from MC Dropout, ensemble variance), ∈ [0, 1].
        base_rate : float
            Prior probability absent evidence, ∈ [0, 1].

        Returns
        -------
        Opinion
            b = probability * (1 - uncertainty)
            d = (1 - probability) * (1 - uncertainty)
            u = uncertainty
            a = base_rate
        """
        _validate_unit(probability, "probability")
        _validate_unit(uncertainty, "uncertainty")

        evidence_mass = 1.0 - uncertainty
        b = probability * evidence_mass
        d = (1.0 - probability) * evidence_mass
        u = uncertainty
        return cls(b=b, d=d, u=u, a=base_rate)

    @classmethod
    def from_evidence(
        cls,
        positive: int | float,
        negative: int | float,
        base_rate: float,
        prior_weight: float = _DEFAULT_PRIOR_WEIGHT,
    ) -> Opinion:
        """Construct an opinion from positive/negative evidence counts.

        Uses the standard SL mapping from evidence to opinion:
            b = r / (r + s + W)
            d = s / (r + s + W)
            u = W / (r + s + W)

        where r = positive evidence, s = negative evidence, W = prior weight.

        Parameters
        ----------
        positive : int or float
            Count of positive evidence observations (≥ 0).
        negative : int or float
            Count of negative evidence observations (≥ 0).
        base_rate : float
            Prior probability absent evidence, ∈ [0, 1].
        prior_weight : float
            Non-informative prior weight (default 2.0, standard in SL).

        Returns
        -------
        Opinion
        """
        if positive < 0:
            raise ValueError(f"positive must be ≥ 0, got {positive}")
        if negative < 0:
            raise ValueError(f"negative must be ≥ 0, got {negative}")
        if prior_weight <= 0:
            raise ValueError(f"prior_weight must be > 0, got {prior_weight}")

        total = positive + negative + prior_weight
        b = positive / total
        d = negative / total
        u = prior_weight / total
        return cls(b=b, d=d, u=u, a=base_rate)

    # -------------------------------------------------------------------
    # Numpy interop
    # -------------------------------------------------------------------

    def to_array(self) -> np.ndarray:
        """Convert to numpy array [b, d, u, a]."""
        return np.array([self.b, self.d, self.u, self.a], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> Opinion:
        """Construct from numpy array [b, d, u, a].

        Parameters
        ----------
        arr : np.ndarray
            Array of shape (4,) containing [b, d, u, a].

        Returns
        -------
        Opinion
        """
        if arr.shape != (4,):
            raise ValueError(f"Expected array of shape (4,), got {arr.shape}")
        return cls(b=float(arr[0]), d=float(arr[1]), u=float(arr[2]), a=float(arr[3]))

    # -------------------------------------------------------------------
    # Representation
    # -------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"Opinion(b={self.b:.4f}, d={self.d:.4f}, u={self.u:.4f}, a={self.a:.4f})"
