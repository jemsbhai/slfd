"""Synthetic fraud scenario generator for controlled ablation studies.

Produces multi-source fraud detection scenarios with tunable parameters:
    - Transaction count, fraud rate, number of signal sources
    - Per-source accuracy (how well each source separates fraud/legit)
    - Adversarial/corrupted source injection
    - Temporal drift (fraud rate shift mid-dataset)
    - Signal staleness (per-source age → higher uncertainty)
    - Reproducible via seed

Output is a FraudDataset containing ground truth labels, per-source
scalar scores, per-source Opinion objects, and metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from slfd.opinion import Opinion

# Default source accuracies for a 4-source ensemble
_DEFAULT_ACCURACIES = [0.85, 0.90, 0.75, 0.95]
_VALID_ADVERSARIAL_STRATEGIES = {"invert", "random"}

# Default staleness decay half-life (hours) for opinion construction
_STALENESS_HALF_LIFE = 12.0


@dataclass
class FraudDataset:
    """Container for a synthetic fraud detection scenario.

    Attributes
    ----------
    labels : np.ndarray
        Ground truth fraud labels, shape (n_transactions,). 1=fraud, 0=legit.
    scalar_scores : np.ndarray
        Per-source fraud scores, shape (n_transactions, n_sources). Each ∈ [0, 1].
    opinions : list[list[Opinion]]
        Per-source Opinion objects, [n_transactions][n_sources].
    timestamps : np.ndarray
        Per-transaction timestamps (float, arbitrary units), shape (n_transactions,).
    metadata : dict[str, Any]
        Reproducibility metadata (seed, parameters, etc.).
    """

    labels: np.ndarray
    scalar_scores: np.ndarray
    opinions: list[list[Opinion]]
    timestamps: np.ndarray
    metadata: dict[str, Any]

    @property
    def n_transactions(self) -> int:
        return len(self.labels)

    @property
    def n_sources(self) -> int:
        return self.scalar_scores.shape[1]


class FraudScenarioGenerator:
    """Generate synthetic fraud detection scenarios with controlled properties.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def generate(
        self,
        n_transactions: int = 10000,
        fraud_rate: float = 0.02,
        n_sources: int = 4,
        source_accuracies: list[float] | None = None,
        n_adversarial: int = 0,
        adversarial_strategy: str = "invert",
        temporal_drift: bool = False,
        drift_point: float = 0.5,
        drift_fraud_rate: float | None = None,
        signal_staleness: list[float] | None = None,
    ) -> FraudDataset:
        """Generate a synthetic fraud scenario.

        Parameters
        ----------
        n_transactions : int
            Number of transactions to generate.
        fraud_rate : float
            Fraction of transactions that are fraudulent, ∈ [0, 1].
        n_sources : int
            Number of signal sources.
        source_accuracies : list[float] or None
            Per-source accuracy ∈ (0, 1]. Length must equal n_sources.
            None → use defaults (clipped/extended to n_sources).
        n_adversarial : int
            Number of sources to corrupt (last N sources).
        adversarial_strategy : str
            How to corrupt adversarial sources: "invert" or "random".
        temporal_drift : bool
            If True, fraud rate changes at drift_point.
        drift_point : float
            Fraction of dataset where drift occurs, ∈ (0, 1).
        drift_fraud_rate : float or None
            Post-drift fraud rate. Required if temporal_drift is True.
        signal_staleness : list[float] or None
            Per-source staleness in hours. Length must equal n_sources.
            Higher → more uncertainty in generated opinions.

        Returns
        -------
        FraudDataset
        """
        rng = np.random.default_rng(self.seed)

        # --- Validate inputs ---
        self._validate_inputs(
            fraud_rate, n_sources, source_accuracies, n_adversarial,
            adversarial_strategy, signal_staleness,
        )

        # --- Resolve source accuracies ---
        if source_accuracies is None:
            accuracies = _DEFAULT_ACCURACIES[:n_sources]
            if len(accuracies) < n_sources:
                accuracies = accuracies + [0.85] * (n_sources - len(accuracies))
        else:
            accuracies = list(source_accuracies)

        # --- Generate labels ---
        labels = self._generate_labels(
            rng, n_transactions, fraud_rate,
            temporal_drift, drift_point, drift_fraud_rate,
        )

        # --- Generate timestamps ---
        timestamps = np.sort(rng.uniform(0.0, float(n_transactions), size=n_transactions))

        # --- Generate scalar scores ---
        scalar_scores = self._generate_scores(
            rng, labels, n_sources, accuracies,
            n_adversarial, adversarial_strategy,
        )

        # --- Determine adversarial indices ---
        adversarial_indices = list(range(n_sources - n_adversarial, n_sources))

        # --- Construct opinions ---
        # Use the requested fraud_rate as the prior base rate, not the
        # realized sample rate — the base rate is a prior belief, not
        # an empirical observation.
        opinions = self._construct_opinions(
            scalar_scores, fraud_rate, signal_staleness,
        )

        # --- Metadata ---
        metadata: dict[str, Any] = {
            "seed": self.seed,
            "n_transactions": n_transactions,
            "n_sources": n_sources,
            "fraud_rate": fraud_rate,
            "source_accuracies": accuracies,
            "n_adversarial": n_adversarial,
            "adversarial_strategy": adversarial_strategy if n_adversarial > 0 else None,
            "adversarial_indices": adversarial_indices,
            "temporal_drift": temporal_drift,
        }
        if signal_staleness is not None:
            metadata["signal_staleness"] = signal_staleness

        return FraudDataset(
            labels=labels,
            scalar_scores=scalar_scores,
            opinions=opinions,
            timestamps=timestamps,
            metadata=metadata,
        )

    # -------------------------------------------------------------------
    # Internal methods
    # -------------------------------------------------------------------

    @staticmethod
    def _validate_inputs(
        fraud_rate: float,
        n_sources: int,
        source_accuracies: list[float] | None,
        n_adversarial: int,
        adversarial_strategy: str,
        signal_staleness: list[float] | None,
    ) -> None:
        if fraud_rate < 0.0 or fraud_rate > 1.0:
            raise ValueError(f"fraud_rate must be in [0, 1], got {fraud_rate}")

        if source_accuracies is not None:
            if len(source_accuracies) != n_sources:
                raise ValueError(
                    f"source_accuracies length ({len(source_accuracies)}) "
                    f"must match n_sources ({n_sources})"
                )
            for acc in source_accuracies:
                if acc < 0.0 or acc > 1.0:
                    raise ValueError(
                        f"source accuracy must be in [0, 1], got {acc}"
                    )

        if n_adversarial > n_sources:
            raise ValueError(
                f"n_adversarial ({n_adversarial}) cannot exceed "
                f"n_sources ({n_sources})"
            )

        if adversarial_strategy not in _VALID_ADVERSARIAL_STRATEGIES:
            raise ValueError(
                f"adversarial_strategy must be one of {_VALID_ADVERSARIAL_STRATEGIES}, "
                f"got '{adversarial_strategy}'"
            )

        if signal_staleness is not None and len(signal_staleness) != n_sources:
            raise ValueError(
                f"signal_staleness length ({len(signal_staleness)}) "
                f"must match n_sources ({n_sources})"
            )

    @staticmethod
    def _generate_labels(
        rng: np.random.Generator,
        n_transactions: int,
        fraud_rate: float,
        temporal_drift: bool,
        drift_point: float,
        drift_fraud_rate: float | None,
    ) -> np.ndarray:
        """Generate ground truth labels, optionally with temporal drift."""
        if not temporal_drift:
            return rng.binomial(1, fraud_rate, size=n_transactions).astype(np.int32)

        split = int(n_transactions * drift_point)
        pre = rng.binomial(1, fraud_rate, size=split)
        post_rate = drift_fraud_rate if drift_fraud_rate is not None else fraud_rate
        post = rng.binomial(1, post_rate, size=n_transactions - split)
        return np.concatenate([pre, post]).astype(np.int32)

    @staticmethod
    def _generate_scores(
        rng: np.random.Generator,
        labels: np.ndarray,
        n_sources: int,
        accuracies: list[float],
        n_adversarial: int,
        adversarial_strategy: str,
    ) -> np.ndarray:
        """Generate per-source scalar scores.

        For each honest source with accuracy `acc`:
            - Fraud transactions: score ~ Beta(α, β) with mean ≈ acc
            - Legit transactions: score ~ Beta(β, α) with mean ≈ 1-acc

        Beta distribution is used to keep scores in [0, 1] with
        controllable concentration.
        """
        n = len(labels)
        scores = np.zeros((n, n_sources), dtype=np.float64)
        fraud_mask = labels == 1
        legit_mask = labels == 0

        # Concentration parameter — controls score spread
        concentration = 10.0

        honest_count = n_sources - n_adversarial

        for s in range(honest_count):
            acc = accuracies[s]
            # Beta parameters: mean = α/(α+β) = acc
            alpha = acc * concentration
            beta = (1.0 - acc) * concentration

            # Fraud → high scores (centered at acc)
            scores[fraud_mask, s] = rng.beta(alpha, beta, size=int(fraud_mask.sum()))
            # Legit → low scores (centered at 1-acc)
            scores[legit_mask, s] = rng.beta(beta, alpha, size=int(legit_mask.sum()))

        # Adversarial sources
        for i in range(n_adversarial):
            s = honest_count + i
            acc = accuracies[s]
            alpha = acc * concentration
            beta = (1.0 - acc) * concentration

            if adversarial_strategy == "invert":
                # Inverted: fraud gets LOW scores, legit gets HIGH
                scores[fraud_mask, s] = rng.beta(beta, alpha, size=int(fraud_mask.sum()))
                scores[legit_mask, s] = rng.beta(alpha, beta, size=int(legit_mask.sum()))
            elif adversarial_strategy == "random":
                # Pure noise: uniform [0, 1]
                scores[:, s] = rng.uniform(0.0, 1.0, size=n)

        return np.clip(scores, 0.0, 1.0)

    @staticmethod
    def _construct_opinions(
        scalar_scores: np.ndarray,
        fraud_rate: float,
        signal_staleness: list[float] | None,
    ) -> list[list[Opinion]]:
        """Convert scalar scores to Opinion objects.

        Each score p is mapped to an opinion with moderate uncertainty.
        Staleness increases uncertainty via exponential decay.
        """
        n_transactions, n_sources = scalar_scores.shape
        opinions: list[list[Opinion]] = []

        # Base uncertainty for each source (reflecting model calibration)
        base_uncertainty = 0.15

        # Compute per-source staleness multiplier
        staleness_multipliers = np.ones(n_sources)
        if signal_staleness is not None:
            for s in range(n_sources):
                # λ = 2^(-staleness / half_life)
                lam = 2.0 ** (-signal_staleness[s] / _STALENESS_HALF_LIFE)
                # Higher staleness → lower lambda → more uncertainty
                staleness_multipliers[s] = lam

        for i in range(n_transactions):
            txn_opinions: list[Opinion] = []
            for s in range(n_sources):
                p = float(scalar_scores[i, s])
                # Staleness increases effective uncertainty
                effective_evidence = (1.0 - base_uncertainty) * staleness_multipliers[s]
                u = 1.0 - effective_evidence
                b = p * effective_evidence
                d = (1.0 - p) * effective_evidence

                txn_opinions.append(Opinion(b=b, d=d, u=u, a=fraud_rate))
            opinions.append(txn_opinions)

        return opinions
