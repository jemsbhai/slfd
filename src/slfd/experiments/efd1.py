"""E-FD1: Scalar Collapse Diagnostic.

Demonstrates that scalar fraud scores conflate distinct epistemic states
that Subjective Logic opinions can separate.

Three clusters are constructed with SIMILAR scalar score averages (~0.5)
but DIFFERENT underlying signal patterns:

    Cluster A (Ambiguity):  All sources moderately suspicious.
        → Moderate b, moderate d, moderate u.
        → Optimal action: could go either way, needs threshold tuning.

    Cluster B (Conflict):   Some sources highly suspicious, others legitimate.
        → Scalar average ~0.5, but high conflict between sources.
        → Optimal action: ESCALATE (investigate the disagreement).

    Cluster C (Ignorance):  Insufficient data / new customer.
        → Scalar average ~0.5 due to uninformative scores, high uncertainty.
        → Optimal action: ESCALATE (gather more data).

A scalar system sees all three clusters as "score ≈ 0.5" and treats them
identically. The SL opinion representation distinguishes them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import silhouette_score

from slfd.opinion import Opinion
from slfd.fusion import cumulative_fuse, conflict_metric
from slfd.decision import Decision, ThreeWayDecider


# Number of signal sources in the scenario
_N_SOURCES = 4


# ===================================================================
# Data containers
# ===================================================================

@dataclass
class ScalarCollapseScenario:
    """Container for E-FD1 scenario data.

    Attributes
    ----------
    cluster_labels : np.ndarray
        Cluster assignment per transaction. 0=Ambiguity, 1=Conflict, 2=Ignorance.
    scalar_averages : np.ndarray
        Mean scalar score across sources per transaction, shape (n,).
    per_source_opinions : list[list[Opinion]]
        Per-source opinions, [n_transactions][n_sources].
    fused_opinions : list[Opinion]
        Cumulatively fused opinions per transaction.
    n_per_cluster : int
        Number of transactions per cluster.
    seed : int
        Random seed used.
    """

    cluster_labels: np.ndarray
    scalar_averages: np.ndarray
    per_source_opinions: list[list[Opinion]]
    fused_opinions: list[Opinion]
    n_per_cluster: int
    seed: int


@dataclass
class ClusterStats:
    """Per-cluster summary statistics for reporting."""

    label: str
    mean_b: float
    mean_d: float
    mean_u: float
    mean_conflict: float


@dataclass
class EFD1Results:
    """Results of the E-FD1 separation analysis.

    Attributes
    ----------
    scalar_silhouette : float
        Silhouette score in scalar average space (1D).
    opinion_silhouette : float
        Silhouette score in opinion (b, d, u) space (3D).
    cluster_stats : list[ClusterStats]
        Per-cluster mean opinion components.
    """

    scalar_silhouette: float
    opinion_silhouette: float
    cluster_stats: list[ClusterStats]


# ===================================================================
# Scenario construction
# ===================================================================

def _build_cluster_a(
    rng: np.random.Generator,
    n: int,
    base_rate: float,
) -> tuple[np.ndarray, list[list[Opinion]]]:
    """Cluster A (Ambiguity): all sources moderately suspicious.

    Each source produces scores near 0.5 with low uncertainty.
    The scalar average is ~0.5, and opinions have moderate b, d, u.
    """
    # All sources give scores centered at 0.5 with moderate spread
    scores = np.zeros((n, _N_SOURCES))
    opinions: list[list[Opinion]] = []

    for i in range(n):
        txn_opinions: list[Opinion] = []
        for s in range(_N_SOURCES):
            # Moderate score with some noise
            p = np.clip(rng.normal(0.5, 0.08), 0.05, 0.95)
            scores[i, s] = p
            # Low uncertainty — the source is confident, just ambiguous
            o = Opinion.from_confidence(
                probability=p, uncertainty=0.15, base_rate=base_rate,
            )
            txn_opinions.append(o)
        opinions.append(txn_opinions)

    return scores, opinions


def _build_cluster_b(
    rng: np.random.Generator,
    n: int,
    base_rate: float,
) -> tuple[np.ndarray, list[list[Opinion]]]:
    """Cluster B (Conflict): sources strongly disagree.

    Half the sources say "fraud" (high score), half say "legit" (low score).
    Scalar average is ~0.5, but conflict is high.
    """
    scores = np.zeros((n, _N_SOURCES))
    opinions: list[list[Opinion]] = []

    n_high = _N_SOURCES // 2
    n_low = _N_SOURCES - n_high

    for i in range(n):
        txn_opinions: list[Opinion] = []
        for s in range(_N_SOURCES):
            if s < n_high:
                # Pro-fraud source: high score
                p = np.clip(rng.normal(0.85, 0.07), 0.55, 0.99)
            else:
                # Pro-legit source: low score
                p = np.clip(rng.normal(0.15, 0.07), 0.01, 0.45)
            scores[i, s] = p
            # Low uncertainty — each source is confident in its (opposing) view
            o = Opinion.from_confidence(
                probability=p, uncertainty=0.12, base_rate=base_rate,
            )
            txn_opinions.append(o)
        opinions.append(txn_opinions)

    return scores, opinions


def _build_cluster_c(
    rng: np.random.Generator,
    n: int,
    base_rate: float,
) -> tuple[np.ndarray, list[list[Opinion]]]:
    """Cluster C (Ignorance): insufficient evidence.

    Sources produce scores near 0.5 but with very high uncertainty
    (new customer, sparse data). Scalar average ~0.5.
    """
    scores = np.zeros((n, _N_SOURCES))
    opinions: list[list[Opinion]] = []

    for i in range(n):
        txn_opinions: list[Opinion] = []
        for s in range(_N_SOURCES):
            # Uninformative score near 0.5
            p = np.clip(rng.normal(0.5, 0.10), 0.05, 0.95)
            scores[i, s] = p
            # Very high uncertainty — source has little evidence
            o = Opinion.from_confidence(
                probability=p, uncertainty=0.85, base_rate=base_rate,
            )
            txn_opinions.append(o)
        opinions.append(txn_opinions)

    return scores, opinions


def build_scenario(
    n_per_cluster: int = 300,
    seed: int = 42,
    base_rate: float = 0.5,
) -> ScalarCollapseScenario:
    """Build the three-cluster scalar collapse scenario.

    Parameters
    ----------
    n_per_cluster : int
        Number of transactions per cluster.
    seed : int
        Random seed for reproducibility.
    base_rate : float
        Base fraud rate for opinion construction.

    Returns
    -------
    ScalarCollapseScenario
    """
    rng = np.random.default_rng(seed)

    # Build each cluster
    scores_a, opinions_a = _build_cluster_a(rng, n_per_cluster, base_rate)
    scores_b, opinions_b = _build_cluster_b(rng, n_per_cluster, base_rate)
    scores_c, opinions_c = _build_cluster_c(rng, n_per_cluster, base_rate)

    # Concatenate
    all_scores = np.vstack([scores_a, scores_b, scores_c])
    all_opinions = opinions_a + opinions_b + opinions_c
    labels = np.array(
        [0] * n_per_cluster + [1] * n_per_cluster + [2] * n_per_cluster,
        dtype=np.int32,
    )

    # Scalar averages (what deployed systems use)
    scalar_averages = np.mean(all_scores, axis=1)

    # Fuse per-source opinions into a single opinion per transaction
    fused_opinions: list[Opinion] = []
    for txn_opinions in all_opinions:
        fused = cumulative_fuse(txn_opinions)
        fused_opinions.append(fused)

    return ScalarCollapseScenario(
        cluster_labels=labels,
        scalar_averages=scalar_averages,
        per_source_opinions=all_opinions,
        fused_opinions=fused_opinions,
        n_per_cluster=n_per_cluster,
        seed=seed,
    )


# ===================================================================
# Separation analysis
# ===================================================================

def analyze_separation(scenario: ScalarCollapseScenario) -> EFD1Results:
    """Measure cluster separability in scalar vs. opinion space.

    Computes silhouette scores and per-cluster statistics.

    Parameters
    ----------
    scenario : ScalarCollapseScenario
        The constructed scenario.

    Returns
    -------
    EFD1Results
    """
    labels = scenario.cluster_labels

    # --- Scalar space: 1D (average score) ---
    X_scalar = scenario.scalar_averages.reshape(-1, 1)
    sil_scalar = silhouette_score(X_scalar, labels)

    # --- Opinion space: 3D (b, d, u) ---
    X_opinion = np.array([
        [o.b, o.d, o.u] for o in scenario.fused_opinions
    ])
    sil_opinion = silhouette_score(X_opinion, labels)

    # --- Per-cluster stats ---
    cluster_names = ["Ambiguity", "Conflict", "Ignorance"]
    cluster_stats: list[ClusterStats] = []

    for c in range(3):
        mask = labels == c
        cluster_opinions = [scenario.fused_opinions[i] for i in range(len(labels)) if mask[i]]
        cluster_source_opinions = [
            scenario.per_source_opinions[i] for i in range(len(labels)) if mask[i]
        ]

        mean_b = float(np.mean([o.b for o in cluster_opinions]))
        mean_d = float(np.mean([o.d for o in cluster_opinions]))
        mean_u = float(np.mean([o.u for o in cluster_opinions]))
        mean_conflict = float(np.mean([
            conflict_metric(src_ops) for src_ops in cluster_source_opinions
        ]))

        cluster_stats.append(ClusterStats(
            label=cluster_names[c],
            mean_b=mean_b,
            mean_d=mean_d,
            mean_u=mean_u,
            mean_conflict=mean_conflict,
        ))

    return EFD1Results(
        scalar_silhouette=sil_scalar,
        opinion_silhouette=sil_opinion,
        cluster_stats=cluster_stats,
    )


# ===================================================================
# Decision analysis
# ===================================================================

def analyze_decisions(
    scenario: ScalarCollapseScenario,
) -> list[dict[Decision, float]]:
    """Analyze the decision distribution per cluster.

    Uses a standard ThreeWayDecider to show that different epistemic
    states lead to different optimal actions.

    Parameters
    ----------
    scenario : ScalarCollapseScenario

    Returns
    -------
    list[dict[Decision, float]]
        Per-cluster distribution over decisions (fractions summing to 1).
    """
    # Thresholds chosen so that:
    # - Cluster A (confident, agreeing sources, b≈d≈0.48, u≈0.04) auto-decides
    # - Cluster B (high conflict) escalates via conflict override
    # - Cluster C (high uncertainty) escalates via uncertainty override
    decider = ThreeWayDecider(
        block_threshold=0.40,
        approve_threshold=0.40,
        escalate_uncertainty=0.35,
        escalate_conflict=0.25,
    )

    labels = scenario.cluster_labels
    distributions: list[dict[Decision, float]] = []

    for c in range(3):
        mask = labels == c
        n_cluster = int(mask.sum())
        counts = {Decision.BLOCK: 0, Decision.APPROVE: 0, Decision.ESCALATE: 0}

        for i in range(len(labels)):
            if not mask[i]:
                continue
            opinion = scenario.fused_opinions[i]
            conf = conflict_metric(scenario.per_source_opinions[i])
            result = decider.decide(opinion, conflict=conf)
            counts[result.decision] += 1

        # Normalize to fractions
        dist = {d: counts[d] / n_cluster for d in Decision}
        distributions.append(dist)

    return distributions
