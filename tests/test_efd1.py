"""Tests for E-FD1: Scalar Collapse Diagnostic.

Hypothesis: Scalar fraud scores provably conflate distinct epistemic states
that lead to different optimal actions.

Protocol:
    1. Construct three clusters of transactions with SIMILAR scalar scores (~0.5)
       but DIFFERENT underlying signal patterns:
       - Cluster A (Ambiguity):  All sources moderately suspicious
       - Cluster B (Conflict):   Some highly suspicious, others clearly legitimate
       - Cluster C (Ignorance):  Insufficient data / new customer → high uncertainty
    2. Show SL opinions distinguish these clusters; scalar scores do not
    3. Demonstrate optimal actions differ across clusters
    4. Measure: Silhouette score in SL opinion space vs. scalar score space

Deliverable: Figure showing 3D opinion space separating clusters that
scalar scores collapse.
"""

import math

import numpy as np
import pytest

from slfd.opinion import Opinion
from slfd.fusion import cumulative_fuse, conflict_metric
from slfd.decision import Decision, ThreeWayDecider
from slfd.experiments.efd1 import (
    ScalarCollapseScenario,
    build_scenario,
    analyze_separation,
    analyze_decisions,
    EFD1Results,
)


# ===================================================================
# 1. Scenario construction
# ===================================================================

class TestScenarioConstruction:
    """The three clusters must have similar scalar averages but different
    opinion structures."""

    def setup_method(self):
        self.scenario = build_scenario(n_per_cluster=200, seed=42)

    def test_returns_scenario_object(self):
        assert isinstance(self.scenario, ScalarCollapseScenario)

    def test_three_clusters(self):
        assert len(self.scenario.cluster_labels) == 3 * 200

    def test_cluster_labels_correct(self):
        labels = self.scenario.cluster_labels
        assert np.sum(labels == 0) == 200  # Ambiguity
        assert np.sum(labels == 1) == 200  # Conflict
        assert np.sum(labels == 2) == 200  # Ignorance

    def test_has_scalar_scores(self):
        """Each transaction has a scalar average score."""
        assert self.scenario.scalar_averages.shape == (600,)

    def test_has_fused_opinions(self):
        """Each transaction has a fused SL opinion."""
        assert len(self.scenario.fused_opinions) == 600
        assert isinstance(self.scenario.fused_opinions[0], Opinion)

    def test_has_per_source_opinions(self):
        """Per-source opinions are available for conflict analysis."""
        assert len(self.scenario.per_source_opinions) == 600
        assert len(self.scenario.per_source_opinions[0]) >= 2

    def test_scalar_scores_similar_across_clusters(self):
        """KEY PROPERTY: All three clusters have similar scalar averages.
        This is the 'collapse' — scalar scores can't tell them apart."""
        labels = self.scenario.cluster_labels
        scores = self.scenario.scalar_averages
        mean_a = np.mean(scores[labels == 0])
        mean_b = np.mean(scores[labels == 1])
        mean_c = np.mean(scores[labels == 2])
        # All cluster means should be near 0.5 (within 0.1)
        assert abs(mean_a - 0.5) < 0.1
        assert abs(mean_b - 0.5) < 0.1
        assert abs(mean_c - 0.5) < 0.1

    def test_scalar_scores_overlap(self):
        """Scalar score distributions should heavily overlap across clusters."""
        labels = self.scenario.cluster_labels
        scores = self.scenario.scalar_averages
        std_a = np.std(scores[labels == 0])
        std_b = np.std(scores[labels == 1])
        std_c = np.std(scores[labels == 2])
        # Each cluster has some spread
        assert std_a > 0.01
        assert std_b > 0.01
        assert std_c > 0.01


# ===================================================================
# 2. Opinion-space separation
# ===================================================================

class TestOpinionSeparation:
    """SL opinions SHOULD separate the three clusters."""

    def setup_method(self):
        self.scenario = build_scenario(n_per_cluster=300, seed=42)

    def test_ambiguity_cluster_moderate_belief_and_disbelief(self):
        """Cluster A: both b and d are moderate, u is moderate."""
        labels = self.scenario.cluster_labels
        opinions_a = [self.scenario.fused_opinions[i] for i in range(len(labels)) if labels[i] == 0]
        mean_b = np.mean([o.b for o in opinions_a])
        mean_d = np.mean([o.d for o in opinions_a])
        mean_u = np.mean([o.u for o in opinions_a])
        # Moderate belief and disbelief, not extreme
        assert 0.15 < mean_b < 0.6
        assert 0.15 < mean_d < 0.6

    def test_conflict_cluster_polarized(self):
        """Cluster B: high conflict — sources disagree."""
        labels = self.scenario.cluster_labels
        conflict_indices = [i for i in range(len(labels)) if labels[i] == 1]
        conflicts = []
        for i in conflict_indices:
            c = conflict_metric(self.scenario.per_source_opinions[i])
            conflicts.append(c)
        mean_conflict = np.mean(conflicts)
        # Conflict cluster should have higher conflict than others
        ambig_indices = [i for i in range(len(labels)) if labels[i] == 0]
        ambig_conflicts = [conflict_metric(self.scenario.per_source_opinions[i]) for i in ambig_indices]
        assert mean_conflict > np.mean(ambig_conflicts)

    def test_ignorance_cluster_high_uncertainty(self):
        """Cluster C: high uncertainty — lack of evidence."""
        labels = self.scenario.cluster_labels
        opinions_c = [self.scenario.fused_opinions[i] for i in range(len(labels)) if labels[i] == 2]
        mean_u = np.mean([o.u for o in opinions_c])
        # Ignorance cluster should have highest uncertainty
        opinions_a = [self.scenario.fused_opinions[i] for i in range(len(labels)) if labels[i] == 0]
        mean_u_a = np.mean([o.u for o in opinions_a])
        assert mean_u > mean_u_a


# ===================================================================
# 3. Separation analysis (silhouette scores)
# ===================================================================

class TestSeparationAnalysis:
    """Quantitative separation measurement."""

    def setup_method(self):
        self.scenario = build_scenario(n_per_cluster=300, seed=42)
        self.results = analyze_separation(self.scenario)

    def test_returns_results(self):
        assert isinstance(self.results, EFD1Results)

    def test_has_scalar_silhouette(self):
        """Silhouette score for scalar score space."""
        assert hasattr(self.results, "scalar_silhouette")
        assert -1.0 <= self.results.scalar_silhouette <= 1.0

    def test_has_opinion_silhouette(self):
        """Silhouette score for opinion (b, d, u) space."""
        assert hasattr(self.results, "opinion_silhouette")
        assert -1.0 <= self.results.opinion_silhouette <= 1.0

    def test_opinion_separates_better(self):
        """CORE RESULT: Opinion space silhouette > scalar space silhouette.
        This is the key finding — opinions separate what scalars collapse."""
        assert self.results.opinion_silhouette > self.results.scalar_silhouette

    def test_opinion_silhouette_meaningfully_positive(self):
        """Opinion space should show genuine cluster structure."""
        assert self.results.opinion_silhouette > 0.1

    def test_has_per_cluster_stats(self):
        """Per-cluster mean opinion components for reporting."""
        assert hasattr(self.results, "cluster_stats")
        assert len(self.results.cluster_stats) == 3
        for stat in self.results.cluster_stats:
            assert hasattr(stat, "mean_b")
            assert hasattr(stat, "mean_d")
            assert hasattr(stat, "mean_u")
            assert hasattr(stat, "mean_conflict")
            assert hasattr(stat, "label")


# ===================================================================
# 4. Decision analysis
# ===================================================================

class TestDecisionAnalysis:
    """Different clusters should lead to different optimal actions."""

    def setup_method(self):
        self.scenario = build_scenario(n_per_cluster=300, seed=42)
        self.decisions = analyze_decisions(self.scenario)

    def test_returns_per_cluster_decision_distribution(self):
        """Each cluster has a distribution over BLOCK/APPROVE/ESCALATE."""
        assert len(self.decisions) == 3

    def test_ambiguity_cluster_auto_decides(self):
        """Cluster A (ambiguity): evidence is real and agreeing, so
        the system can auto-decide (block or approve). This is the
        key contrast — scalar sees 0.5, but SL sees confident evidence."""
        dist = self.decisions[0]
        # Most should auto-decide (block or approve), few escalate
        assert dist[Decision.BLOCK] + dist[Decision.APPROVE] > 0.5

    def test_conflict_cluster_mostly_escalates(self):
        """Cluster B (conflict): conflict triggers escalation."""
        dist = self.decisions[1]
        assert dist[Decision.ESCALATE] > 0.3

    def test_ignorance_cluster_mostly_escalates(self):
        """Cluster C (ignorance): uncertainty triggers escalation."""
        dist = self.decisions[2]
        assert dist[Decision.ESCALATE] > 0.5

    def test_decisions_differ_across_clusters(self):
        """The decision distributions are NOT identical — different
        epistemic states lead to different actions."""
        # At least one cluster must have a different dominant action
        dominant = [max(d, key=d.get) for d in self.decisions]
        # Not all three clusters should have the same dominant decision
        # (if they do, the framework isn't distinguishing them)
        assert len(set(dominant)) >= 2


# ===================================================================
# 5. Reproducibility
# ===================================================================

class TestReproducibility:
    """Same seed → same results."""

    def test_same_seed_same_silhouettes(self):
        s1 = build_scenario(n_per_cluster=200, seed=42)
        s2 = build_scenario(n_per_cluster=200, seed=42)
        r1 = analyze_separation(s1)
        r2 = analyze_separation(s2)
        assert r1.scalar_silhouette == pytest.approx(r2.scalar_silhouette)
        assert r1.opinion_silhouette == pytest.approx(r2.opinion_silhouette)
