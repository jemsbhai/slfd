"""Tests for the synthetic fraud scenario generator.

The generator produces multi-source fraud detection scenarios with
controllable properties for ablation studies:
    - Configurable number of transactions, fraud rate, sources
    - Per-source accuracy and correlation
    - Adversarial/corrupted source injection
    - Temporal drift simulation
    - Signal staleness per source

The output is a FraudDataset containing:
    - Ground truth labels (is_fraud)
    - Per-source scalar scores (what existing systems produce)
    - Per-source Opinion objects (what our framework produces)
    - Transaction metadata (timestamps, source info)
"""

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from slfd.opinion import Opinion
from slfd.synthetic import FraudScenarioGenerator, FraudDataset


# ===================================================================
# 1. FraudDataset structure
# ===================================================================

class TestFraudDatasetStructure:
    """The dataset object exposes the expected fields."""

    def setup_method(self):
        gen = FraudScenarioGenerator(seed=42)
        self.ds = gen.generate(n_transactions=100, n_sources=4)

    def test_has_labels(self):
        """Ground truth fraud labels."""
        assert hasattr(self.ds, "labels")
        assert len(self.ds.labels) == 100

    def test_labels_are_binary(self):
        assert set(np.unique(self.ds.labels)).issubset({0, 1})

    def test_has_scalar_scores(self):
        """Per-source scalar scores (n_transactions × n_sources)."""
        assert hasattr(self.ds, "scalar_scores")
        assert self.ds.scalar_scores.shape == (100, 4)

    def test_scalar_scores_in_unit_interval(self):
        assert np.all(self.ds.scalar_scores >= 0.0)
        assert np.all(self.ds.scalar_scores <= 1.0)

    def test_has_opinions(self):
        """Per-source Opinion objects (n_transactions × n_sources)."""
        assert hasattr(self.ds, "opinions")
        assert len(self.ds.opinions) == 100
        assert len(self.ds.opinions[0]) == 4
        assert isinstance(self.ds.opinions[0][0], Opinion)

    def test_has_metadata(self):
        """Dataset metadata for reproducibility."""
        assert hasattr(self.ds, "metadata")
        assert "seed" in self.ds.metadata
        assert "n_transactions" in self.ds.metadata
        assert "n_sources" in self.ds.metadata
        assert "fraud_rate" in self.ds.metadata

    def test_has_timestamps(self):
        """Per-transaction timestamps for temporal experiments."""
        assert hasattr(self.ds, "timestamps")
        assert len(self.ds.timestamps) == 100

    def test_timestamps_monotonic(self):
        """Timestamps are in non-decreasing order."""
        assert np.all(np.diff(self.ds.timestamps) >= 0)

    def test_n_transactions_property(self):
        assert self.ds.n_transactions == 100

    def test_n_sources_property(self):
        assert self.ds.n_sources == 4


# ===================================================================
# 2. Fraud rate control
# ===================================================================

class TestFraudRateControl:
    """Generator respects the requested fraud rate."""

    def test_default_fraud_rate(self):
        gen = FraudScenarioGenerator(seed=42)
        ds = gen.generate(n_transactions=10000)
        actual_rate = np.mean(ds.labels)
        assert actual_rate == pytest.approx(0.02, abs=0.01)

    def test_custom_fraud_rate(self):
        gen = FraudScenarioGenerator(seed=42)
        ds = gen.generate(n_transactions=10000, fraud_rate=0.10)
        actual_rate = np.mean(ds.labels)
        assert actual_rate == pytest.approx(0.10, abs=0.02)

    def test_high_fraud_rate(self):
        gen = FraudScenarioGenerator(seed=42)
        ds = gen.generate(n_transactions=10000, fraud_rate=0.50)
        actual_rate = np.mean(ds.labels)
        assert actual_rate == pytest.approx(0.50, abs=0.03)

    def test_zero_fraud_rate(self):
        gen = FraudScenarioGenerator(seed=42)
        ds = gen.generate(n_transactions=1000, fraud_rate=0.0)
        assert np.sum(ds.labels) == 0

    def test_invalid_fraud_rate_above_one(self):
        gen = FraudScenarioGenerator(seed=42)
        with pytest.raises(ValueError):
            gen.generate(n_transactions=100, fraud_rate=1.5)

    def test_invalid_fraud_rate_negative(self):
        gen = FraudScenarioGenerator(seed=42)
        with pytest.raises(ValueError):
            gen.generate(n_transactions=100, fraud_rate=-0.1)


# ===================================================================
# 3. Source accuracy control
# ===================================================================

class TestSourceAccuracy:
    """Each source's accuracy can be configured independently."""

    def test_default_four_sources(self):
        gen = FraudScenarioGenerator(seed=42)
        ds = gen.generate(n_transactions=1000)
        assert ds.n_sources == 4

    def test_custom_source_count(self):
        gen = FraudScenarioGenerator(seed=42)
        ds = gen.generate(n_transactions=100, n_sources=8)
        assert ds.n_sources == 8
        assert ds.scalar_scores.shape[1] == 8

    def test_accurate_source_separates_classes(self):
        """A 95% accurate source should give higher scores to fraud."""
        gen = FraudScenarioGenerator(seed=42)
        ds = gen.generate(
            n_transactions=5000,
            fraud_rate=0.10,
            n_sources=1,
            source_accuracies=[0.95],
        )
        fraud_mask = ds.labels == 1
        legit_mask = ds.labels == 0
        mean_fraud_score = np.mean(ds.scalar_scores[fraud_mask, 0])
        mean_legit_score = np.mean(ds.scalar_scores[legit_mask, 0])
        assert mean_fraud_score > mean_legit_score

    def test_random_source_no_separation(self):
        """A 50% accurate source is basically random — no separation."""
        gen = FraudScenarioGenerator(seed=42)
        ds = gen.generate(
            n_transactions=5000,
            fraud_rate=0.10,
            n_sources=1,
            source_accuracies=[0.50],
        )
        fraud_mask = ds.labels == 1
        legit_mask = ds.labels == 0
        mean_fraud_score = np.mean(ds.scalar_scores[fraud_mask, 0])
        mean_legit_score = np.mean(ds.scalar_scores[legit_mask, 0])
        # Should be close — random source has weak separation
        assert abs(mean_fraud_score - mean_legit_score) < 0.15

    def test_source_accuracies_length_must_match(self):
        gen = FraudScenarioGenerator(seed=42)
        with pytest.raises(ValueError, match="source_accuracies"):
            gen.generate(n_transactions=100, n_sources=4, source_accuracies=[0.8, 0.9])

    def test_source_accuracy_out_of_range(self):
        gen = FraudScenarioGenerator(seed=42)
        with pytest.raises(ValueError):
            gen.generate(n_transactions=100, n_sources=1, source_accuracies=[1.5])


# ===================================================================
# 4. Adversarial source injection
# ===================================================================

class TestAdversarialSources:
    """Corrupted sources for Byzantine robustness experiments (E-FD5)."""

    def test_no_adversarial_by_default(self):
        gen = FraudScenarioGenerator(seed=42)
        ds = gen.generate(n_transactions=100)
        assert ds.metadata.get("n_adversarial", 0) == 0

    def test_adversarial_invert(self):
        """Inverted source gives HIGH scores to LEGITIMATE transactions."""
        gen = FraudScenarioGenerator(seed=42)
        ds = gen.generate(
            n_transactions=5000,
            fraud_rate=0.10,
            n_sources=4,
            source_accuracies=[0.90, 0.90, 0.90, 0.90],
            n_adversarial=1,
            adversarial_strategy="invert",
        )
        # Last source should be adversarial (inverted)
        adv_idx = ds.metadata["adversarial_indices"][0]
        fraud_mask = ds.labels == 1
        legit_mask = ds.labels == 0
        mean_fraud = np.mean(ds.scalar_scores[fraud_mask, adv_idx])
        mean_legit = np.mean(ds.scalar_scores[legit_mask, adv_idx])
        # Inverted: legit scores should be higher than fraud scores
        assert mean_legit > mean_fraud

    def test_adversarial_random(self):
        """Random noise source — no correlation with true labels."""
        gen = FraudScenarioGenerator(seed=42)
        ds = gen.generate(
            n_transactions=5000,
            fraud_rate=0.10,
            n_sources=4,
            source_accuracies=[0.90, 0.90, 0.90, 0.90],
            n_adversarial=1,
            adversarial_strategy="random",
        )
        adv_idx = ds.metadata["adversarial_indices"][0]
        fraud_mask = ds.labels == 1
        legit_mask = ds.labels == 0
        mean_fraud = np.mean(ds.scalar_scores[fraud_mask, adv_idx])
        mean_legit = np.mean(ds.scalar_scores[legit_mask, adv_idx])
        # Random: no meaningful separation
        assert abs(mean_fraud - mean_legit) < 0.15

    def test_adversarial_count_exceeds_sources_rejected(self):
        gen = FraudScenarioGenerator(seed=42)
        with pytest.raises(ValueError, match="n_adversarial"):
            gen.generate(n_transactions=100, n_sources=4, n_adversarial=5)

    def test_invalid_adversarial_strategy(self):
        gen = FraudScenarioGenerator(seed=42)
        with pytest.raises(ValueError, match="adversarial_strategy"):
            gen.generate(
                n_transactions=100, n_sources=4,
                n_adversarial=1, adversarial_strategy="unknown",
            )


# ===================================================================
# 5. Opinion construction
# ===================================================================

class TestOpinionConstruction:
    """Opinions are constructed from scalar scores with uncertainty."""

    def test_opinions_are_valid(self):
        """Every generated Opinion satisfies constraints."""
        gen = FraudScenarioGenerator(seed=42)
        ds = gen.generate(n_transactions=200, n_sources=4)
        for txn_opinions in ds.opinions:
            for o in txn_opinions:
                assert math.isclose(o.b + o.d + o.u, 1.0, rel_tol=1e-6)
                assert 0.0 <= o.b <= 1.0
                assert 0.0 <= o.d <= 1.0
                assert 0.0 <= o.u <= 1.0

    def test_high_score_means_high_belief(self):
        """Source giving a high fraud score → opinion with high belief."""
        gen = FraudScenarioGenerator(seed=42)
        ds = gen.generate(
            n_transactions=1000,
            n_sources=1,
            source_accuracies=[0.95],
            fraud_rate=0.50,
        )
        # Find transactions where source gave high scalar score
        high_score_mask = ds.scalar_scores[:, 0] > 0.8
        if np.any(high_score_mask):
            high_score_idx = np.where(high_score_mask)[0][0]
            o = ds.opinions[high_score_idx][0]
            assert o.b > o.d

    def test_opinions_have_fraud_base_rate(self):
        """Opinions should reflect the dataset fraud rate as base rate."""
        gen = FraudScenarioGenerator(seed=42)
        ds = gen.generate(n_transactions=100, fraud_rate=0.05)
        o = ds.opinions[0][0]
        assert o.a == pytest.approx(0.05)


# ===================================================================
# 6. Reproducibility
# ===================================================================

class TestReproducibility:
    """Same seed → same output."""

    def test_same_seed_same_labels(self):
        ds1 = FraudScenarioGenerator(seed=42).generate(n_transactions=500)
        ds2 = FraudScenarioGenerator(seed=42).generate(n_transactions=500)
        np.testing.assert_array_equal(ds1.labels, ds2.labels)

    def test_same_seed_same_scores(self):
        ds1 = FraudScenarioGenerator(seed=42).generate(n_transactions=500)
        ds2 = FraudScenarioGenerator(seed=42).generate(n_transactions=500)
        np.testing.assert_array_almost_equal(ds1.scalar_scores, ds2.scalar_scores)

    def test_different_seed_different_labels(self):
        ds1 = FraudScenarioGenerator(seed=42).generate(n_transactions=500)
        ds2 = FraudScenarioGenerator(seed=99).generate(n_transactions=500)
        # Extremely unlikely to be identical with different seeds
        assert not np.array_equal(ds1.labels, ds2.labels)


# ===================================================================
# 7. Temporal drift
# ===================================================================

class TestTemporalDrift:
    """Fraud rate can shift mid-dataset to simulate concept drift."""

    def test_drift_changes_fraud_rate(self):
        """Post-drift fraud rate should differ from pre-drift."""
        gen = FraudScenarioGenerator(seed=42)
        ds = gen.generate(
            n_transactions=10000,
            fraud_rate=0.02,
            temporal_drift=True,
            drift_point=0.5,
            drift_fraud_rate=0.10,
        )
        midpoint = len(ds.labels) // 2
        pre_rate = np.mean(ds.labels[:midpoint])
        post_rate = np.mean(ds.labels[midpoint:])
        assert pre_rate == pytest.approx(0.02, abs=0.015)
        assert post_rate == pytest.approx(0.10, abs=0.03)

    def test_no_drift_by_default(self):
        gen = FraudScenarioGenerator(seed=42)
        ds = gen.generate(n_transactions=10000, fraud_rate=0.05)
        midpoint = len(ds.labels) // 2
        pre_rate = np.mean(ds.labels[:midpoint])
        post_rate = np.mean(ds.labels[midpoint:])
        assert pre_rate == pytest.approx(post_rate, abs=0.02)


# ===================================================================
# 8. Signal staleness
# ===================================================================

class TestSignalStaleness:
    """Per-source staleness injects temporal noise."""

    def test_staleness_recorded_in_metadata(self):
        gen = FraudScenarioGenerator(seed=42)
        ds = gen.generate(
            n_transactions=100,
            n_sources=3,
            source_accuracies=[0.9, 0.9, 0.9],
            signal_staleness=[0.0, 5.0, 24.0],
        )
        assert ds.metadata["signal_staleness"] == [0.0, 5.0, 24.0]

    def test_staleness_length_must_match(self):
        gen = FraudScenarioGenerator(seed=42)
        with pytest.raises(ValueError, match="signal_staleness"):
            gen.generate(
                n_transactions=100, n_sources=4,
                signal_staleness=[0.0, 1.0],
            )

    def test_stale_source_has_more_uncertainty(self):
        """A stale source should produce opinions with higher uncertainty."""
        gen = FraudScenarioGenerator(seed=42)
        ds_fresh = gen.generate(
            n_transactions=1000, n_sources=1,
            source_accuracies=[0.9],
            signal_staleness=[0.0],
        )
        gen2 = FraudScenarioGenerator(seed=42)
        ds_stale = gen2.generate(
            n_transactions=1000, n_sources=1,
            source_accuracies=[0.9],
            signal_staleness=[48.0],
        )
        mean_u_fresh = np.mean([ds_fresh.opinions[i][0].u for i in range(1000)])
        mean_u_stale = np.mean([ds_stale.opinions[i][0].u for i in range(1000)])
        assert mean_u_stale > mean_u_fresh
