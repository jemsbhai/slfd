"""Tests for E-FD2 experiment runner.

Orchestrates the full 9-arm multi-source fusion experiment:
    1. Train 4-model ensemble
    2. Collect val + test predictions
    3. Run all 9 fusion strategies
    4. Evaluate each arm on all metrics
    5. Pairwise significance tests
    6. Serialize results

Tests use small synthetic data for speed. Full IEEE-CIS run is a
separate slow-marked integration test.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from slfd.decision import Decision, ThreeWayDecider
from slfd.metrics import ArmResult, PRCurveData, EscalationAnalysis, CostConfig
from slfd.models.ensemble import PredictionSet

from slfd.experiments.efd2 import (
    EFD2Config,
    EFD2Results,
    run_efd2,
    run_efd2_from_predictions,
    serialize_results,
)


# ===================================================================
# Synthetic data fixture (small, fast)
# ===================================================================

def _make_synthetic_data(
    n_train: int = 200,
    n_val: int = 80,
    n_test: int = 100,
    n_features: int = 20,
    fraud_rate: float = 0.10,
    seed: int = 42,
):
    """Generate small synthetic dataset for testing the runner."""
    rng = np.random.default_rng(seed)

    def _make_split(n):
        X = rng.standard_normal((n, n_features)).astype(np.float32)
        y = (rng.uniform(size=n) < fraud_rate).astype(np.int32)
        # Make features somewhat predictive: add signal
        X[:, 0] += y * 2.0
        X[:, 1] -= y * 1.5
        return X, y

    X_train, y_train = _make_split(n_train)
    X_val, y_val = _make_split(n_val)
    X_test, y_test = _make_split(n_test)

    return X_train, y_train, X_val, y_val, X_test, y_test


@pytest.fixture
def synthetic_data():
    return _make_synthetic_data()


@pytest.fixture
def default_config():
    return EFD2Config(
        seed=42,
        base_rate=0.10,
        cost_config=CostConfig(),
        n_bootstrap=50,  # Low for speed in tests
        decider=ThreeWayDecider(
            block_threshold=0.6,
            approve_threshold=0.6,
            escalate_uncertainty=0.4,
            escalate_conflict=0.3,
        ),
        robust_threshold=0.15,
    )


# ===================================================================
# EFD2Config
# ===================================================================

class TestEFD2Config:
    """Configuration for the E-FD2 experiment."""

    def test_default_construction(self):
        cfg = EFD2Config()
        assert cfg.seed == 42
        assert cfg.base_rate == 0.035
        assert cfg.n_bootstrap == 1000

    def test_custom_construction(self):
        cfg = EFD2Config(seed=99, base_rate=0.05, n_bootstrap=500)
        assert cfg.seed == 99
        assert cfg.base_rate == 0.05

    def test_has_decider(self):
        cfg = EFD2Config()
        assert isinstance(cfg.decider, ThreeWayDecider)

    def test_has_cost_config(self):
        cfg = EFD2Config()
        assert isinstance(cfg.cost_config, CostConfig)


# ===================================================================
# run_efd2_from_predictions (core logic, no model training)
# ===================================================================

class TestRunEFD2FromPredictions:
    """Test the runner using pre-computed predictions.

    This tests the fusion + evaluation logic without model training.
    """

    def _make_pred_sets(self, n_val=80, n_test=100, n_models=4, seed=42):
        rng = np.random.default_rng(seed)
        val_probs = rng.uniform(0.0, 1.0, (n_val, n_models))
        val_uncerts = rng.uniform(0.05, 0.50, (n_val, n_models))
        test_probs = rng.uniform(0.0, 1.0, (n_test, n_models))
        test_uncerts = rng.uniform(0.05, 0.50, (n_test, n_models))
        names = [f"model_{i}" for i in range(n_models)]

        val_pset = PredictionSet(val_probs, val_uncerts, names)
        test_pset = PredictionSet(test_probs, test_uncerts, names)

        val_labels = (rng.uniform(size=n_val) < 0.1).astype(np.int32)
        test_labels = (rng.uniform(size=n_test) < 0.1).astype(np.int32)

        return val_pset, val_labels, test_pset, test_labels

    def test_returns_efd2_results(self, default_config):
        vp, vl, tp, tl = self._make_pred_sets()
        results = run_efd2_from_predictions(
            val_predictions=vp, val_labels=vl,
            test_predictions=tp, test_labels=tl,
            config=default_config,
        )
        assert isinstance(results, EFD2Results)

    def test_has_all_eleven_arms(self, default_config):
        vp, vl, tp, tl = self._make_pred_sets()
        results = run_efd2_from_predictions(vp, vl, tp, tl, default_config)
        assert len(results.arm_results) == 11

    def test_arm_names_match_protocol(self, default_config):
        vp, vl, tp, tl = self._make_pred_sets()
        results = run_efd2_from_predictions(vp, vl, tp, tl, default_config)
        expected_names = {
            "A_majority_vote", "B_weighted_average", "C_stacking",
            "D_bayesian_model_average", "E_noisy_or",
            "F_sl_cumulative", "G_sl_three_way",
            "H_sl_robust_three_way", "I_confidence_feature",
            "J_sl_trust_cumulative", "K_sl_trust_three_way",
        }
        actual_names = {r.arm_name for r in results.arm_results}
        assert actual_names == expected_names

    def test_each_arm_has_pr_curve(self, default_config):
        vp, vl, tp, tl = self._make_pred_sets()
        results = run_efd2_from_predictions(vp, vl, tp, tl, default_config)
        for arm in results.arm_results:
            assert isinstance(arm.pr_curve, PRCurveData), f"{arm.arm_name} missing PR curve"

    def test_each_arm_has_cost(self, default_config):
        vp, vl, tp, tl = self._make_pred_sets()
        results = run_efd2_from_predictions(vp, vl, tp, tl, default_config)
        for arm in results.arm_results:
            assert isinstance(arm.expected_cost, float), f"{arm.arm_name} missing cost"
            assert arm.expected_cost >= 0.0

    def test_three_way_arms_have_escalation(self, default_config):
        """Arms G, H, K should have escalation analysis."""
        vp, vl, tp, tl = self._make_pred_sets()
        results = run_efd2_from_predictions(vp, vl, tp, tl, default_config)
        three_way_arms = {"G_sl_three_way", "H_sl_robust_three_way", "K_sl_trust_three_way"}
        for arm in results.arm_results:
            if arm.arm_name in three_way_arms:
                assert isinstance(arm.escalation, EscalationAnalysis), (
                    f"{arm.arm_name} missing escalation analysis"
                )

    def test_scalar_arms_no_escalation(self, default_config):
        """Arms A-F should NOT have escalation analysis."""
        vp, vl, tp, tl = self._make_pred_sets()
        results = run_efd2_from_predictions(vp, vl, tp, tl, default_config)
        scalar_arms = {
            "A_majority_vote", "B_weighted_average", "C_stacking",
            "D_bayesian_model_average", "E_noisy_or", "F_sl_cumulative",
            "I_confidence_feature", "J_sl_trust_cumulative",
        }
        for arm in results.arm_results:
            if arm.arm_name in scalar_arms:
                assert arm.escalation is None, (
                    f"{arm.arm_name} should not have escalation"
                )

    def test_has_significance_tests(self, default_config):
        """Results should include pairwise significance comparisons."""
        vp, vl, tp, tl = self._make_pred_sets()
        results = run_efd2_from_predictions(vp, vl, tp, tl, default_config)
        assert results.significance_tests is not None
        assert len(results.significance_tests) > 0

    def test_has_bootstrap_cis(self, default_config):
        """Results should include bootstrap CIs for key comparisons."""
        vp, vl, tp, tl = self._make_pred_sets()
        results = run_efd2_from_predictions(vp, vl, tp, tl, default_config)
        assert results.bootstrap_cis is not None
        assert len(results.bootstrap_cis) > 0

    def test_has_metadata(self, default_config):
        """Results should carry experiment metadata for reproducibility."""
        vp, vl, tp, tl = self._make_pred_sets()
        results = run_efd2_from_predictions(vp, vl, tp, tl, default_config)
        assert results.metadata is not None
        assert "seed" in results.metadata
        assert "base_rate" in results.metadata
        assert "n_test" in results.metadata

    def test_deterministic(self, default_config):
        """Same inputs + same config → same results."""
        vp, vl, tp, tl = self._make_pred_sets()
        r1 = run_efd2_from_predictions(vp, vl, tp, tl, default_config)
        r2 = run_efd2_from_predictions(vp, vl, tp, tl, default_config)
        for a1, a2 in zip(r1.arm_results, r2.arm_results):
            assert a1.arm_name == a2.arm_name
            assert a1.pr_curve.auc == pytest.approx(a2.pr_curve.auc)
            assert a1.expected_cost == pytest.approx(a2.expected_cost)


# ===================================================================
# run_efd2 (full pipeline: train + predict + evaluate)
# ===================================================================

class TestRunEFD2:
    """Full pipeline: train models → predict → fuse → evaluate.

    Uses small synthetic data for speed.
    """

    def test_returns_efd2_results(self, synthetic_data, default_config):
        X_tr, y_tr, X_v, y_v, X_te, y_te = synthetic_data
        results = run_efd2(
            X_train=X_tr, y_train=y_tr,
            X_val=X_v, y_val=y_v,
            X_test=X_te, y_test=y_te,
            config=default_config,
        )
        assert isinstance(results, EFD2Results)

    def test_has_all_eleven_arms(self, synthetic_data, default_config):
        X_tr, y_tr, X_v, y_v, X_te, y_te = synthetic_data
        results = run_efd2(X_tr, y_tr, X_v, y_v, X_te, y_te, default_config)
        assert len(results.arm_results) == 11

    def test_scores_in_valid_range(self, synthetic_data, default_config):
        X_tr, y_tr, X_v, y_v, X_te, y_te = synthetic_data
        results = run_efd2(X_tr, y_tr, X_v, y_v, X_te, y_te, default_config)
        for arm in results.arm_results:
            assert arm.pr_curve.auc >= 0.0
            assert arm.pr_curve.auc <= 1.0

    def test_metadata_includes_n_train(self, synthetic_data, default_config):
        X_tr, y_tr, X_v, y_v, X_te, y_te = synthetic_data
        results = run_efd2(X_tr, y_tr, X_v, y_v, X_te, y_te, default_config)
        assert results.metadata["n_train"] == len(y_tr)
        assert results.metadata["n_val"] == len(y_v)
        assert results.metadata["n_test"] == len(y_te)


# ===================================================================
# Serialization
# ===================================================================

class TestSerializeResults:
    """Results must be JSON-serializable for reproducibility."""

    def test_serialize_returns_dict(self, synthetic_data, default_config):
        X_tr, y_tr, X_v, y_v, X_te, y_te = synthetic_data
        results = run_efd2(X_tr, y_tr, X_v, y_v, X_te, y_te, default_config)
        serialized = serialize_results(results)
        assert isinstance(serialized, dict)

    def test_json_serializable(self, synthetic_data, default_config):
        X_tr, y_tr, X_v, y_v, X_te, y_te = synthetic_data
        results = run_efd2(X_tr, y_tr, X_v, y_v, X_te, y_te, default_config)
        serialized = serialize_results(results)
        # Must not raise
        json_str = json.dumps(serialized, indent=2)
        assert len(json_str) > 0

    def test_roundtrip_arm_names(self, synthetic_data, default_config):
        X_tr, y_tr, X_v, y_v, X_te, y_te = synthetic_data
        results = run_efd2(X_tr, y_tr, X_v, y_v, X_te, y_te, default_config)
        serialized = serialize_results(results)
        arm_names = [a["arm_name"] for a in serialized["arms"]]
        assert "A_majority_vote" in arm_names
        assert "G_sl_three_way" in arm_names

    def test_has_metadata_in_output(self, synthetic_data, default_config):
        X_tr, y_tr, X_v, y_v, X_te, y_te = synthetic_data
        results = run_efd2(X_tr, y_tr, X_v, y_v, X_te, y_te, default_config)
        serialized = serialize_results(results)
        assert "metadata" in serialized

    def test_has_significance_in_output(self, synthetic_data, default_config):
        X_tr, y_tr, X_v, y_v, X_te, y_te = synthetic_data
        results = run_efd2(X_tr, y_tr, X_v, y_v, X_te, y_te, default_config)
        serialized = serialize_results(results)
        assert "significance_tests" in serialized

    def test_write_to_file(self, synthetic_data, default_config, tmp_path):
        X_tr, y_tr, X_v, y_v, X_te, y_te = synthetic_data
        results = run_efd2(X_tr, y_tr, X_v, y_v, X_te, y_te, default_config)
        out_path = tmp_path / "efd2_results.json"
        serialized = serialize_results(results)
        out_path.write_text(json.dumps(serialized, indent=2))
        assert out_path.exists()
        loaded = json.loads(out_path.read_text())
        assert len(loaded["arms"]) == 11
