"""E-FD2: Multi-Source Fraud Signal Fusion — IEEE-CIS run script.

Runs the full 9-arm experiment on the IEEE-CIS Fraud Detection dataset:
    1. Load and preprocess (~590K transactions)
    2. Three-way stratified split (60/20/20)
    3. Train 4-model ensemble on train set
    4. Generate val + test predictions
    5. Run all 9 fusion strategies
    6. Evaluate: PR-AUC, FPR@TPR, cost, escalation, significance
    7. Serialize results to JSON

Usage:
    python experiments/efd2_fusion/run.py [--seed 42] [--n-bootstrap 1000] [--output results/efd2_results.json]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add project root to path for imports
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import numpy as np

from slfd.data.ieee_cis import load_raw, preprocess, make_three_way_splits, DATA_DIR
from slfd.experiments.efd2 import EFD2Config, run_efd2, serialize_results
from slfd.metrics import CostConfig
from slfd.decision import ThreeWayDecider
from slfd.results import save_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("efd2")


def main() -> None:
    parser = argparse.ArgumentParser(description="E-FD2: Multi-Source Fraud Signal Fusion")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--n-bootstrap", type=int, default=1000,
                        help="Bootstrap resamples for CIs (default: 1000)")
    parser.add_argument("--output-dir", type=str,
                        default=str(_PROJECT_ROOT / "results"),
                        help="Output directory (default: results/)")
    parser.add_argument("--suffix", type=str, default=None,
                        help="Optional suffix for result filename (e.g. 'run1')")
    parser.add_argument("--train-size", type=float, default=0.6)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)

    t_start = time.perf_counter()

    # --- 1. Load and preprocess ---
    log.info("Loading IEEE-CIS dataset from %s", DATA_DIR)
    df_raw = load_raw(DATA_DIR)
    log.info("Raw dataset: %d transactions, %d columns", len(df_raw), len(df_raw.columns))
    log.info("Fraud rate: %.4f", df_raw["isFraud"].mean())

    log.info("Preprocessing...")
    df_clean = preprocess(df_raw)
    log.info("After preprocessing: %d transactions, %d features",
             len(df_clean), len(df_clean.columns) - 1)

    # --- 2. Three-way stratified split ---
    log.info("Splitting: train=%.0f%% val=%.0f%% test=%.0f%%",
             args.train_size * 100, args.val_size * 100, args.test_size * 100)
    data = make_three_way_splits(
        df_clean,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    log.info("Train: %d (fraud %.4f)", len(data.y_train), np.mean(data.y_train))
    log.info("Val:   %d (fraud %.4f)", len(data.y_val), np.mean(data.y_val))
    log.info("Test:  %d (fraud %.4f)", len(data.y_test), np.mean(data.y_test))

    # --- 3. Configure experiment ---
    config = EFD2Config(
        seed=args.seed,
        base_rate=float(np.mean(data.y_train)),  # Empirical fraud rate from training set
        cost_config=CostConfig(
            review_cost=2.0,
            missed_fraud_cost=50.0,
            false_block_cost=0.50,
        ),
        n_bootstrap=args.n_bootstrap,
        decider=ThreeWayDecider(
            block_threshold=0.6,
            approve_threshold=0.6,
            escalate_uncertainty=0.4,
            escalate_conflict=0.3,
        ),
        robust_threshold=0.15,
    )

    # --- 4-6. Train + predict + fuse + evaluate ---
    log.info("Running E-FD2 (train → predict → fuse → evaluate)...")
    log.info("  Training 4-model ensemble (XGBoost, RF, MLP, IsolationForest)...")
    t_run = time.perf_counter()

    results = run_efd2(
        X_train=data.X_train, y_train=data.y_train,
        X_val=data.X_val, y_val=data.y_val,
        X_test=data.X_test, y_test=data.y_test,
        config=config,
    )

    t_run_end = time.perf_counter()
    log.info("E-FD2 completed in %.1f seconds", t_run_end - t_run)

    # --- 7. Report summary ---
    log.info("")
    log.info("=" * 72)
    log.info("E-FD2 RESULTS SUMMARY")
    log.info("=" * 72)
    log.info("%-30s %8s %8s %10s", "Arm", "PR-AUC", "F1-max", "Cost/txn")
    log.info("-" * 72)
    for arm in results.arm_results:
        esc_str = ""
        if arm.escalation is not None:
            esc_str = f"  [esc={arm.escalation.escalation_rate:.1%}]"
        log.info("%-30s %8.4f %8.4f %10.4f%s",
                 arm.arm_name,
                 arm.pr_curve.auc,
                 arm.pr_curve.f1_max,
                 arm.expected_cost,
                 esc_str)

    log.info("")
    log.info("FPR at fixed TPR:")
    log.info("%-30s %10s %10s %10s", "Arm", "TPR=0.90", "TPR=0.95", "TPR=0.99")
    log.info("-" * 72)
    for arm in results.arm_results:
        fpr = arm.fpr_at_tpr
        log.info("%-30s %10.4f %10.4f %10.4f",
                 arm.arm_name,
                 fpr.fpr_values[0], fpr.fpr_values[1], fpr.fpr_values[2])

    log.info("")
    log.info("Significance tests (McNemar, α=0.05):")
    for test in results.significance_tests:
        sig = "***" if test["p_value"] < 0.001 else "**" if test["p_value"] < 0.01 else "*" if test["p_value"] < 0.05 else "ns"
        log.info("  %s vs %s: χ²=%.2f, p=%.4f %s",
                 test["arm_a"], test["arm_b"],
                 test["statistic"], test["p_value"], sig)

    log.info("")
    log.info("Bootstrap CIs (PR-AUC difference, 95%%):")
    for ci in results.bootstrap_cis:
        sig = "SIG" if ci["is_significant"] else "ns"
        log.info("  %s vs %s: diff=%.4f [%.4f, %.4f] %s",
                 ci["arm_a"], ci["arm_b"],
                 ci["observed_diff"], ci["ci_lower"], ci["ci_upper"], sig)

    # --- 8. Serialize and save with timestamped filename ---
    serialized = serialize_results(results)
    serialized["timing"] = {
        "total_seconds": round(time.perf_counter() - t_start, 1),
        "run_seconds": round(t_run_end - t_run, 1),
    }
    serialized["split_config"] = {
        "train_size": args.train_size,
        "val_size": args.val_size,
        "test_size": args.test_size,
    }

    out_path = save_results(
        data=serialized,
        experiment="efd2",
        results_dir=out_dir,
        suffix=args.suffix,
    )
    log.info("")
    log.info("Results written to %s", out_path)
    log.info("Total time: %.1f seconds", time.perf_counter() - t_start)


if __name__ == "__main__":
    main()
