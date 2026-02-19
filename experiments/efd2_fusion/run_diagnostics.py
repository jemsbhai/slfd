"""E-FD2 Diagnostic Runner — analyze WHY the first run produced its results.

Loads the IEEE-CIS data, trains the same 4-model ensemble, and runs
the diagnostic analysis to understand opinion distributions, fused
opinion shapes, and decider threshold behavior.

Usage:
    python experiments/efd2_fusion/run_diagnostics.py [--seed 42]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import numpy as np

from slfd.data.ieee_cis import load_raw, preprocess, make_three_way_splits, DATA_DIR
from slfd.models.ensemble import train_model_suite
from slfd.decision import ThreeWayDecider
from slfd.experiments.diagnose_efd2 import run_full_diagnostics
from slfd.results import save_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("efd2-diag")


def main() -> None:
    parser = argparse.ArgumentParser(description="E-FD2 Diagnostics")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str,
                        default=str(_PROJECT_ROOT / "results"))
    args = parser.parse_args()

    t_start = time.perf_counter()

    # --- Load and split (same as run.py) ---
    log.info("Loading IEEE-CIS dataset...")
    df_raw = load_raw(DATA_DIR)
    df_clean = preprocess(df_raw)
    data = make_three_way_splits(df_clean, seed=args.seed)

    log.info("Train: %d | Val: %d | Test: %d",
             len(data.y_train), len(data.y_val), len(data.y_test))

    # --- Train ensemble ---
    log.info("Training 4-model ensemble...")
    t_train = time.perf_counter()
    suite = train_model_suite(data.X_train, data.y_train, seed=args.seed)
    log.info("Training done in %.1fs", time.perf_counter() - t_train)

    # --- Get test predictions ---
    log.info("Generating test predictions...")
    test_preds = suite.predict_all(data.X_test)

    # --- Run diagnostics ---
    log.info("Running diagnostics on %d test transactions...", len(data.y_test))
    decider = ThreeWayDecider(
        block_threshold=0.6,
        approve_threshold=0.6,
        escalate_uncertainty=0.4,
        escalate_conflict=0.3,
    )
    base_rate = float(np.mean(data.y_train))

    t_diag = time.perf_counter()
    diag = run_full_diagnostics(
        pred_set=test_preds,
        labels=data.y_test,
        base_rate=base_rate,
        decider=decider,
    )
    log.info("Diagnostics done in %.1fs", time.perf_counter() - t_diag)

    # --- Print summary ---
    log.info("")
    log.info("=" * 72)
    log.info("PER-MODEL OPINION DISTRIBUTIONS")
    log.info("=" * 72)
    for s in diag.opinion_distributions:
        log.info("")
        log.info("  %s:", s.model_name)
        log.info("    prob:  mean=%.4f  std=%.4f", s.prob_mean, s.prob_std)
        log.info("    uncert: mean=%.4f  std=%.4f", s.raw_uncert_mean, s.raw_uncert_std)
        log.info("    b:     mean=%.4f  std=%.4f  [%.4f, %.4f]",
                 s.b_mean, s.b_std, s.b_min, s.b_max)
        log.info("    d:     mean=%.4f  std=%.4f  [%.4f, %.4f]",
                 s.d_mean, s.d_std, s.d_min, s.d_max)
        log.info("    u:     mean=%.4f  std=%.4f  p25=%.4f  p50=%.4f  p75=%.4f",
                 s.u_mean, s.u_std, s.u_p25, s.u_p50, s.u_p75)

    log.info("")
    log.info("=" * 72)
    log.info("FUSED OPINION DISTRIBUTIONS (cumulative_fuse of 4 sources)")
    log.info("=" * 72)
    f = diag.fused_opinions
    log.info("  b:     mean=%.4f  std=%.4f  p25=%.4f  p50=%.4f  p75=%.4f",
             f.b_mean, f.b_std, f.b_p25, f.b_p50, f.b_p75)
    log.info("  d:     mean=%.4f  std=%.4f  p25=%.4f  p50=%.4f  p75=%.4f",
             f.d_mean, f.d_std, f.d_p25, f.d_p50, f.d_p75)
    log.info("  u:     mean=%.4f  std=%.4f  p25=%.4f  p50=%.4f  p75=%.4f",
             f.u_mean, f.u_std, f.u_p25, f.u_p50, f.u_p75)
    log.info("  conflict: mean=%.4f  std=%.4f  p25=%.4f  p50=%.4f  p75=%.4f",
             f.conflict_mean, f.conflict_std, f.conflict_p25, f.conflict_p50, f.conflict_p75)
    log.info("  E[P]:  mean=%.4f  std=%.4f  [%.4f, %.4f]",
             f.expected_prob_mean, f.expected_prob_std,
             f.expected_prob_min, f.expected_prob_max)

    log.info("")
    log.info("=" * 72)
    log.info("DECIDER DIAGNOSTICS (block=%.2f, approve=%.2f, esc_u=%.2f, esc_c=%.2f)",
             decider.block_threshold, decider.approve_threshold,
             decider.escalate_uncertainty, decider.escalate_conflict)
    log.info("=" * 72)
    d = diag.decider_diagnostics
    log.info("  Decision fractions:")
    log.info("    BLOCK:    %.4f (%.1f%%)", d.frac_block, d.frac_block * 100)
    log.info("    APPROVE:  %.4f (%.1f%%)", d.frac_approve, d.frac_approve * 100)
    log.info("    ESCALATE: %.4f (%.1f%%)", d.frac_escalate, d.frac_escalate * 100)
    log.info("  Escalation reasons:")
    log.info("    conflict > %.2f:    %.4f (%.1f%%)",
             decider.escalate_conflict, d.escalate_by_conflict, d.escalate_by_conflict * 100)
    log.info("    uncertainty > %.2f:  %.4f (%.1f%%)",
             decider.escalate_uncertainty, d.escalate_by_uncertainty, d.escalate_by_uncertainty * 100)
    log.info("    default (no thresh): %.4f (%.1f%%)",
             d.escalate_by_default, d.escalate_by_default * 100)
    log.info("  Threshold exceedance (fused opinions):")
    log.info("    b > %.2f (block):    %.1f%%",
             decider.block_threshold, d.frac_b_above_block * 100)
    log.info("    d > %.2f (approve):  %.1f%%",
             decider.approve_threshold, d.frac_d_above_approve * 100)
    log.info("    u > %.2f (escalate): %.1f%%",
             decider.escalate_uncertainty, d.frac_u_above_escalate * 100)
    log.info("    conflict > %.2f:     %.1f%%",
             decider.escalate_conflict, d.frac_conflict_above_threshold * 100)

    log.info("")
    log.info("=" * 72)
    log.info("PER-MODEL INDIVIDUAL PERFORMANCE")
    log.info("=" * 72)
    log.info("  %-20s %8s %8s %8s %10s %10s",
             "Model", "Acc", "ROC-AUC", "PR-AUC", "P(fraud)", "P(legit)")
    log.info("  " + "-" * 68)
    for m in diag.model_performance:
        log.info("  %-20s %8.4f %8.4f %8.4f %10.4f %10.4f",
                 m.model_name, m.accuracy, m.roc_auc, m.pr_auc,
                 m.prob_mean_fraud, m.prob_mean_legit)

    # --- Save ---
    result_data = diag.to_dict()
    result_data["metadata"] = {
        "seed": args.seed,
        "base_rate": base_rate,
        "n_test": len(data.y_test),
        "decider": {
            "block_threshold": decider.block_threshold,
            "approve_threshold": decider.approve_threshold,
            "escalate_uncertainty": decider.escalate_uncertainty,
            "escalate_conflict": decider.escalate_conflict,
        },
        "timing_seconds": round(time.perf_counter() - t_start, 1),
    }

    out_path = save_results(
        data=result_data,
        experiment="efd2_diag",
        results_dir=Path(args.output_dir),
    )
    log.info("")
    log.info("Diagnostics saved to %s", out_path)
    log.info("Total time: %.1fs", time.perf_counter() - t_start)


if __name__ == "__main__":
    main()
