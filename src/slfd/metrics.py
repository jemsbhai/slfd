"""E-FD2 evaluation metrics.

Computes all metrics for the 9-arm fusion experiment:
    - PR curve data + PR-AUC + best F1
    - FPR at fixed TPR levels (0.90, 0.95, 0.99)
    - Escalation analysis (three-way strategies)
    - Cost-sensitive expected cost per transaction
    - McNemar's test for pairwise statistical significance
    - Bootstrap confidence intervals on metric differences
    - All-in-one arm evaluation convenience function
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
)


# ===================================================================
# PR Curve
# ===================================================================

@dataclass(frozen=True, slots=True)
class PRCurveData:
    """Precision-recall curve data.

    Attributes
    ----------
    precision : np.ndarray
        Precision values at each threshold.
    recall : np.ndarray
        Recall values at each threshold.
    thresholds : np.ndarray
        Decision thresholds (one fewer than precision/recall).
    auc : float
        Area under the PR curve.
    f1_max : float
        Maximum F1 score achievable.
    f1_max_threshold : float
        Threshold achieving the maximum F1.
    """

    precision: np.ndarray
    recall: np.ndarray
    thresholds: np.ndarray
    auc: float
    f1_max: float
    f1_max_threshold: float


def compute_pr_curve(y_true: np.ndarray, scores: np.ndarray) -> PRCurveData:
    """Compute precision-recall curve, PR-AUC, and best F1.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels, shape (n,).
    scores : np.ndarray
        Predicted fraud scores, shape (n,), values in [0, 1].

    Returns
    -------
    PRCurveData
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    pr_auc = auc(recall, precision)

    # Compute F1 at each threshold (precision and recall have len = thresholds + 1;
    # the last precision/recall point has no corresponding threshold)
    # Use precision[:-1] and recall[:-1] which align with thresholds
    p = precision[:-1]
    r = recall[:-1]
    denom = p + r
    # Use safe division to avoid RuntimeWarning from numpy evaluating
    # both branches of np.where before masking
    safe_denom = np.where(denom > 0, denom, 1.0)
    f1_scores = np.where(denom > 0, 2 * p * r / safe_denom, 0.0)

    if len(f1_scores) > 0:
        best_idx = np.argmax(f1_scores)
        f1_max = float(f1_scores[best_idx])
        f1_max_threshold = float(thresholds[best_idx])
    else:
        f1_max = 0.0
        f1_max_threshold = 0.5

    return PRCurveData(
        precision=precision,
        recall=recall,
        thresholds=thresholds,
        auc=float(pr_auc),
        f1_max=f1_max,
        f1_max_threshold=f1_max_threshold,
    )


# ===================================================================
# FPR at fixed TPR
# ===================================================================

_DEFAULT_TPR_TARGETS = np.array([0.90, 0.95, 0.99])


@dataclass(frozen=True, slots=True)
class FPRAtTPR:
    """FPR at fixed TPR operating points.

    Attributes
    ----------
    tpr_targets : np.ndarray
        Requested TPR levels.
    fpr_values : np.ndarray
        Achieved FPR at each TPR target.
    thresholds : np.ndarray
        Score thresholds achieving each operating point.
    """

    tpr_targets: np.ndarray
    fpr_values: np.ndarray
    thresholds: np.ndarray


def compute_fpr_at_tpr(
    y_true: np.ndarray,
    scores: np.ndarray,
    tpr_targets: np.ndarray | None = None,
) -> FPRAtTPR:
    """Compute FPR at fixed TPR operating points from the ROC curve.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels, shape (n,).
    scores : np.ndarray
        Predicted fraud scores, shape (n,).
    tpr_targets : np.ndarray or None
        Desired TPR levels. Defaults to [0.90, 0.95, 0.99].

    Returns
    -------
    FPRAtTPR
    """
    if tpr_targets is None:
        tpr_targets = _DEFAULT_TPR_TARGETS.copy()

    fpr_arr, tpr_arr, thresh_arr = roc_curve(y_true, scores)

    fpr_values = np.zeros(len(tpr_targets), dtype=np.float64)
    thresholds = np.zeros(len(tpr_targets), dtype=np.float64)

    for i, target in enumerate(tpr_targets):
        # Find the lowest threshold where TPR >= target
        valid = tpr_arr >= target
        if np.any(valid):
            # Among points with TPR >= target, pick the one with lowest FPR
            idx = np.where(valid)[0][0]
            fpr_values[i] = fpr_arr[idx]
            thresholds[i] = thresh_arr[idx] if idx < len(thresh_arr) else 0.0
        else:
            # Cannot achieve this TPR — report FPR=1.0
            fpr_values[i] = 1.0
            thresholds[i] = 0.0

    return FPRAtTPR(
        tpr_targets=tpr_targets,
        fpr_values=fpr_values,
        thresholds=thresholds,
    )


# ===================================================================
# Escalation Analysis
# ===================================================================

@dataclass(frozen=True, slots=True)
class EscalationAnalysis:
    """Metrics for three-way decision escalation.

    Attributes
    ----------
    escalation_rate : float
        Fraction of transactions routed to escalation.
    fraud_rate_escalated : float
        Fraud rate among escalated transactions (NaN if none escalated).
    fraud_rate_auto_decided : float
        Fraud rate among auto-decided transactions (NaN if all escalated).
    n_escalated : int
        Count of escalated transactions.
    n_auto_decided : int
        Count of auto-decided transactions.
    precision_auto_decided : float
        Precision on auto-decided subset (NaN if no positives predicted).
    recall_auto_decided : float
        Recall on auto-decided subset (NaN if no actual positives).
    """

    escalation_rate: float
    fraud_rate_escalated: float
    fraud_rate_auto_decided: float
    n_escalated: int
    n_auto_decided: int
    precision_auto_decided: float
    recall_auto_decided: float


def compute_escalation_analysis(
    y_true: np.ndarray,
    scores: np.ndarray,
    escalation_mask: np.ndarray,
    threshold: float = 0.5,
) -> EscalationAnalysis:
    """Compute escalation metrics for a three-way strategy.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels, shape (n,).
    scores : np.ndarray
        Fraud scores, shape (n,).
    escalation_mask : np.ndarray
        Boolean — True for escalated transactions, shape (n,).
    threshold : float
        Score threshold for binary decision on auto-decided subset.

    Returns
    -------
    EscalationAnalysis
    """
    n = len(y_true)
    n_esc = int(np.sum(escalation_mask))
    n_auto = n - n_esc

    escalation_rate = n_esc / n if n > 0 else 0.0

    # Fraud rates in each partition
    if n_esc > 0:
        fraud_rate_esc = float(np.mean(y_true[escalation_mask]))
    else:
        fraud_rate_esc = float("nan")

    auto_mask = ~escalation_mask
    if n_auto > 0:
        fraud_rate_auto = float(np.mean(y_true[auto_mask]))
        # Precision/recall on auto-decided subset
        y_auto = y_true[auto_mask]
        preds_auto = (scores[auto_mask] > threshold).astype(np.int32)
        if np.sum(preds_auto) > 0:
            prec = float(precision_score(y_auto, preds_auto, zero_division=0.0))
        else:
            prec = float("nan")
        if np.sum(y_auto) > 0:
            rec = float(recall_score(y_auto, preds_auto, zero_division=0.0))
        else:
            rec = float("nan")
    else:
        fraud_rate_auto = float("nan")
        prec = float("nan")
        rec = float("nan")

    return EscalationAnalysis(
        escalation_rate=escalation_rate,
        fraud_rate_escalated=fraud_rate_esc,
        fraud_rate_auto_decided=fraud_rate_auto,
        n_escalated=n_esc,
        n_auto_decided=n_auto,
        precision_auto_decided=prec,
        recall_auto_decided=rec,
    )


# ===================================================================
# Cost-Sensitive Metric
# ===================================================================

@dataclass(frozen=True, slots=True)
class CostConfig:
    """Cost structure for fraud decision evaluation.

    Defaults match the paper protocol:
        $2/review, $50/missed fraud, $0.50/false block.

    Parameters
    ----------
    review_cost : float
        Cost per human review (escalated transaction).
    missed_fraud_cost : float
        Cost per false negative (fraud not caught).
    false_block_cost : float
        Cost per false positive (legitimate transaction blocked).
    """

    review_cost: float = 2.0
    missed_fraud_cost: float = 50.0
    false_block_cost: float = 0.50

    def __post_init__(self) -> None:
        if self.review_cost < 0:
            raise ValueError(f"review_cost must be >= 0, got {self.review_cost}")
        if self.missed_fraud_cost < 0:
            raise ValueError(f"missed_fraud_cost must be >= 0, got {self.missed_fraud_cost}")
        if self.false_block_cost < 0:
            raise ValueError(f"false_block_cost must be >= 0, got {self.false_block_cost}")


def compute_expected_cost(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
    escalation_mask: np.ndarray | None = None,
    cost_config: CostConfig | None = None,
) -> float:
    """Compute total expected cost per transaction.

    Cost components:
        - False positive (block legit): false_block_cost × count
        - False negative (miss fraud):  missed_fraud_cost × count
        - Escalation (human review):    review_cost × count

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels, shape (n,).
    scores : np.ndarray
        Fraud scores, shape (n,).
    threshold : float
        Decision threshold for auto-decided transactions.
    escalation_mask : np.ndarray or None
        Boolean — True for escalated (reviewed) transactions.
        If None, no escalation (all auto-decided).
    cost_config : CostConfig or None
        Cost parameters. Defaults to paper values.

    Returns
    -------
    float
        Mean cost per transaction.
    """
    if cost_config is None:
        cost_config = CostConfig()

    n = len(y_true)
    if n == 0:
        return 0.0

    total_cost = 0.0

    if escalation_mask is None:
        escalation_mask = np.zeros(n, dtype=bool)

    # --- Escalated transactions: incur review cost ---
    n_esc = int(np.sum(escalation_mask))
    total_cost += n_esc * cost_config.review_cost

    # --- Auto-decided transactions: incur error costs ---
    auto_mask = ~escalation_mask
    if np.any(auto_mask):
        y_auto = y_true[auto_mask]
        preds_auto = (scores[auto_mask] > threshold).astype(np.int32)

        # False positives: predicted fraud but actually legit
        fp = np.sum((preds_auto == 1) & (y_auto == 0))
        total_cost += fp * cost_config.false_block_cost

        # False negatives: predicted legit but actually fraud
        fn = np.sum((preds_auto == 0) & (y_auto == 1))
        total_cost += fn * cost_config.missed_fraud_cost

    return total_cost / n


# ===================================================================
# McNemar's Test
# ===================================================================

@dataclass(frozen=True, slots=True)
class McNemarResult:
    """Result of McNemar's test comparing two classifiers.

    Attributes
    ----------
    statistic : float
        McNemar's chi-squared statistic.
    p_value : float
        Two-sided p-value.
    n_discordant_a : int
        Count where A is correct and B is wrong.
    n_discordant_b : int
        Count where B is correct and A is wrong.
    """

    statistic: float
    p_value: float
    n_discordant_a: int
    n_discordant_b: int


def mcnemar_test(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    threshold: float = 0.5,
) -> McNemarResult:
    """McNemar's test for comparing two classifiers' predictions.

    Compares the binary predictions of two models at the given threshold.
    Tests whether the discordant error rates differ significantly.

    Uses continuity-corrected McNemar's test:
        χ² = (|b - c| - 1)² / (b + c)
    where b = A correct & B wrong, c = A wrong & B correct.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels, shape (n,).
    scores_a : np.ndarray
        Scores from model A, shape (n,).
    scores_b : np.ndarray
        Scores from model B, shape (n,).
    threshold : float
        Decision threshold for binarizing scores.

    Returns
    -------
    McNemarResult
    """
    from scipy.stats import chi2

    preds_a = (scores_a > threshold).astype(np.int32)
    preds_b = (scores_b > threshold).astype(np.int32)

    correct_a = (preds_a == y_true)
    correct_b = (preds_b == y_true)

    # Discordant pairs
    # b: A correct, B wrong
    b = int(np.sum(correct_a & ~correct_b))
    # c: A wrong, B correct
    c = int(np.sum(~correct_a & correct_b))

    # McNemar's test with continuity correction
    if b + c == 0:
        # No discordant pairs — models agree on everything
        return McNemarResult(
            statistic=0.0,
            p_value=1.0,
            n_discordant_a=b,
            n_discordant_b=c,
        )

    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1.0 - chi2.cdf(statistic, df=1)

    return McNemarResult(
        statistic=float(statistic),
        p_value=float(p_value),
        n_discordant_a=b,
        n_discordant_b=c,
    )


# ===================================================================
# Bootstrap Confidence Intervals
# ===================================================================

@dataclass(frozen=True, slots=True)
class BootstrapCIResult:
    """Result of bootstrap confidence interval for metric difference.

    Attributes
    ----------
    metric_name : str
        Name of the metric compared.
    observed_diff : float
        Observed difference: metric(A) - metric(B).
    ci_lower : float
        Lower bound of 95% CI.
    ci_upper : float
        Upper bound of 95% CI.
    p_value : float
        Approximate p-value (fraction of bootstrap diffs with opposite sign).
    n_bootstrap : int
        Number of bootstrap iterations.
    """

    metric_name: str
    observed_diff: float
    ci_lower: float
    ci_upper: float
    p_value: float
    n_bootstrap: int

    @property
    def is_significant(self) -> bool:
        """True if 95% CI excludes zero."""
        return self.ci_lower > 0.0 or self.ci_upper < 0.0


def _compute_metric(
    y_true: np.ndarray,
    scores: np.ndarray,
    metric_name: str,
) -> float:
    """Compute a single scalar metric from labels and scores.

    Supported metrics: 'pr_auc', 'f1'.
    """
    if metric_name == "pr_auc":
        precision, recall, _ = precision_recall_curve(y_true, scores)
        return float(auc(recall, precision))
    elif metric_name == "f1":
        preds = (scores > 0.5).astype(np.int32)
        return float(f1_score(y_true, preds, zero_division=0.0))
    else:
        raise ValueError(f"Unknown metric: {metric_name!r}. Supported: 'pr_auc', 'f1'.")


def bootstrap_metric_ci(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    metric_name: str = "pr_auc",
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> BootstrapCIResult:
    """Bootstrap confidence interval for the difference in a metric.

    Computes metric(A) - metric(B) on bootstrap resamples to produce
    a confidence interval and approximate p-value.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels, shape (n,).
    scores_a : np.ndarray
        Scores from system A, shape (n,).
    scores_b : np.ndarray
        Scores from system B, shape (n,).
    metric_name : str
        Metric to compare: 'pr_auc' or 'f1'.
    n_bootstrap : int
        Number of bootstrap resamples.
    confidence : float
        Confidence level for CI (default 0.95).
    seed : int
        Random seed.

    Returns
    -------
    BootstrapCIResult
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)

    # Observed difference
    obs_a = _compute_metric(y_true, scores_a, metric_name)
    obs_b = _compute_metric(y_true, scores_b, metric_name)
    observed_diff = obs_a - obs_b

    # Bootstrap
    diffs = np.zeros(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_boot = y_true[idx]

        # Skip degenerate resamples (all one class)
        if len(np.unique(y_boot)) < 2:
            diffs[i] = observed_diff
            continue

        s_a = scores_a[idx]
        s_b = scores_b[idx]
        m_a = _compute_metric(y_boot, s_a, metric_name)
        m_b = _compute_metric(y_boot, s_b, metric_name)
        diffs[i] = m_a - m_b

    # Confidence interval
    alpha = 1.0 - confidence
    ci_lower = float(np.percentile(diffs, 100 * alpha / 2))
    ci_upper = float(np.percentile(diffs, 100 * (1 - alpha / 2)))

    # Approximate p-value: fraction of bootstrap diffs on the opposite
    # side of zero from the observed diff
    if observed_diff > 0:
        p_value = float(np.mean(diffs <= 0))
    elif observed_diff < 0:
        p_value = float(np.mean(diffs >= 0))
    else:
        p_value = 1.0

    return BootstrapCIResult(
        metric_name=metric_name,
        observed_diff=observed_diff,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
        n_bootstrap=n_bootstrap,
    )


# ===================================================================
# ArmResult — all-in-one evaluation
# ===================================================================

@dataclass
class ArmResult:
    """Full evaluation results for a single treatment arm.

    Attributes
    ----------
    arm_name : str
        Identifier for this arm (e.g., 'A_majority_vote').
    pr_curve : PRCurveData or None
        PR curve data.
    fpr_at_tpr : FPRAtTPR or None
        FPR at fixed TPR operating points.
    expected_cost : float
        Mean cost per transaction.
    escalation : EscalationAnalysis or None
        Escalation metrics (three-way strategies only).
    """

    arm_name: str
    pr_curve: PRCurveData | None
    fpr_at_tpr: FPRAtTPR | None
    expected_cost: float
    escalation: EscalationAnalysis | None = None


def evaluate_arm(
    arm_name: str,
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
    escalation_mask: np.ndarray | None = None,
    cost_config: CostConfig | None = None,
) -> ArmResult:
    """Evaluate a single treatment arm on all metrics.

    Parameters
    ----------
    arm_name : str
        Arm identifier.
    y_true : np.ndarray
        True binary labels.
    scores : np.ndarray
        Fraud scores in [0, 1].
    threshold : float
        Decision threshold for cost and escalation metrics.
    escalation_mask : np.ndarray or None
        If provided, compute escalation analysis.
    cost_config : CostConfig or None
        Cost parameters.

    Returns
    -------
    ArmResult
    """
    pr = compute_pr_curve(y_true, scores)
    fpr = compute_fpr_at_tpr(y_true, scores)
    cost = compute_expected_cost(
        y_true, scores,
        threshold=threshold,
        escalation_mask=escalation_mask,
        cost_config=cost_config,
    )

    esc = None
    if escalation_mask is not None:
        esc = compute_escalation_analysis(
            y_true, scores, escalation_mask, threshold=threshold,
        )

    return ArmResult(
        arm_name=arm_name,
        pr_curve=pr,
        fpr_at_tpr=fpr,
        expected_cost=cost,
        escalation=esc,
    )
