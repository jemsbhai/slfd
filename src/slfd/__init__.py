"""SLFD — Subjective Logic Fraud Detection.

Uncertainty-native fraud detection via principled multi-source signal fusion
with Subjective Logic.
"""

__version__ = "0.1.0"

from slfd.opinion import Opinion
from slfd.fusion import cumulative_fuse, averaging_fuse, conflict_metric
from slfd.decay import decay_opinion
from slfd.trust import trust_discount
from slfd.robust import robust_fuse, RobustFuseResult
from slfd.synthetic import FraudScenarioGenerator, FraudDataset
from slfd.decision import (
    Decision,
    DecisionResult,
    ThreeWayDecider,
    CostMatrix,
    compute_optimal_thresholds,
)
from slfd.strategies import (
    ThreeWayFusionResult,
    majority_vote,
    weighted_average,
    compute_accuracy_weights,
    StackingMetaLearner,
    bayesian_model_average,
    compute_bma_weights,
    noisy_or,
    sl_cumulative_scores,
    sl_three_way,
    sl_robust_three_way,
    ConfidenceFeatureLearner,
)
from slfd.metrics import (
    PRCurveData,
    compute_pr_curve,
    FPRAtTPR,
    compute_fpr_at_tpr,
    EscalationAnalysis,
    compute_escalation_analysis,
    CostConfig,
    compute_expected_cost,
    McNemarResult,
    mcnemar_test,
    BootstrapCIResult,
    bootstrap_metric_ci,
    ArmResult,
    evaluate_arm,
)

__all__ = [
    "Opinion",
    "cumulative_fuse",
    "averaging_fuse",
    "conflict_metric",
    "decay_opinion",
    "trust_discount",
    "robust_fuse",
    "RobustFuseResult",
    "FraudScenarioGenerator",
    "FraudDataset",
    "Decision",
    "DecisionResult",
    "ThreeWayDecider",
    "CostMatrix",
    "compute_optimal_thresholds",
    "ThreeWayFusionResult",
    "majority_vote",
    "weighted_average",
    "compute_accuracy_weights",
    "StackingMetaLearner",
    "bayesian_model_average",
    "compute_bma_weights",
    "noisy_or",
    "sl_cumulative_scores",
    "sl_three_way",
    "sl_robust_three_way",
    "ConfidenceFeatureLearner",
    "PRCurveData",
    "compute_pr_curve",
    "FPRAtTPR",
    "compute_fpr_at_tpr",
    "EscalationAnalysis",
    "compute_escalation_analysis",
    "CostConfig",
    "compute_expected_cost",
    "McNemarResult",
    "mcnemar_test",
    "BootstrapCIResult",
    "bootstrap_metric_ci",
    "ArmResult",
    "evaluate_arm",
]
