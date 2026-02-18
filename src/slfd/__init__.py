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
]
