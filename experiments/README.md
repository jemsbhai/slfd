# Experiments

Each subdirectory corresponds to an experiment from the research plan.

| Directory | Experiment | Description |
|-----------|-----------|-------------|
| `efd1_scalar_collapse/` | E-FD1 | Scalar collapse diagnostic |
| `efd2_fusion/` | E-FD2 | Multi-source fraud signal fusion (core result) |
| `efd3_temporal/` | E-FD3 | Temporal decay for signal aging |
| `efd4_conflict/` | E-FD4 | Conflict-driven escalation value |
| `efd5_byzantine/` | E-FD5 | Byzantine-robust fusion under adversarial signals |
| `efd6_provenance/` | E-FD6 | Provenance and audit trail |
| `efd7_compliance/` | E-FD7 | Regulatory compliance overlay |
| `efd8_scalability/` | E-FD8 | Scalability at financial transaction volumes |
| `efd9_dataquality/` | E-FD9 | Data quality gates via shape validation |
| `synthetic/` | E-FD-Synth | Synthetic multi-source generator |

## Running Experiments

Each experiment directory contains its own runner script. All experiments log
reproducibility metadata (seeds, package versions, hardware info) to `../results/`.
