# SLFD — Subjective Logic Fraud Detection

**Uncertainty-Native Fraud Detection: Principled Multi-Source Signal Fusion with Subjective Logic**

## Overview

SLFD is a research framework that applies Subjective Logic (Jøsang, 2016) to fraud detection pipelines, replacing scalar fraud scores with structured opinions that explicitly represent belief, disbelief, uncertainty, and base rates.

**This is not a new fraud detection algorithm.** It is a data representation and fusion framework that makes existing fraud detection systems more transparent, auditable, and epistemically honest.

## Key Capabilities

- **Opinion construction** from heterogeneous signal sources (ML models, rule engines, database lookups)
- **Multi-source fusion** with mathematically principled operators (cumulative, averaging, robust)
- **Conflict detection** that identifies disagreement between sources
- **Three-way decisions** (block / approve / escalate) grounded in cost-sensitive boundaries
- **Temporal decay** that degrades stale signals toward uncertainty
- **Provenance chains** for regulatory audit (BSA/AML, EU 6AMLD, PSD2)

## Status

Pre-alpha research prototype. This accompanies a research paper in preparation.

## Installation

```bash
# Core only (numpy + pandas)
pip install -e .

# With ML dependencies
pip install -e ".[ml]"

# Full development environment
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest
```

## Citation

Paper in preparation. Citation details will be added upon publication.

## License

MIT
