"""Results file naming and saving utilities.

Provides timestamped, collision-safe result file management so that
experiment results are never accidentally overwritten.

Naming convention:
    {experiment}_{YYYYMMDD}_{HHMMSS}.json
    {experiment}_{YYYYMMDD}_{HHMMSS}_{suffix}.json

Examples:
    efd2_20260218_143052.json
    efd2_20260218_143052_diag.json
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


def make_result_filename(
    experiment: str,
    suffix: str | None = None,
    timestamp: datetime | None = None,
) -> str:
    """Generate a timestamped result filename.

    Parameters
    ----------
    experiment : str
        Experiment identifier (e.g. "efd2", "efd3").
    suffix : str or None
        Optional suffix appended before .json (e.g. "diag", "run1").
    timestamp : datetime or None
        Specific timestamp to use. If None, uses current UTC time.

    Returns
    -------
    str
        Filename like "efd2_20260218_143052.json".
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)

    ts_str = timestamp.strftime("%Y%m%d_%H%M%S")

    if suffix:
        return f"{experiment}_{ts_str}_{suffix}.json"
    return f"{experiment}_{ts_str}.json"


def save_results(
    data: dict,
    experiment: str,
    results_dir: Path | str,
    suffix: str | None = None,
    timestamp: datetime | None = None,
) -> Path:
    """Save results to a timestamped JSON file, never overwriting.

    If a file with the same name already exists (e.g. two saves in
    the same second), appends an incrementing counter to deduplicate.

    Parameters
    ----------
    data : dict
        JSON-serializable results data.
    experiment : str
        Experiment identifier.
    results_dir : Path or str
        Directory to save into. Created if it doesn't exist.
    suffix : str or None
        Optional suffix for the filename.
    timestamp : datetime or None
        Specific timestamp. If None, uses current UTC time.

    Returns
    -------
    Path
        The path to the saved file.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    filename = make_result_filename(experiment, suffix=suffix, timestamp=timestamp)
    filepath = Path(os.path.join(str(results_dir), filename))

    # Deduplicate if file already exists
    if filepath.exists():
        stem = filepath.stem  # e.g. "efd2_20260218_143052"
        counter = 2
        while True:
            deduped = results_dir / f"{stem}_{counter}.json"
            if not deduped.exists():
                filepath = deduped
                break
            counter += 1

    # Use a safe encoder that handles numpy types
    class _NumpySafeEncoder(json.JSONEncoder):
        def default(self, obj):
            import numpy as _np
            if isinstance(obj, (_np.integer,)):
                return int(obj)
            if isinstance(obj, (_np.floating,)):
                return float(obj)
            if isinstance(obj, _np.ndarray):
                return obj.tolist()
            return super().default(obj)

    filepath.write_text(json.dumps(data, indent=2, cls=_NumpySafeEncoder))
    return filepath


def list_results(
    experiment: str,
    results_dir: Path | str,
) -> list[Path]:
    """List existing result files for an experiment, newest first.

    Parameters
    ----------
    experiment : str
        Experiment identifier to filter by.
    results_dir : Path or str
        Directory to search.

    Returns
    -------
    list[Path]
        Matching files sorted by name descending (newest first).
    """
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return []

    matches = sorted(
        [
            f
            for f in results_dir.iterdir()
            if f.is_file()
            and f.suffix == ".json"
            and f.name.startswith(f"{experiment}_")
        ],
        key=lambda p: p.name,
        reverse=True,
    )
    return matches
