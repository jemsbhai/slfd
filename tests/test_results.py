"""Tests for results file naming and saving utilities."""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import patch
from datetime import datetime, timezone

import pytest

from slfd.results import (
    make_result_filename,
    save_results,
    list_results,
)


# ===================================================================
# Tests: make_result_filename
# ===================================================================

class TestMakeResultFilename:
    """Test timestamped filename generation."""

    def test_contains_experiment_name(self) -> None:
        name = make_result_filename("efd2")
        assert "efd2" in name

    def test_ends_with_json(self) -> None:
        name = make_result_filename("efd2")
        assert name.endswith(".json")

    def test_contains_timestamp(self) -> None:
        name = make_result_filename("efd2")
        # Should contain a YYYYMMDD_HHMMSS pattern
        assert re.search(r"\d{8}_\d{6}", name)

    def test_different_experiment_names(self) -> None:
        name1 = make_result_filename("efd2")
        name2 = make_result_filename("efd3")
        assert "efd2" in name1
        assert "efd3" in name2

    def test_optional_suffix(self) -> None:
        name = make_result_filename("efd2", suffix="run1")
        assert "run1" in name
        assert name.endswith(".json")

    def test_no_suffix_by_default(self) -> None:
        name = make_result_filename("efd2")
        # Should be like efd2_20260218_143052.json — no extra suffix
        parts = name.replace(".json", "").split("_")
        # efd2, YYYYMMDD, HHMMSS
        assert len(parts) == 3

    def test_fixed_timestamp_for_reproducibility(self) -> None:
        """When a specific datetime is provided, the filename is deterministic."""
        dt = datetime(2026, 2, 18, 14, 30, 52, tzinfo=timezone.utc)
        name = make_result_filename("efd2", timestamp=dt)
        assert name == "efd2_20260218_143052.json"

    def test_fixed_timestamp_with_suffix(self) -> None:
        dt = datetime(2026, 2, 18, 14, 30, 52, tzinfo=timezone.utc)
        name = make_result_filename("efd2", suffix="diag", timestamp=dt)
        assert name == "efd2_20260218_143052_diag.json"


# ===================================================================
# Tests: save_results
# ===================================================================

class TestSaveResults:
    """Test saving results with automatic timestamped filenames."""

    def test_creates_file(self, tmp_path: Path) -> None:
        data = {"experiment": "efd2", "score": 0.95}
        filepath = save_results(data, experiment="efd2", results_dir=tmp_path)
        assert filepath.exists()

    def test_file_contains_valid_json(self, tmp_path: Path) -> None:
        data = {"experiment": "efd2", "score": 0.95}
        filepath = save_results(data, experiment="efd2", results_dir=tmp_path)
        loaded = json.loads(filepath.read_text())
        assert loaded["experiment"] == "efd2"
        assert loaded["score"] == 0.95

    def test_creates_directory_if_missing(self, tmp_path: Path) -> None:
        subdir = tmp_path / "nested" / "results"
        data = {"x": 1}
        filepath = save_results(data, experiment="efd2", results_dir=subdir)
        assert filepath.exists()

    def test_never_overwrites(self, tmp_path: Path) -> None:
        """Two rapid saves should produce two distinct files."""
        dt1 = datetime(2026, 2, 18, 14, 30, 52, tzinfo=timezone.utc)
        dt2 = datetime(2026, 2, 18, 14, 30, 53, tzinfo=timezone.utc)
        data1 = {"run": 1}
        data2 = {"run": 2}
        f1 = save_results(data1, experiment="efd2", results_dir=tmp_path, timestamp=dt1)
        f2 = save_results(data2, experiment="efd2", results_dir=tmp_path, timestamp=dt2)
        assert f1 != f2
        assert f1.exists()
        assert f2.exists()
        assert json.loads(f1.read_text())["run"] == 1
        assert json.loads(f2.read_text())["run"] == 2

    def test_same_timestamp_gets_deduplicated(self, tmp_path: Path) -> None:
        """If two saves happen in the same second, append a counter."""
        dt = datetime(2026, 2, 18, 14, 30, 52, tzinfo=timezone.utc)
        data1 = {"run": 1}
        data2 = {"run": 2}
        f1 = save_results(data1, experiment="efd2", results_dir=tmp_path, timestamp=dt)
        f2 = save_results(data2, experiment="efd2", results_dir=tmp_path, timestamp=dt)
        assert f1 != f2
        assert f1.exists()
        assert f2.exists()

    def test_returns_path_object(self, tmp_path: Path) -> None:
        data = {"x": 1}
        result = save_results(data, experiment="efd2", results_dir=tmp_path)
        assert isinstance(result, Path)

    def test_suffix_in_filename(self, tmp_path: Path) -> None:
        data = {"x": 1}
        filepath = save_results(
            data, experiment="efd2", results_dir=tmp_path, suffix="diag"
        )
        assert "diag" in filepath.name


# ===================================================================
# Tests: list_results
# ===================================================================

class TestListResults:
    """Test listing existing result files."""

    def test_empty_directory(self, tmp_path: Path) -> None:
        results = list_results(experiment="efd2", results_dir=tmp_path)
        assert results == []

    def test_finds_matching_files(self, tmp_path: Path) -> None:
        # Create some result files
        (tmp_path / "efd2_20260218_143052.json").write_text("{}")
        (tmp_path / "efd2_20260218_150000.json").write_text("{}")
        (tmp_path / "efd3_20260218_143052.json").write_text("{}")

        results = list_results(experiment="efd2", results_dir=tmp_path)
        assert len(results) == 2
        assert all("efd2" in r.name for r in results)

    def test_sorted_newest_first(self, tmp_path: Path) -> None:
        (tmp_path / "efd2_20260218_100000.json").write_text("{}")
        (tmp_path / "efd2_20260218_150000.json").write_text("{}")
        (tmp_path / "efd2_20260218_120000.json").write_text("{}")

        results = list_results(experiment="efd2", results_dir=tmp_path)
        names = [r.name for r in results]
        assert names[0] == "efd2_20260218_150000.json"
        assert names[-1] == "efd2_20260218_100000.json"

    def test_ignores_non_json(self, tmp_path: Path) -> None:
        (tmp_path / "efd2_20260218_143052.json").write_text("{}")
        (tmp_path / "efd2_readme.md").write_text("notes")
        results = list_results(experiment="efd2", results_dir=tmp_path)
        assert len(results) == 1
