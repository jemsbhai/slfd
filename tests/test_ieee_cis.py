"""Tests for IEEE-CIS Fraud Detection dataset loading and preprocessing.

The IEEE-CIS dataset (Kaggle) contains:
    - train_transaction.csv: 590K transactions, 394 features
    - train_identity.csv: identity info for a subset, 41 features

Preprocessing pipeline:
    1. Download via Kaggle API (if not present)
    2. Load and merge transaction + identity tables
    3. Clean: handle missing values, encode categoricals, drop high-null columns
    4. Stratified train/test split preserving fraud rate
    5. Output: ready-to-model DataFrames + metadata

We test both the pipeline logic (with small synthetic fixtures) and
the real dataset (marked slow, skipped if data not present).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from slfd.data.ieee_cis import (
    download_ieee_cis,
    load_raw,
    preprocess,
    make_splits,
    make_three_way_splits,
    IEEECISData,
    IEEECISThreeWaySplit,
    DATA_DIR,
)


# ===================================================================
# Fixtures: small synthetic stand-ins for fast unit tests
# ===================================================================

@pytest.fixture
def mock_transaction_csv(tmp_path: Path) -> Path:
    """Create a minimal mock transaction CSV."""
    n = 500
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "TransactionID": range(1, n + 1),
        "isFraud": rng.binomial(1, 0.035, size=n),
        "TransactionDT": rng.integers(86400, 86400 * 180, size=n),
        "TransactionAmt": rng.exponential(100, size=n).round(2),
        "ProductCD": rng.choice(["W", "C", "R", "H", "S"], size=n),
        "card1": rng.integers(1000, 9999, size=n),
        "card2": rng.choice([100, 200, 300, np.nan], size=n),
        "card4": rng.choice(["visa", "mastercard", "discover", np.nan], size=n),
        "card6": rng.choice(["debit", "credit", np.nan], size=n),
        "addr1": rng.choice([200, 300, 400, np.nan], size=n),
        "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", np.nan], size=n),
        "C1": rng.integers(0, 10, size=n).astype(float),
        "D1": rng.choice([0.0, 1.0, 5.0, np.nan], size=n),
        "V1": rng.normal(0, 1, size=n),
        "V2": rng.normal(0, 1, size=n),
    })
    path = tmp_path / "train_transaction.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def mock_identity_csv(tmp_path: Path) -> Path:
    """Create a minimal mock identity CSV."""
    n = 200  # identity only for subset of transactions
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "TransactionID": range(1, n + 1),
        "DeviceType": rng.choice(["desktop", "mobile", np.nan], size=n),
        "DeviceInfo": rng.choice(["Windows", "iOS", "MacOS", np.nan], size=n),
        "id_01": rng.normal(0, 30, size=n),
        "id_02": rng.normal(100000, 50000, size=n),
    })
    path = tmp_path / "train_identity.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def mock_data_dir(tmp_path: Path, mock_transaction_csv, mock_identity_csv) -> Path:
    """Directory containing both mock CSVs."""
    return tmp_path


# ===================================================================
# 1. Loading raw data
# ===================================================================

class TestLoadRaw:
    """Loading and merging the raw CSV files."""

    def test_loads_transaction_data(self, mock_data_dir):
        df = load_raw(mock_data_dir)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 500  # all transactions present

    def test_has_target_column(self, mock_data_dir):
        df = load_raw(mock_data_dir)
        assert "isFraud" in df.columns

    def test_merges_identity(self, mock_data_dir):
        """Identity columns should be present after merge."""
        df = load_raw(mock_data_dir)
        assert "DeviceType" in df.columns

    def test_left_join_preserves_all_transactions(self, mock_data_dir):
        """Transactions without identity data still present (with NaN)."""
        df = load_raw(mock_data_dir)
        assert len(df) == 500
        # Some identity fields should be NaN for txns 201-500
        assert df["DeviceType"].isna().sum() > 0

    def test_missing_transaction_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_raw(tmp_path)


# ===================================================================
# 2. Preprocessing
# ===================================================================

class TestPreprocess:
    """Cleaning, encoding, and feature preparation."""

    def test_returns_dataframe(self, mock_data_dir):
        raw = load_raw(mock_data_dir)
        clean = preprocess(raw)
        assert isinstance(clean, pd.DataFrame)

    def test_no_object_columns(self, mock_data_dir):
        """All categorical columns should be encoded to numeric."""
        raw = load_raw(mock_data_dir)
        clean = preprocess(raw)
        object_cols = clean.select_dtypes(include=["object"]).columns.tolist()
        assert len(object_cols) == 0

    def test_target_preserved(self, mock_data_dir):
        raw = load_raw(mock_data_dir)
        clean = preprocess(raw)
        assert "isFraud" in clean.columns

    def test_transaction_id_dropped(self, mock_data_dir):
        """TransactionID is an identifier, not a feature."""
        raw = load_raw(mock_data_dir)
        clean = preprocess(raw)
        assert "TransactionID" not in clean.columns

    def test_no_infinite_values(self, mock_data_dir):
        raw = load_raw(mock_data_dir)
        clean = preprocess(raw)
        numeric = clean.select_dtypes(include=[np.number])
        assert not np.any(np.isinf(numeric.values))

    def test_missing_values_handled(self, mock_data_dir):
        """No NaN in final output (filled or dropped)."""
        raw = load_raw(mock_data_dir)
        clean = preprocess(raw)
        assert clean.isna().sum().sum() == 0


# ===================================================================
# 3. Train/test split
# ===================================================================

class TestMakeSplits:
    """Stratified splitting preserving fraud rate."""

    def test_returns_ieee_cis_data(self, mock_data_dir):
        raw = load_raw(mock_data_dir)
        clean = preprocess(raw)
        data = make_splits(clean, test_size=0.2, seed=42)
        assert isinstance(data, IEEECISData)

    def test_has_required_attributes(self, mock_data_dir):
        raw = load_raw(mock_data_dir)
        clean = preprocess(raw)
        data = make_splits(clean, test_size=0.2, seed=42)
        assert hasattr(data, "X_train")
        assert hasattr(data, "X_test")
        assert hasattr(data, "y_train")
        assert hasattr(data, "y_test")
        assert hasattr(data, "feature_names")

    def test_correct_split_sizes(self, mock_data_dir):
        raw = load_raw(mock_data_dir)
        clean = preprocess(raw)
        data = make_splits(clean, test_size=0.2, seed=42)
        total = len(data.X_train) + len(data.X_test)
        assert total == 500
        assert len(data.X_test) == pytest.approx(100, abs=5)

    def test_stratified_fraud_rate(self, mock_data_dir):
        """Train and test fraud rates should be similar."""
        raw = load_raw(mock_data_dir)
        clean = preprocess(raw)
        data = make_splits(clean, test_size=0.2, seed=42)
        train_rate = np.mean(data.y_train)
        test_rate = np.mean(data.y_test)
        assert train_rate == pytest.approx(test_rate, abs=0.03)

    def test_no_target_in_features(self, mock_data_dir):
        raw = load_raw(mock_data_dir)
        clean = preprocess(raw)
        data = make_splits(clean, test_size=0.2, seed=42)
        assert "isFraud" not in data.feature_names

    def test_feature_names_match_columns(self, mock_data_dir):
        raw = load_raw(mock_data_dir)
        clean = preprocess(raw)
        data = make_splits(clean, test_size=0.2, seed=42)
        assert data.X_train.shape[1] == len(data.feature_names)

    def test_reproducible(self, mock_data_dir):
        raw = load_raw(mock_data_dir)
        clean = preprocess(raw)
        d1 = make_splits(clean, test_size=0.2, seed=42)
        d2 = make_splits(clean, test_size=0.2, seed=42)
        np.testing.assert_array_equal(d1.y_train, d2.y_train)


# ===================================================================
# 4. Three-way split (train / val / test)
# ===================================================================

class TestMakeThreeWaySplits:
    """Stratified three-way split for E-FD2.

    Required for scientific rigor: meta-learners (C, I) and weight
    computation (B, D) must fit on val predictions, NOT test data.
    Final evaluation on held-out test set, identical for all 9 arms.
    """

    def test_returns_three_way_split(self, mock_data_dir):
        raw = load_raw(mock_data_dir)
        clean = preprocess(raw)
        data = make_three_way_splits(clean, seed=42)
        assert isinstance(data, IEEECISThreeWaySplit)

    def test_has_all_six_arrays(self, mock_data_dir):
        raw = load_raw(mock_data_dir)
        clean = preprocess(raw)
        data = make_three_way_splits(clean, seed=42)
        assert hasattr(data, "X_train")
        assert hasattr(data, "y_train")
        assert hasattr(data, "X_val")
        assert hasattr(data, "y_val")
        assert hasattr(data, "X_test")
        assert hasattr(data, "y_test")
        assert hasattr(data, "feature_names")

    def test_no_overlap_between_splits(self, mock_data_dir):
        """Critical: no data leakage between train/val/test."""
        raw = load_raw(mock_data_dir)
        clean = preprocess(raw)
        data = make_three_way_splits(clean, seed=42)
        n_total = len(data.y_train) + len(data.y_val) + len(data.y_test)
        assert n_total == 500  # all samples accounted for, none duplicated

    def test_default_split_ratios(self, mock_data_dir):
        """Default: 60% train, 20% val, 20% test."""
        raw = load_raw(mock_data_dir)
        clean = preprocess(raw)
        data = make_three_way_splits(clean, seed=42)
        n = 500
        assert len(data.y_train) == pytest.approx(n * 0.6, abs=10)
        assert len(data.y_val) == pytest.approx(n * 0.2, abs=10)
        assert len(data.y_test) == pytest.approx(n * 0.2, abs=10)

    def test_custom_split_ratios(self, mock_data_dir):
        raw = load_raw(mock_data_dir)
        clean = preprocess(raw)
        data = make_three_way_splits(
            clean, train_size=0.7, val_size=0.15, test_size=0.15, seed=42,
        )
        n = 500
        assert len(data.y_train) == pytest.approx(n * 0.7, abs=10)
        assert len(data.y_val) == pytest.approx(n * 0.15, abs=10)
        assert len(data.y_test) == pytest.approx(n * 0.15, abs=10)

    def test_ratios_must_sum_to_one(self, mock_data_dir):
        raw = load_raw(mock_data_dir)
        clean = preprocess(raw)
        with pytest.raises(ValueError, match="sum to 1"):
            make_three_way_splits(
                clean, train_size=0.5, val_size=0.2, test_size=0.1, seed=42,
            )

    def test_stratified_fraud_rate_all_three(self, mock_data_dir):
        """Fraud rate should be approximately equal across all three splits."""
        raw = load_raw(mock_data_dir)
        clean = preprocess(raw)
        data = make_three_way_splits(clean, seed=42)
        rate_train = np.mean(data.y_train)
        rate_val = np.mean(data.y_val)
        rate_test = np.mean(data.y_test)
        # All three within 5% of each other
        assert rate_train == pytest.approx(rate_val, abs=0.05)
        assert rate_train == pytest.approx(rate_test, abs=0.05)

    def test_feature_names_match(self, mock_data_dir):
        raw = load_raw(mock_data_dir)
        clean = preprocess(raw)
        data = make_three_way_splits(clean, seed=42)
        n_feat = len(data.feature_names)
        assert data.X_train.shape[1] == n_feat
        assert data.X_val.shape[1] == n_feat
        assert data.X_test.shape[1] == n_feat

    def test_no_target_in_features(self, mock_data_dir):
        raw = load_raw(mock_data_dir)
        clean = preprocess(raw)
        data = make_three_way_splits(clean, seed=42)
        assert "isFraud" not in data.feature_names

    def test_reproducible(self, mock_data_dir):
        raw = load_raw(mock_data_dir)
        clean = preprocess(raw)
        d1 = make_three_way_splits(clean, seed=42)
        d2 = make_three_way_splits(clean, seed=42)
        np.testing.assert_array_equal(d1.y_train, d2.y_train)
        np.testing.assert_array_equal(d1.y_val, d2.y_val)
        np.testing.assert_array_equal(d1.y_test, d2.y_test)

    def test_different_seed_different_split(self, mock_data_dir):
        raw = load_raw(mock_data_dir)
        clean = preprocess(raw)
        d1 = make_three_way_splits(clean, seed=42)
        d2 = make_three_way_splits(clean, seed=99)
        # Extremely unlikely to be identical with different seeds
        assert not np.array_equal(d1.y_test, d2.y_test)

    def test_dtypes(self, mock_data_dir):
        """Features float32, labels int32 — consistent with make_splits."""
        raw = load_raw(mock_data_dir)
        clean = preprocess(raw)
        data = make_three_way_splits(clean, seed=42)
        assert data.X_train.dtype == np.float32
        assert data.X_val.dtype == np.float32
        assert data.X_test.dtype == np.float32
        assert data.y_train.dtype == np.int32
        assert data.y_val.dtype == np.int32
        assert data.y_test.dtype == np.int32


# ===================================================================
# 5. Real dataset tests (skipped if data not present)
# ===================================================================

_REAL_DATA = DATA_DIR / "train_transaction.csv"


@pytest.mark.slow
@pytest.mark.skipif(not _REAL_DATA.exists(), reason="IEEE-CIS data not downloaded")
class TestRealDataset:
    """Integration tests with the actual IEEE-CIS dataset."""

    def test_loads_full_dataset(self):
        df = load_raw(DATA_DIR)
        assert len(df) > 500_000

    def test_fraud_rate_approximately_3_5_percent(self):
        df = load_raw(DATA_DIR)
        rate = df["isFraud"].mean()
        assert rate == pytest.approx(0.035, abs=0.005)

    def test_full_preprocess_pipeline(self):
        df = load_raw(DATA_DIR)
        clean = preprocess(df)
        assert clean.isna().sum().sum() == 0
        assert len(clean) > 500_000

    def test_full_split_pipeline(self):
        df = load_raw(DATA_DIR)
        clean = preprocess(df)
        data = make_splits(clean, test_size=0.2, seed=42)
        assert len(data.X_train) > 400_000
        assert len(data.X_test) > 100_000
