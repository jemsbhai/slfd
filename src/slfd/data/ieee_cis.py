"""IEEE-CIS Fraud Detection dataset loading and preprocessing.

Handles download via Kaggle API, loading, merging transaction + identity
tables, preprocessing (missing values, encoding, feature cleaning),
and stratified train/test splitting.

Dataset: https://www.kaggle.com/c/ieee-fraud-detection
~590K transactions, 394 transaction features + 41 identity features.
"""

from __future__ import annotations

import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Default data directory (project-level)
DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "ieee-cis"

# Kaggle competition identifier
_COMPETITION = "ieee-fraud-detection"

# Columns with excessive missing values to drop (threshold: >90% null)
_NULL_THRESHOLD = 0.90

# Numeric fill value for missing numerics
_NUMERIC_FILL = -999


@dataclass
class IEEECISData:
    """Preprocessed and split IEEE-CIS dataset.

    Attributes
    ----------
    X_train : np.ndarray
        Training features, shape (n_train, n_features).
    X_test : np.ndarray
        Test features, shape (n_test, n_features).
    y_train : np.ndarray
        Training labels (0/1), shape (n_train,).
    y_test : np.ndarray
        Test labels (0/1), shape (n_test,).
    feature_names : list[str]
        Feature column names matching X columns.
    """

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]


# ===================================================================
# Download
# ===================================================================

def download_ieee_cis(dest: Path | None = None) -> Path:
    """Download IEEE-CIS dataset via Kaggle API.

    Parameters
    ----------
    dest : Path or None
        Destination directory. Defaults to DATA_DIR.

    Returns
    -------
    Path
        Directory containing the extracted CSV files.
    """
    dest = dest or DATA_DIR
    dest.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    txn_file = dest / "train_transaction.csv"
    if txn_file.exists():
        return dest

    # Download competition files
    subprocess.run(
        [
            "kaggle", "competitions", "download",
            "-c", _COMPETITION,
            "-p", str(dest),
        ],
        check=True,
    )

    # Extract zip
    zip_path = dest / f"{_COMPETITION}.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest)
        zip_path.unlink()

    return dest


# ===================================================================
# Load raw
# ===================================================================

def load_raw(data_dir: Path | None = None) -> pd.DataFrame:
    """Load and merge transaction + identity CSVs.

    Left-joins identity onto transactions so all transactions are
    preserved even if identity data is missing.

    Parameters
    ----------
    data_dir : Path or None
        Directory containing the CSV files. Defaults to DATA_DIR.

    Returns
    -------
    pd.DataFrame
        Merged dataframe with all columns.

    Raises
    ------
    FileNotFoundError
        If transaction CSV is not found.
    """
    data_dir = data_dir or DATA_DIR
    txn_path = data_dir / "train_transaction.csv"
    id_path = data_dir / "train_identity.csv"

    if not txn_path.exists():
        raise FileNotFoundError(
            f"Transaction file not found at {txn_path}. "
            f"Run download_ieee_cis() first."
        )

    df_txn = pd.read_csv(txn_path)

    if id_path.exists():
        df_id = pd.read_csv(id_path)
        df = df_txn.merge(df_id, on="TransactionID", how="left")
    else:
        df = df_txn

    return df


# ===================================================================
# Preprocess
# ===================================================================

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and encode the merged dataframe.

    Steps:
        1. Drop TransactionID (identifier, not a feature)
        2. Drop columns with >90% missing values
        3. Fill remaining numeric NaN with sentinel (-999)
        4. Label-encode categorical (object) columns
        5. Verify no NaN or inf remain

    Parameters
    ----------
    df : pd.DataFrame
        Raw merged dataframe from load_raw().

    Returns
    -------
    pd.DataFrame
        Clean dataframe ready for modeling.
    """
    df = df.copy()

    # 1. Drop identifier
    if "TransactionID" in df.columns:
        df = df.drop(columns=["TransactionID"])

    # 2. Drop high-null columns
    null_fracs = df.isna().mean()
    high_null = null_fracs[null_fracs > _NULL_THRESHOLD].index.tolist()
    # Never drop the target
    high_null = [c for c in high_null if c != "isFraud"]
    df = df.drop(columns=high_null)

    # 3. Separate numeric and categorical
    cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # 4. Fill numeric NaN with sentinel
    df[num_cols] = df[num_cols].fillna(_NUMERIC_FILL)

    # 5. Label-encode categoricals
    for col in cat_cols:
        df[col] = df[col].fillna("__MISSING__")
        df[col] = df[col].astype("category").cat.codes.astype(np.int32)

    # 6. Replace any remaining inf
    df = df.replace([np.inf, -np.inf], _NUMERIC_FILL)

    return df


# ===================================================================
# Train/test split
# ===================================================================

def make_splits(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
) -> IEEECISData:
    """Stratified train/test split.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataframe with 'isFraud' column.
    test_size : float
        Fraction of data for test set.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    IEEECISData
    """
    target = "isFraud"
    feature_cols = [c for c in df.columns if c != target]

    X = df[feature_cols].values.astype(np.float32)
    y = df[target].values.astype(np.int32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    return IEEECISData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_cols,
    )
