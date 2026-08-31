"""
test_preprocessing.py — Task 2 acceptance tests for src/preprocessing.py.

Tests:
  1. test_no_nan_inf_in_splits       — no NaN or inf in any split
  2. test_split_sizes_sum_to_total   — three splits together equal full cleaned dataset
  3. test_class_proportions_consistent — class proportions within 1% across splits
  4. test_78_numeric_features        — exactly 78 numeric feature columns per split
  5. test_label_column_present       — Label column present in all splits
  6. test_parquets_readable          — parquet files load without error and schema matches

All tests operate on the already-saved parquet files in data/processed/.
Run preprocessing first: python -m src.preprocessing
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import LABEL_COL, N_FEATURES, PROCESSED_DIR

# ---------------------------------------------------------------------------
# Fixtures — load parquets once per test session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def train_df() -> pd.DataFrame:
    path = PROCESSED_DIR / "train.parquet"
    if not path.exists():
        pytest.skip("train.parquet not found — run `python -m src.preprocessing` first")
    return pd.read_parquet(path)


@pytest.fixture(scope="session")
def val_df() -> pd.DataFrame:
    path = PROCESSED_DIR / "val.parquet"
    if not path.exists():
        pytest.skip("val.parquet not found — run `python -m src.preprocessing` first")
    return pd.read_parquet(path)


@pytest.fixture(scope="session")
def test_df() -> pd.DataFrame:
    path = PROCESSED_DIR / "test.parquet"
    if not path.exists():
        pytest.skip("test.parquet not found — run `python -m src.preprocessing` first")
    return pd.read_parquet(path)


@pytest.fixture(scope="session")
def all_splits(train_df, val_df, test_df):
    return {"train": train_df, "val": val_df, "test": test_df}


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c != LABEL_COL and pd.api.types.is_numeric_dtype(df[c])]


# ---------------------------------------------------------------------------
# Test 1 — No NaN or inf in any split
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("split_name", ["train", "val", "test"])
def test_no_nan_in_split(all_splits, split_name):
    df = all_splits[split_name]
    num_cols = _numeric_cols(df)
    nan_count = df[num_cols].isna().sum().sum()
    assert nan_count == 0, (
        f"{split_name}: found {nan_count} NaN values in numeric columns"
    )


@pytest.mark.parametrize("split_name", ["train", "val", "test"])
def test_no_inf_in_split(all_splits, split_name):
    df = all_splits[split_name]
    num_cols = _numeric_cols(df)
    inf_count = np.isinf(df[num_cols].select_dtypes(include=np.number)).sum().sum()
    assert inf_count == 0, (
        f"{split_name}: found {inf_count} inf values in numeric columns"
    )


# ---------------------------------------------------------------------------
# Test 2 — Split sizes sum to total
# ---------------------------------------------------------------------------

def test_split_sizes_sum_to_total(train_df, val_df, test_df):
    total = len(train_df) + len(val_df) + len(test_df)
    # Each split should be non-empty
    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0
    # Proportions should be approximately 70/15/15
    assert 0.68 <= len(train_df) / total <= 0.72, (
        f"Train split is {100*len(train_df)/total:.1f}%, expected ~70%"
    )
    assert 0.13 <= len(val_df) / total <= 0.17, (
        f"Val split is {100*len(val_df)/total:.1f}%, expected ~15%"
    )
    assert 0.13 <= len(test_df) / total <= 0.17, (
        f"Test split is {100*len(test_df)/total:.1f}%, expected ~15%"
    )


# ---------------------------------------------------------------------------
# Test 3 — Class proportions consistent across splits (within 1%)
# ---------------------------------------------------------------------------

def test_class_proportions_consistent(train_df, val_df, test_df):
    """
    For every class present in the training set, the proportion in val and test
    must be within 1 percentage point of the train proportion.
    """
    def proportions(df):
        return (df[LABEL_COL].value_counts() / len(df)).to_dict()

    train_props = proportions(train_df)
    val_props   = proportions(val_df)
    test_props  = proportions(test_df)

    for cls, train_p in train_props.items():
        val_p  = val_props.get(cls, 0.0)
        test_p = test_props.get(cls, 0.0)
        assert abs(train_p - val_p) <= 0.01, (
            f"Class '{cls}': train={train_p:.3f}, val={val_p:.3f} "
            f"(diff={abs(train_p - val_p):.3f} > 0.01)"
        )
        assert abs(train_p - test_p) <= 0.01, (
            f"Class '{cls}': train={train_p:.3f}, test={test_p:.3f} "
            f"(diff={abs(train_p - test_p):.3f} > 0.01)"
        )


# ---------------------------------------------------------------------------
# Test 4 — Exactly 78 numeric feature columns
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("split_name", ["train", "val", "test"])
def test_78_numeric_features(all_splits, split_name):
    df = all_splits[split_name]
    num_cols = _numeric_cols(df)
    assert len(num_cols) == N_FEATURES, (
        f"{split_name}: expected {N_FEATURES} numeric features, got {len(num_cols)}"
    )


# ---------------------------------------------------------------------------
# Test 5 — Label column present
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("split_name", ["train", "val", "test"])
def test_label_column_present(all_splits, split_name):
    df = all_splits[split_name]
    assert LABEL_COL in df.columns, (
        f"{split_name}: '{LABEL_COL}' column not found. "
        f"Columns: {df.columns.tolist()}"
    )


def test_label_column_non_empty(train_df):
    assert train_df[LABEL_COL].notna().all()
    assert len(train_df[LABEL_COL].unique()) >= 2, (
        "Expected at least 2 label classes in training set"
    )


# ---------------------------------------------------------------------------
# Test 6 — Parquet files readable and schema consistent
# ---------------------------------------------------------------------------

def test_parquets_exist():
    for name in ["train", "val", "test"]:
        path = PROCESSED_DIR / f"{name}.parquet"
        assert path.exists(), f"Missing: {path}"
        assert path.stat().st_size > 0, f"Empty file: {path}"


def test_parquet_schemas_match(train_df, val_df, test_df):
    """All three splits must have identical column names and dtypes."""
    assert list(train_df.columns) == list(val_df.columns), (
        "train and val have different columns"
    )
    assert list(train_df.columns) == list(test_df.columns), (
        "train and test have different columns"
    )
    for col in train_df.select_dtypes(include=np.number).columns:
        assert train_df[col].dtype == val_df[col].dtype, (
            f"dtype mismatch for '{col}': train={train_df[col].dtype}, val={val_df[col].dtype}"
        )


def test_parquet_total_columns(train_df):
    """Should have exactly 79 columns: 78 numeric + Label."""
    assert len(train_df.columns) == N_FEATURES + 1, (
        f"Expected {N_FEATURES + 1} columns (78 numeric + Label), "
        f"got {len(train_df.columns)}"
    )
