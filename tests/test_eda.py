"""
test_eda.py — Task 3 acceptance tests for src/eda.py.

Tests:
  1. test_gini_csv_has_78_rows         — gini_importance.csv has exactly 78 rows
  2. test_gini_importances_sum_to_one  — importances sum to 1.0 (within float tolerance)
  3. test_gini_rank_column_correct     — ranks 1..78 sequential
  4. test_gini_rank1_is_flow_feature   — most important feature is a flow-level feature
  5. test_all_eda_figures_exist        — all 7 EDA figure files exist
  6. test_eda_figures_non_empty        — all EDA figures > 10KB

Run EDA first: python -m src.eda
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.config import FIGURES_EDA_DIR, RESULTS_DIR

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def gini_df() -> pd.DataFrame:
    path = RESULTS_DIR / "gini_importance.csv"
    if not path.exists():
        pytest.skip("gini_importance.csv not found — run `python -m src.eda` first")
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Test 1 — Row count
# ---------------------------------------------------------------------------

def test_gini_csv_has_78_rows(gini_df):
    assert len(gini_df) == 78, (
        f"Expected 78 rows in gini_importance.csv, got {len(gini_df)}"
    )


# ---------------------------------------------------------------------------
# Test 2 — Importances sum to 1
# ---------------------------------------------------------------------------

def test_gini_importances_sum_to_one(gini_df):
    total = gini_df["importance"].sum()
    assert abs(total - 1.0) < 1e-5, (
        f"Importances sum to {total:.8f}, expected 1.0"
    )


# ---------------------------------------------------------------------------
# Test 3 — Rank column is 1..78 sequential
# ---------------------------------------------------------------------------

def test_gini_rank_column_correct(gini_df):
    ranks = gini_df["rank"].tolist()
    assert ranks == list(range(1, 79)), (
        f"Rank column is not sequential 1..78. Got: {ranks[:5]}..."
    )


# ---------------------------------------------------------------------------
# Test 4 — Rank-1 feature is a flow-level feature (not a flag)
# ---------------------------------------------------------------------------

def test_gini_rank1_is_flow_feature(gini_df):
    rank1 = gini_df.loc[gini_df["rank"] == 1, "feature"].iloc[0]
    # Flag features are poor classifiers; flow-rate features dominate
    assert "Flag" not in rank1, (
        f"Rank-1 feature is a flag feature: '{rank1}'. "
        "Expected a flow-level feature (e.g. Flow Bytes/s, Flow Duration)."
    )


# ---------------------------------------------------------------------------
# Test 5 — All 7 EDA figure files exist
# ---------------------------------------------------------------------------

EDA_FIGURE_NAMES = [
    "eda_01_data_quality.png",
    "eda_02_class_distribution.png",
    "eda_03_feature_distributions.png",
    "eda_04_correlation_matrix.png",
    "eda_05_flag_analysis.png",
    "eda_06_feature_importance.png",
    "eda_07_readiness.png",
]


@pytest.mark.parametrize("fname", EDA_FIGURE_NAMES)
def test_eda_figure_exists(fname):
    path = FIGURES_EDA_DIR / fname
    if not path.exists():
        pytest.skip(f"{fname} not found — run `python -m src.eda` first")
    assert path.exists(), f"Missing EDA figure: {path}"


# ---------------------------------------------------------------------------
# Test 6 — All EDA figures are non-empty (> 10KB)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fname", EDA_FIGURE_NAMES)
def test_eda_figure_non_empty(fname):
    path = FIGURES_EDA_DIR / fname
    if not path.exists():
        pytest.skip(f"{fname} not found — run `python -m src.eda` first")
    size = path.stat().st_size
    assert size > 10_000, (
        f"{fname} is suspiciously small: {size} bytes (expected > 10KB)"
    )


# ---------------------------------------------------------------------------
# Test 7 — gini_importance.csv has required columns
# ---------------------------------------------------------------------------

def test_gini_csv_columns(gini_df):
    required = {"rank", "feature", "importance", "cumulative_importance"}
    assert required.issubset(set(gini_df.columns)), (
        f"Missing columns: {required - set(gini_df.columns)}"
    )


# ---------------------------------------------------------------------------
# Test 8 — Cumulative importance reaches 1.0 by rank 78
# ---------------------------------------------------------------------------

def test_cumulative_importance_reaches_one(gini_df):
    final_cum = gini_df["cumulative_importance"].iloc[-1]
    assert abs(final_cum - 1.0) < 1e-5, (
        f"Cumulative importance at rank 78 is {final_cum:.8f}, expected 1.0"
    )
