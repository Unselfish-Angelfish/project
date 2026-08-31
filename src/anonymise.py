"""
anonymise.py — Anonymisation pipeline for CICIDS-2017 training set.

Techniques:
  apply_noise()       → Gaussian noise perturbation at σ ∈ {0.1, 0.5}
  apply_suppression() → Replace bottom-N features (by Gini) with training mean
  apply_dp()          → Laplace mechanism with IQR-based empirical sensitivity

Output: 11 parquet files saved to data/anonymised/
  train_noise_0.1.parquet
  train_noise_0.5.parquet
  train_suppression_0.25.parquet
  train_suppression_0.50.parquet
  train_suppression_0.75.parquet
  train_dp_0.01.parquet
  train_dp_0.1.parquet
  train_dp_0.5.parquet
  train_dp_1.0.parquet
  train_dp_5.0.parquet
  train_dp_10.0.parquet
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    ANONYMISED_DIR,
    DP_EPSILONS,
    LABEL_COL,
    NOISE_SIGMAS,
    PROCESSED_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
    SUPPRESSION_LEVELS,
)


def apply_noise(
    df: pd.DataFrame,
    sigma: float,
    feature_cols: list[str],
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Stub — not yet implemented."""
    pass


def apply_suppression(
    df: pd.DataFrame,
    level: float,
    feature_cols: list[str],
    gini_path: Path = RESULTS_DIR / "gini_importance.csv",
) -> tuple[pd.DataFrame, list[str]]:
    """Stub — not yet implemented. Returns (modified_df, suppressed_feature_names)."""
    pass


def apply_dp(
    df: pd.DataFrame,
    epsilon: float,
    feature_cols: list[str],
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Stub — not yet implemented."""
    pass


def save_variant(df: pd.DataFrame, name: str, anonymised_dir: Path = ANONYMISED_DIR) -> None:
    """Stub — not yet implemented."""
    pass


def main() -> None:
    print("[anonymise] not yet implemented")


if __name__ == "__main__":
    main()
