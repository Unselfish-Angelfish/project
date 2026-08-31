"""
metrics.py — Divergence metric computation and Spearman ρ analysis.

Functions:
  get_bin_edges()          → n_bins+1 edges spanning original feature range
  compute_histogram()      → normalised histogram from values + edges
  compute_jsd()            → Jensen-Shannon Divergence squared ∈ [0,1]
  compute_wasserstein()    → Wasserstein distance normalised by feature std
  compute_kl()             → KL Divergence with additive smoothing (unbounded)
  compute_all_metrics()    → per-feature JSD/Wasserstein/KL for one variant
  compute_spearman_table() → Spearman ρ per (technique, level, metric)
  main()                   → iterate all 11 variants, save metrics.csv + spearman.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.spatial.distance
import scipy.special
import scipy.stats

from src.config import (
    ANONYMISED_DIR,
    KL_SMOOTH,
    N_BINS,
    PROCESSED_DIR,
    RESULTS_DIR,
)


def get_bin_edges(values: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """Stub — not yet implemented."""
    pass


def compute_histogram(values: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Stub — not yet implemented."""
    pass


def compute_jsd(p: np.ndarray, q: np.ndarray, smooth: float = KL_SMOOTH) -> float:
    """Stub — not yet implemented."""
    pass


def compute_wasserstein(orig: np.ndarray, anon: np.ndarray) -> float:
    """Stub — not yet implemented."""
    pass


def compute_kl(p: np.ndarray, q: np.ndarray, smooth: float = KL_SMOOTH) -> float:
    """Stub — not yet implemented."""
    pass


def compute_all_metrics(
    orig_df: pd.DataFrame,
    anon_df: pd.DataFrame,
    feature_cols: list[str],
    n_bins: int = N_BINS,
) -> pd.DataFrame:
    """Stub — not yet implemented."""
    pass


def compute_spearman_table(
    metrics_df: pd.DataFrame,
    gini_df: pd.DataFrame,
) -> pd.DataFrame:
    """Stub — not yet implemented."""
    pass


def main() -> None:
    print("[metrics] not yet implemented")


if __name__ == "__main__":
    main()
