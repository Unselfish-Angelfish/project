"""
visualise.py — Publication-ready result figures for the COS700 experiment.

Produces 12 figures saved to figures/experiment/:
  fig_01_jsd_noise.png          JSD heatmap — noise (78 features × 2 levels)
  fig_02_jsd_suppression.png    JSD heatmap — suppression (78 × 3)
  fig_03_jsd_dp.png             JSD heatmap — DP (78 × 6)
  fig_04_wass_noise.png         Wasserstein heatmap — noise
  fig_05_wass_suppression.png   Wasserstein heatmap — suppression
  fig_06_wass_dp.png            Wasserstein heatmap — DP
  fig_07_kl_noise.png           KL heatmap — noise (log colour scale)
  fig_08_kl_suppression.png     KL heatmap — suppression (log colour scale)
  fig_09_kl_dp.png              KL heatmap — DP (log colour scale)
  fig_10_dp_epsilon_curve.png   DP privacy-utility curve (JSD + Wasserstein vs ε)
  fig_11_spearman.png           Spearman ρ grouped bar chart with significance stars
  fig_12_summary.png            Master 3-panel summary figure
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    FIGURES_EXP_DIR,
    PLOT_STYLE,
    RESULTS_DIR,
)


def plot_heatmap(
    metrics_df: pd.DataFrame,
    gini_df: pd.DataFrame,
    technique: str,
    metric: str,
    outpath,
) -> None:
    """Stub — not yet implemented."""
    pass


def plot_dp_epsilon_curve(
    metrics_df: pd.DataFrame,
    gini_df: pd.DataFrame,
    outpath,
) -> None:
    """Stub — not yet implemented."""
    pass


def plot_spearman(spearman_df: pd.DataFrame, outpath) -> None:
    """Stub — not yet implemented."""
    pass


def plot_summary(
    metrics_df: pd.DataFrame,
    spearman_df: pd.DataFrame,
    outpath,
) -> None:
    """Stub — not yet implemented."""
    pass


def main() -> None:
    print("[visualise] not yet implemented")


if __name__ == "__main__":
    main()
