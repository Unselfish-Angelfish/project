"""
eda.py — Exploratory Data Analysis for CICIDS-2017 (all 7 steps).

All analysis uses the training set only to avoid data leakage.

Steps:
  eda_01_data_quality()          → missing/zero-rate per feature + skewness chart
  eda_02_class_distribution()    → class counts on linear and log scale
  eda_03_feature_distributions() → 2×4 KDE grid for 8 representative features by class
  eda_04_correlation_matrix()    → 78×78 Pearson r heatmap (upper triangle)
  eda_05_flag_analysis()         → TCP flag mean-rate heatmap + grouped bar chart
  eda_06_feature_importance()    → RF Gini top-20 + cumulative curve
                                    saves results/gini_importance.csv
  eda_07_readiness()             → 4-panel anonymisation readiness summary
  main()                         → run all 7 steps in sequence
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must be set before pyplot import
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

from src.config import (
    FIGURES_EDA_DIR,
    LABEL_COL,
    PLOT_STYLE,
    PROCESSED_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Shared style helper
# ---------------------------------------------------------------------------

def _apply_dark_style(fig, axes_list: list) -> None:
    """Apply the project dark theme to a figure and its axes."""
    bg   = PLOT_STYLE["bg_color"]
    fg   = PLOT_STYLE["fg_color"]
    grid = PLOT_STYLE["grid_color"]

    fig.patch.set_facecolor(bg)
    for ax in axes_list:
        if ax is None:
            continue
        ax.set_facecolor(bg)
        ax.tick_params(colors=fg, labelsize=PLOT_STYLE["font_size"] - 1)
        ax.xaxis.label.set_color(fg)
        ax.yaxis.label.set_color(fg)
        ax.title.set_color(fg)
        for spine in ax.spines.values():
            spine.set_edgecolor(grid)
        ax.grid(color=grid, linewidth=0.5)


def _save(fig, filename: str) -> Path:
    """Save figure to figures/eda/ at configured DPI, close it, return path."""
    out = FIGURES_EDA_DIR / filename
    fig.savefig(out, dpi=PLOT_STYLE["dpi"], bbox_inches="tight",
                facecolor=PLOT_STYLE["bg_color"])
    plt.close(fig)
    print(f"    Saved → {out.name}")
    return out


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != LABEL_COL
            and pd.api.types.is_numeric_dtype(df[c])]


# ---------------------------------------------------------------------------
# EDA 1 — Data quality audit
# ---------------------------------------------------------------------------

def eda_01_data_quality(train: pd.DataFrame) -> None:
    """
    Two-panel figure:
    Left:  missing % and zero-rate per feature (grouped horizontal bars),
           sorted by zero-rate descending.
    Right: |skewness| per feature, colour-coded by severity,
           sorted by |skewness| descending.
    """
    print("  EDA 1 — Data quality audit")
    feat_cols = _feature_cols(train)
    bg = PLOT_STYLE["bg_color"]
    fg = PLOT_STYLE["fg_color"]

    # Compute stats
    missing_pct  = train[feat_cols].isna().mean() * 100
    zero_rate    = (train[feat_cols] == 0).mean() * 100
    skewness     = train[feat_cols].skew().abs()

    # Sort by zero-rate for left panel
    sort_zero = zero_rate.sort_values(ascending=True)
    sort_skew = skewness.sort_values(ascending=True)

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(18, 14),
        facecolor=bg,
    )
    fig.suptitle("EDA Step 1 — Data quality audit",
                 color=fg, fontsize=PLOT_STYLE["title_size"] + 1, fontweight="bold", y=1.01)

    # ---- Left: missing % and zero-rate ----
    y_pos = np.arange(len(sort_zero))
    bar_h = 0.35
    ax_left.barh(y_pos - bar_h/2, missing_pct[sort_zero.index],
                 height=bar_h, color="#e74c3c", label="Missing %", alpha=0.85)
    ax_left.barh(y_pos + bar_h/2, sort_zero.values,
                 height=bar_h, color="#3498db", label="Zero rate %", alpha=0.85)
    ax_left.set_yticks(y_pos)
    ax_left.set_yticklabels(sort_zero.index, fontsize=6)
    ax_left.set_xlabel("Percentage (%)", color=fg)
    ax_left.set_title("Missing & zero-rate per feature", color=fg,
                      fontsize=PLOT_STYLE["font_size"])
    legend = ax_left.legend(facecolor=PLOT_STYLE["grid_color"],
                             labelcolor=fg, fontsize=9)

    # ---- Right: skewness ----
    colors_skew = []
    for s in sort_skew.values:
        if s <= 3:
            colors_skew.append("#2ecc71")   # green
        elif s <= 10:
            colors_skew.append("#f39c12")   # orange
        else:
            colors_skew.append("#e74c3c")   # red

    ax_right.barh(np.arange(len(sort_skew)), sort_skew.values,
                  color=colors_skew, alpha=0.85)
    ax_right.set_yticks(np.arange(len(sort_skew)))
    ax_right.set_yticklabels(sort_skew.index, fontsize=6)
    ax_right.axvline(3,  color="#f39c12", linestyle="--", linewidth=1, alpha=0.7)
    ax_right.axvline(10, color="#e74c3c", linestyle="--", linewidth=1, alpha=0.7)
    ax_right.set_xlabel("Skewness", color=fg)
    ax_right.set_title("Feature skewness  (|skew|>3 = highly skewed)",
                       color=fg, fontsize=PLOT_STYLE["font_size"])

    # Legend patches
    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor="#2ecc71", label="|skew| ≤ 3"),
        Patch(facecolor="#f39c12", label="|skew| 3–10"),
        Patch(facecolor="#e74c3c", label="|skew| > 10"),
    ]
    ax_right.legend(handles=legend_els, facecolor=PLOT_STYLE["grid_color"],
                    labelcolor=fg, fontsize=9)

    _apply_dark_style(fig, [ax_left, ax_right])
    plt.tight_layout()
    _save(fig, "eda_01_data_quality.png")


# ---------------------------------------------------------------------------
# EDA 2 — Class distribution
# ---------------------------------------------------------------------------

def eda_02_class_distribution(train: pd.DataFrame) -> None:
    """
    Two-panel figure: class counts on linear scale (left) and log scale (right).
    Annotates imbalance ratio.
    """
    print("  EDA 2 — Class distribution")
    fg = PLOT_STYLE["fg_color"]
    bg = PLOT_STYLE["bg_color"]

    counts = train[LABEL_COL].value_counts().sort_values(ascending=True)
    # Clean label display (replace encoding artefact)
    labels = [l.replace("Web Attack \ufffd", "Web Attack ·") for l in counts.index]
    n = len(counts)
    colours = plt.cm.tab20(np.linspace(0, 1, n))

    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(18, 7), facecolor=bg)
    fig.suptitle("EDA Step 2 — Class distribution",
                 color=fg, fontsize=PLOT_STYLE["title_size"] + 1,
                 fontweight="bold", y=1.01)

    y_pos = np.arange(n)

    for ax, scale, title in [
        (ax_lin, "linear", "Linear scale"),
        (ax_log, "log",    "Log scale — reveals minority classes"),
    ]:
        bars = ax.barh(y_pos, counts.values, color=colours, alpha=0.9)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xscale(scale)
        ax.set_xlabel("Number of records", color=fg)
        ax.set_title(title, color=fg, fontsize=PLOT_STYLE["font_size"])

        # Annotate bars with count and %
        total = counts.sum()
        for bar, val in zip(bars, counts.values):
            pct = 100 * val / total
            label_str = f"{val:,}  ({pct:.2f}%)" if pct >= 0.01 else f"{val:,}"
            ax.text(
                bar.get_width() * (1.02 if scale == "linear" else 1.3),
                bar.get_y() + bar.get_height() / 2,
                label_str, va="center", color=fg, fontsize=7,
            )

    # Imbalance ratio annotation on log plot
    imbalance = counts.iloc[-1] / counts.iloc[0]
    ax_log.text(
        0.97, 0.03,
        f"Imbalance ratio\n{counts.index[-1].split()[0]} : "
        f"{counts.index[0].replace('Web Attack ', 'Web·')}\n"
        f"= {imbalance:,.0f} : 1",
        transform=ax_log.transAxes,
        ha="right", va="bottom", color="#f39c12",
        fontsize=9, bbox=dict(boxstyle="round,pad=0.4",
                              facecolor=PLOT_STYLE["grid_color"], alpha=0.8),
    )

    _apply_dark_style(fig, [ax_lin, ax_log])
    plt.tight_layout()
    _save(fig, "eda_02_class_distribution.png")


# ---------------------------------------------------------------------------
# EDA 3 — Feature distributions by class (KDE)
# ---------------------------------------------------------------------------

def eda_03_feature_distributions(train: pd.DataFrame) -> None:
    """
    2×4 grid of KDE plots for 8 representative features.
    One curve per attack class, x-axis = log1p(|value|).
    """
    print("  EDA 3 — Feature distributions by class (KDE)")
    fg = PLOT_STYLE["fg_color"]
    bg = PLOT_STYLE["bg_color"]

    # 8 representative features spanning all category types
    selected = [
        "Flow Duration",
        "Flow Bytes/s",
        "Total Fwd Packets",
        "Total Backward Packets",
        "Packet Length Mean",
        "SYN Flag Count",
        "FIN Flag Count",
        "RST Flag Count",
    ]

    classes  = sorted(train[LABEL_COL].unique())
    palette  = dict(zip(classes, plt.cm.tab20(np.linspace(0, 1, len(classes)))))
    # Clean labels for display
    clean_label = lambda l: l.replace("Web Attack \ufffd", "Web Attack ·")

    fig, axes = plt.subplots(2, 4, figsize=(22, 9), facecolor=bg)
    fig.suptitle("EDA Step 3 — Feature distributions by attack class (KDE)",
                 color=fg, fontsize=PLOT_STYLE["title_size"] + 1,
                 fontweight="bold", y=1.02)

    for ax, feat in zip(axes.flat, selected):
        for cls in classes:
            vals = train.loc[train[LABEL_COL] == cls, feat].dropna()
            if len(vals) < 10:
                continue
            # log1p transform
            transformed = np.log1p(np.abs(vals.values))
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(transformed, bw_method=0.3)
                x = np.linspace(transformed.min(), transformed.max(), 300)
                ax.plot(x, kde(x), color=palette[cls],
                        alpha=0.75, linewidth=1.1,
                        label=clean_label(cls))
            except Exception:
                pass

        ax.set_title(feat, color=fg, fontsize=9, pad=4)
        ax.set_xlabel("log(1 + |value|)", color=fg, fontsize=8)
        ax.set_ylabel("Density", color=fg, fontsize=8)

    # Single shared legend on first subplot
    handles, lbls = axes.flat[0].get_legend_handles_labels()
    if handles:
        axes.flat[0].legend(
            handles, lbls, fontsize=6,
            facecolor=PLOT_STYLE["grid_color"],
            labelcolor=fg,
            loc="upper right",
            ncol=1,
        )

    _apply_dark_style(fig, list(axes.flat))
    plt.tight_layout()
    _save(fig, "eda_03_feature_distributions.png")


# ---------------------------------------------------------------------------
# EDA 4 — Correlation matrix
# ---------------------------------------------------------------------------

def eda_04_correlation_matrix(train: pd.DataFrame) -> None:
    """
    78×78 Pearson r heatmap (upper triangle only).
    Annotates count of highly correlated pairs (|r| > 0.85).
    """
    print("  EDA 4 — Correlation matrix")
    fg = PLOT_STYLE["fg_color"]
    bg = PLOT_STYLE["bg_color"]

    feat_cols = _feature_cols(train)
    corr = train[feat_cols].corr(method="pearson")

    # Mask lower triangle
    mask = np.tril(np.ones_like(corr, dtype=bool))

    n_high = int(((corr.abs() > 0.85) & ~mask).sum().sum() // 2)

    fig, ax = plt.subplots(figsize=(20, 18), facecolor=bg)
    fig.suptitle(
        f"EDA Step 4 — Feature correlation matrix\n"
        f"Highly correlated pairs (|r|>0.85): {n_high} — "
        "consider dropping redundant features before modelling",
        color="#f39c12", fontsize=PLOT_STYLE["font_size"],
        fontweight="bold", y=1.005,
    )

    sns.heatmap(
        corr,
        mask=mask,
        cmap="RdBu_r",
        vmin=-1, vmax=1,
        center=0,
        ax=ax,
        xticklabels=feat_cols,
        yticklabels=feat_cols,
        linewidths=0,
        cbar_kws={"shrink": 0.6, "label": "Pearson r"},
        annot=False,
    )

    ax.set_facecolor(bg)
    ax.tick_params(labelsize=5.5, colors=fg)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right", color=fg)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, color=fg)
    ax.title.set_color(fg)

    # Style colourbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors=fg, labelsize=8)
    cbar.set_label("Pearson r", color=fg, fontsize=9)
    cbar.ax.yaxis.set_tick_params(color=fg)

    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    for spine in ax.spines.values():
        spine.set_edgecolor(PLOT_STYLE["grid_color"])

    plt.tight_layout()
    _save(fig, "eda_04_correlation_matrix.png")


# ---------------------------------------------------------------------------
# EDA 5 — TCP flag profiles
# ---------------------------------------------------------------------------

def eda_05_flag_analysis(train: pd.DataFrame) -> None:
    """
    Left:  heatmap of mean flag rate per attack class × flag column.
    Right: grouped bar chart of the same data.
    """
    print("  EDA 5 — TCP flag profiles")
    fg = PLOT_STYLE["fg_color"]
    bg = PLOT_STYLE["bg_color"]

    flag_cols = ["SYN Flag Count", "FIN Flag Count", "RST Flag Count",
                 "PSH Flag Count", "ACK Flag Count"]
    # Use binarised flag rate (flag > 0) per class
    df_flag = train[[LABEL_COL] + flag_cols].copy()
    for col in flag_cols:
        df_flag[col] = (df_flag[col] > 0).astype(float)

    # Clean class labels
    df_flag[LABEL_COL] = df_flag[LABEL_COL].str.replace(
        "Web Attack \ufffd", "Web Attack ·", regex=False
    )

    mean_rates = df_flag.groupby(LABEL_COL)[flag_cols].mean().round(2)
    classes_sorted = mean_rates.index.tolist()
    short_flags = ["SYN", "FIN", "RST", "PSH", "ACK"]

    fig, (ax_heat, ax_bar) = plt.subplots(1, 2, figsize=(20, 8), facecolor=bg)
    fig.suptitle("EDA Step 5 — TCP flag profile per attack class",
                 color=fg, fontsize=PLOT_STYLE["title_size"] + 1,
                 fontweight="bold", y=1.01)

    # ---- Heatmap ----
    sns.heatmap(
        mean_rates,
        ax=ax_heat,
        cmap="YlGnBu",
        vmin=0, vmax=1,
        annot=True, fmt=".2f",
        annot_kws={"size": 9, "color": "black"},
        xticklabels=short_flags,
        linewidths=0.5,
        cbar_kws={"shrink": 0.6},
    )
    ax_heat.set_title("Mean flag rate (0=never, 1=always)",
                      color=fg, fontsize=PLOT_STYLE["font_size"])
    ax_heat.set_xlabel("", color=fg)
    ax_heat.set_ylabel("", color=fg)
    ax_heat.tick_params(colors=fg, labelsize=9)
    ax_heat.set_xticklabels(short_flags, color=fg, rotation=0)
    ax_heat.set_yticklabels(ax_heat.get_yticklabels(), color=fg, rotation=0)
    ax_heat.set_facecolor(bg)
    cbar = ax_heat.collections[0].colorbar
    cbar.ax.tick_params(colors=fg)

    # ---- Grouped bar chart ----
    n_classes = len(classes_sorted)
    n_flags   = len(short_flags)
    x         = np.arange(n_classes)
    bar_w     = 0.15
    colours   = ["#e74c3c", "#2ecc71", "#f39c12", "#3498db", "#9b59b6"]

    for i, (flag, col, colour) in enumerate(zip(short_flags, flag_cols, colours)):
        vals = mean_rates[col].values
        ax_bar.bar(x + i * bar_w - (n_flags - 1) * bar_w / 2,
                   vals, width=bar_w, label=flag, color=colour, alpha=0.85)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(classes_sorted, rotation=45, ha="right",
                            color=fg, fontsize=8)
    ax_bar.set_ylabel("Mean flag rate", color=fg)
    ax_bar.set_title("Flag rates per class — grouped",
                     color=fg, fontsize=PLOT_STYLE["font_size"])
    ax_bar.set_ylim(0, 1.1)
    ax_bar.legend(facecolor=PLOT_STYLE["grid_color"], labelcolor=fg, fontsize=9)

    _apply_dark_style(fig, [ax_heat, ax_bar])
    plt.tight_layout()
    _save(fig, "eda_05_flag_analysis.png")


# ---------------------------------------------------------------------------
# EDA 6 — Feature importance (RF Gini) + save gini_importance.csv
# ---------------------------------------------------------------------------

def eda_06_feature_importance(train: pd.DataFrame) -> pd.DataFrame:
    """
    Fits RandomForestClassifier on all 78 features of the training set.
    Plots top-20 features by Gini importance with a cumulative curve.
    Saves the full 78-feature ranking to results/gini_importance.csv.
    Returns the gini_df DataFrame for use by EDA 7.
    """
    print("  EDA 6 — RF Gini feature importance (fitting RF — this may take a few minutes)")
    fg = PLOT_STYLE["fg_color"]
    bg = PLOT_STYLE["bg_color"]

    feat_cols = _feature_cols(train)
    X = train[feat_cols].values
    y = train[LABEL_COL].values

    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        max_depth=None,
        min_samples_leaf=5,
    )
    print("    Fitting RandomForest...")
    rf.fit(X, y)
    print("    Done.")

    importances = rf.feature_importances_
    gini_df = pd.DataFrame({
        "feature":              feat_cols,
        "importance":           importances,
    })
    gini_df = gini_df.sort_values("importance", ascending=False).reset_index(drop=True)
    gini_df["rank"]                = gini_df.index + 1
    gini_df["cumulative_importance"] = gini_df["importance"].cumsum()
    gini_df = gini_df[["rank", "feature", "importance", "cumulative_importance"]]

    # Save CSV
    out_csv = RESULTS_DIR / "gini_importance.csv"
    gini_df.to_csv(out_csv, index=False)
    print(f"    Saved → {out_csv.name}")
    print(f"    Top-5 features: {gini_df['feature'].head(5).tolist()}")

    # --- Figure ---
    top20 = gini_df.head(20)
    # Identify top-8 (≥90% cumulative)
    cum90_rank = int(gini_df[gini_df["cumulative_importance"] <= 0.90].shape[0]) + 1
    cum90_rank = min(cum90_rank, 78)

    fig, ax_main = plt.subplots(figsize=(14, 9), facecolor=bg)
    fig.suptitle("EDA Step 6 — Top-20 feature importances (RF Gini)",
                 color=fg, fontsize=PLOT_STYLE["title_size"] + 1,
                 fontweight="bold", y=1.01)

    colours = ["#4a90d9" if row["rank"] <= 8 else "#7f8c8d"
               for _, row in top20.iterrows()]
    y_pos = np.arange(len(top20) - 1, -1, -1)   # top feature at top

    ax_main.barh(y_pos, top20["importance"].values, color=colours, alpha=0.9)
    ax_main.set_yticks(y_pos)
    ax_main.set_yticklabels(top20["feature"].values, fontsize=9, color=fg)
    ax_main.set_xlabel("Mean decrease in impurity (Gini)", color=fg)

    # Secondary x-axis: cumulative importance %
    ax_top = ax_main.twiny()
    cum_vals_top20 = top20["cumulative_importance"].values * 100
    ax_top.plot(cum_vals_top20, y_pos, "o-",
                color="#f39c12", linewidth=1.5, markersize=4, alpha=0.9,
                label="Cumulative importance")
    ax_top.set_xlim(0, 105)
    ax_top.set_xlabel("Cumulative importance (%)", color="#f39c12")
    ax_top.tick_params(axis="x", colors="#f39c12", labelsize=9)
    ax_top.set_facecolor(bg)
    ax_top.spines["top"].set_edgecolor(PLOT_STYLE["grid_color"])

    # Annotate 90% line
    if cum90_rank <= 20:
        y_90 = len(top20) - cum90_rank  # flipped axis
        ax_top.axhline(y_90, color="#f39c12", linestyle="--",
                       linewidth=1, alpha=0.6)
        ax_top.text(
            102, y_90,
            f"90% captured\nby top-{cum90_rank} features",
            color="#f39c12", va="center", ha="right", fontsize=8,
        )

    # Vertical dashed line at 90% on top axis
    ax_top.axvline(90, color="#f39c12", linestyle=":", linewidth=0.8, alpha=0.5)

    _apply_dark_style(fig, [ax_main])
    ax_top.set_facecolor(bg)
    fig.patch.set_facecolor(bg)
    plt.tight_layout()
    _save(fig, "eda_06_feature_importance.png")

    return gini_df


# ---------------------------------------------------------------------------
# EDA 7 — Anonymisation readiness summary
# ---------------------------------------------------------------------------

def eda_07_readiness(train: pd.DataFrame, gini_df: pd.DataFrame) -> None:
    """
    Four-panel figure:
    Top-left:     distortion sensitivity heatmap (categories × techniques)
    Top-right:    feature count per category
    Bottom-left:  top-15 features by variance (blue = also top-10 Gini)
    Bottom-right: research experiment scope table
    """
    print("  EDA 7 — Anonymisation readiness summary")
    fg   = PLOT_STYLE["fg_color"]
    bg   = PLOT_STYLE["bg_color"]
    grid = PLOT_STYLE["grid_color"]

    feat_cols = _feature_cols(train)

    # ---- Sensitivity matrix (qualitative, hand-coded) ----
    categories   = ["Packet lengths", "Flow timing", "Packet counts",
                    "Byte volumes", "TCP flags", "Window/Bulk"]
    techniques   = ["Masking", "Noise", "Generalisation", "Suppression",
                    "k-Anonymity", "Diff. Privacy", "Synthetic"]
    sensitivity  = np.array([
        [0.30, 0.85, 0.70, 1.00, 0.60, 0.80, 0.55],
        [0.20, 0.80, 0.65, 1.00, 0.55, 0.90, 0.60],
        [0.40, 0.65, 0.55, 1.00, 0.45, 0.70, 0.50],
        [0.30, 0.75, 0.60, 1.00, 0.50, 0.80, 0.60],
        [0.60, 0.40, 0.30, 1.00, 0.30, 0.50, 0.70],
        [0.20, 0.60, 0.45, 1.00, 0.50, 0.70, 0.80],
    ])

    # ---- Feature count per category ----
    category_map = {
        "Packet lengths": [c for c in feat_cols if any(x in c for x in
            ["Packet Length", "Segment Size", "Min Packet", "Max Packet",
             "Average Packet", "Length of"])],
        "Flow timing":    [c for c in feat_cols if any(x in c for x in
            ["IAT", "Duration", "Active", "Idle"])],
        "Packet counts":  [c for c in feat_cols if any(x in c for x in
            ["Fwd Packets", "Bwd Packets", "Total Fwd", "Total Backward",
             "Subflow Fwd Packets", "Subflow Bwd Packets"])],
        "Byte volumes":   [c for c in feat_cols if any(x in c for x in
            ["Bytes/s", "Subflow Fwd Bytes", "Subflow Bwd Bytes",
             "Avg Bulk Rate", "Avg Bytes/Bulk"])],
        "TCP flags":      [c for c in feat_cols if any(x in c for x in
            ["Flag", "PSH Flags", "URG Flags"])],
        "Window/Bulk":    [c for c in feat_cols if any(x in c for x in
            ["Init_Win", "act_data", "min_seg", "Packets/s", "Down/Up",
             "Avg Packets/Bulk", "Bulk Rate", "Header"])],
    }
    cat_counts = {k: len(v) for k, v in category_map.items()}

    # ---- Top-15 by variance ----
    variances = train[feat_cols].var()
    top15_var = variances.nlargest(15).sort_values(ascending=True)
    top10_gini = set(gini_df.head(10)["feature"].tolist())

    # ---- Research scope values ----
    n_classes   = train[LABEL_COL].nunique()
    n_train     = len(train)

    fig = plt.figure(figsize=(22, 16), facecolor=bg)
    gs  = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.30)
    ax_heat  = fig.add_subplot(gs[0, 0])
    ax_cnt   = fig.add_subplot(gs[0, 1])
    ax_var   = fig.add_subplot(gs[1, 0])
    ax_scope = fig.add_subplot(gs[1, 1])

    fig.suptitle("EDA Step 7 — Anonymisation readiness & research scope",
                 color=fg, fontsize=PLOT_STYLE["title_size"] + 1,
                 fontweight="bold", y=1.01)

    # Top-left: sensitivity heatmap
    im = ax_heat.imshow(sensitivity, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
    ax_heat.set_xticks(range(len(techniques)))
    ax_heat.set_xticklabels(techniques, rotation=30, ha="right",
                             color=fg, fontsize=9)
    ax_heat.set_yticks(range(len(categories)))
    ax_heat.set_yticklabels(categories, color=fg, fontsize=9)
    ax_heat.set_title("Distortion sensitivity per category × technique",
                      color=fg, fontsize=PLOT_STYLE["font_size"])
    for i in range(len(categories)):
        for j in range(len(techniques)):
            ax_heat.text(j, i, f"{sensitivity[i, j]:.2f}",
                         ha="center", va="center", fontsize=8,
                         color="black" if sensitivity[i, j] > 0.5 else fg)
    cbar = plt.colorbar(im, ax=ax_heat, shrink=0.8)
    cbar.ax.tick_params(colors=fg, labelsize=8)

    # Top-right: feature counts per category
    cat_names = list(cat_counts.keys())
    cat_vals  = [cat_counts[k] for k in cat_names]
    cat_colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]
    ax_cnt.barh(cat_names, cat_vals, color=cat_colors, alpha=0.85)
    ax_cnt.set_xlabel("Number of features", color=fg)
    ax_cnt.set_title("Feature count per category",
                     color=fg, fontsize=PLOT_STYLE["font_size"])
    for i, v in enumerate(cat_vals):
        ax_cnt.text(v + 0.1, i, str(v), va="center", color=fg, fontsize=10)
    ax_cnt.set_yticklabels(cat_names, color=fg, fontsize=9)

    # Bottom-left: top-15 by variance
    colours_var = ["#4a90d9" if f in top10_gini else "#7f8c8d"
                   for f in top15_var.index]
    ax_var.barh(range(15), np.log1p(top15_var.values),
                color=colours_var, alpha=0.85)
    ax_var.set_yticks(range(15))
    ax_var.set_yticklabels(top15_var.index, fontsize=8, color=fg)
    ax_var.set_xlabel("log(1 + variance)", color=fg)
    ax_var.set_title("Top-15 features by variance\n"
                     "(blue = also top-10 importance)",
                     color=fg, fontsize=PLOT_STYLE["font_size"])
    from matplotlib.patches import Patch
    ax_var.legend(
        handles=[Patch(color="#4a90d9", label="High importance + variance"),
                 Patch(color="#7f8c8d", label="High variance only")],
        facecolor=grid, labelcolor=fg, fontsize=8, loc="lower right",
    )

    # Bottom-right: scope table
    ax_scope.axis("off")
    ax_scope.set_facecolor(bg)
    n_cells = (len([0.1, 0.5]) + len([0.25, 0.50, 0.75]) + len(
        [0.01, 0.1, 0.5, 1.0, 5.0, 10.0])) * 78

    scope_data = [
        ["Dataset",                "CICIDS-2017"],
        ["Total flows",            f"{n_train + len(train):,}"],
        ["Numeric features",       "78"],
        ["Attack classes",         str(n_classes)],
        ["Anon. techniques",       "3 (noise, suppression, DP)"],
        ["Intensity levels",       "2 + 3 + 6 = 11"],
        ["Divergence metrics",     "JSD · Wasserstein · KL"],
        ["Total metric computations", f"11 × 78 × 3 = {11*78*3:,}"],
        ["Experiment cells",       f"858  (11 × 78)"],
        ["Classifiers",            "None (out of scope)"],
        ["Privacy metric",         "ε  (differential privacy)"],
        ["Primary output",         "JSD / Wasserstein / KL\nper feature per technique"],
    ]

    col_labels = ["Parameter", "Value"]
    tbl = ax_scope.table(
        cellText=scope_data,
        colLabels=col_labels,
        cellLoc="left", loc="center",
        colWidths=[0.45, 0.55],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor(bg if row % 2 == 0 else grid)
        cell.set_text_props(color=fg)
        cell.set_edgecolor(grid)
        if row == 0:
            cell.set_facecolor("#2c3e6a")
            cell.set_text_props(color="#f39c12", fontweight="bold")

    ax_scope.set_title("Research experiment scope",
                       color=fg, fontsize=PLOT_STYLE["font_size"], pad=12)

    _apply_dark_style(fig, [ax_heat, ax_cnt, ax_var, ax_scope])
    plt.tight_layout()
    _save(fig, "eda_07_readiness.png")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run all 7 EDA steps in sequence. Loads train.parquet once."""
    print("\n[eda] Starting...")

    print("  Loading train.parquet...")
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    print(f"  Loaded: {train.shape[0]:,} rows × {train.shape[1]} columns")

    eda_01_data_quality(train)
    eda_02_class_distribution(train)
    eda_03_feature_distributions(train)
    eda_04_correlation_matrix(train)
    eda_05_flag_analysis(train)
    gini_df = eda_06_feature_importance(train)
    eda_07_readiness(train, gini_df)

    print("\n[eda] Done ✓  —  7 figures saved to figures/eda/")
    print(f"             gini_importance.csv saved to results/")


if __name__ == "__main__":
    main()
