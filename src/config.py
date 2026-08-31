"""
config.py — Central configuration for the COS700 anonymisation utility-loss experiment.

All paths are resolved relative to the project root (the directory containing this
package) at import time, making the project portable across machines.
All experiment hyperparameters are defined here so no magic numbers appear elsewhere.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project root — two levels up from this file (src/config.py → src/ → Research/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Directory paths
# ---------------------------------------------------------------------------
DATA_DIR        = PROJECT_ROOT / "data"
RAW_DATA_DIR    = DATA_DIR / "raw"
PROCESSED_DIR   = DATA_DIR / "processed"
ANONYMISED_DIR  = DATA_DIR / "anonymised"

RESULTS_DIR     = PROJECT_ROOT / "results"
FIGURES_DIR     = PROJECT_ROOT / "figures"
FIGURES_EDA_DIR = FIGURES_DIR / "eda"
FIGURES_EXP_DIR = FIGURES_DIR / "experiment"

# Source of raw CICIDS-2017 CSVs (outside project tree)
CICIDS_SOURCE_DIR = Path("/Users/siphesihlekhumalo/Downloads/archive")

# ---------------------------------------------------------------------------
# Dataset constants
# ---------------------------------------------------------------------------
LABEL_COL  = "Label"   # column name after stripping whitespace
N_FEATURES = 78        # expected number of numeric features

# ---------------------------------------------------------------------------
# Split parameters
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE    = 0.15    # 15% held-out test set
VAL_SIZE     = 0.15    # 15% validation set (of total; taken from the non-test 85%)
# Effective val split from the 30% pool: 0.15 / 0.30 = 0.5

# ---------------------------------------------------------------------------
# Anonymisation hyperparameters
# ---------------------------------------------------------------------------
NOISE_SIGMAS       = [0.1, 0.5]
SUPPRESSION_LEVELS = [0.25, 0.50, 0.75]
DP_EPSILONS        = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]

# ---------------------------------------------------------------------------
# Divergence metric parameters
# ---------------------------------------------------------------------------
N_BINS    = 100    # histogram bins for JSD and KL computation
KL_SMOOTH = 1e-10  # additive smoothing to prevent log(0)

# ---------------------------------------------------------------------------
# Plot style — dark theme consistent across all figures
# ---------------------------------------------------------------------------
PLOT_STYLE = {
    "bg_color":   "#1a1a2e",   # figure and axes background
    "fg_color":   "#e0e0e0",   # text, tick labels, axis labels
    "grid_color": "#2a2a4a",   # grid lines
    "dpi":        300,
    "font_size":  11,
    "title_size": 13,
}

# Feature categories for the DP ε-curve and EDA readiness plot
FEATURE_CATEGORIES = {
    "Packet lengths": [
        "Fwd Packet Length Max", "Fwd Packet Length Min",
        "Fwd Packet Length Mean", "Fwd Packet Length Std",
        "Bwd Packet Length Max", "Bwd Packet Length Min",
        "Bwd Packet Length Mean", "Bwd Packet Length Std",
        "Min Packet Length", "Max Packet Length",
        "Packet Length Mean", "Packet Length Std",
        "Packet Length Variance", "Average Packet Size",
        "Avg Fwd Segment Size", "Avg Bwd Segment Size",
        "min_seg_size_forward",
    ],
    "Flow timing": [
        "Flow Duration",
        "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std",
        "Fwd IAT Max", "Fwd IAT Min",
        "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std",
        "Bwd IAT Max", "Bwd IAT Min",
        "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
        "Active Mean", "Active Std", "Active Max", "Active Min",
        "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
    ],
    "Packet counts": [
        "Total Fwd Packets", "Total Backward Packets",
        "Total Length of Fwd Packets", "Total Length of Bwd Packets",
        "Subflow Fwd Packets", "Subflow Bwd Packets",
        "Subflow Fwd Bytes", "Subflow Bwd Bytes",
    ],
    "Byte volumes": [
        "Flow Bytes/s",
    ],
    "TCP flags": [
        "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
        "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
        "CWE Flag Count", "ECE Flag Count",
        "Fwd PSH Flags", "Bwd PSH Flags",
        "Fwd URG Flags", "Bwd URG Flags",
    ],
    "Window/Bulk": [
        "Init_Win_bytes_forward", "Init_Win_bytes_backward",
        "act_data_pkt_fwd",
        "Flow Packets/s", "Down/Up Ratio",
    ],
}

# ---------------------------------------------------------------------------
# Convenience: ensure all output directories exist when config is imported
# ---------------------------------------------------------------------------
for _dir in [
    PROCESSED_DIR, ANONYMISED_DIR, RESULTS_DIR,
    FIGURES_EDA_DIR, FIGURES_EXP_DIR,
]:
    _dir.mkdir(parents=True, exist_ok=True)
