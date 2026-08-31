# COS700 — Experiment Implementation Plan

## Project Overview

**Title:** Impact of Data Anonymisation on the Statistical Utility of Cybersecurity Datasets
**Student:** Siphesihle Khumalo (u25759257)
**Supervisor:** Prof. Eloff
**Module:** COS700, University of Pretoria

**Goal:** Produce a systematic, feature-level empirical analysis of how three anonymisation
techniques (noise perturbation, suppression, differential privacy) distort the statistical
utility of CICIDS-2017 across all 78 numerical features. For each technique and intensity
level, compute JSD, Wasserstein Distance, and KL Divergence per feature. Compute Spearman ρ
between distortion scores and RF Gini importance to determine whether techniques
preferentially damage IDS-critical features.

---

## Constraints (non-negotiable)

- All 78 numerical features from CICIDS-2017 (not a subset)
- EDA must be built from scratch — do not reuse existing EDA PNGs
- Gini importance: one RF fit in EDA Step 6, saved to `results/gini_importance.csv`,
  reused by anonymisation pipeline (suppression ordering) and metrics (Spearman ρ)
- Anonymisation applied to training set ONLY; val and test sets are never modified
- No classifier training, no adversarial attacks — out of scope
- Stratified 70/15/15 split, `random_state=42`
- All figures: 300 DPI, dark background theme (`#1a1a2e` background)
- Project root: `/Users/siphesihlekhumalo/Desktop/Research`
- Dataset: `/Users/siphesihlekhumalo/Downloads/archive/` (8 `.pcap_ISCX.csv` files)

---

## Execution Protocol

Each task must be completed and APPROVED before the next begins.
After completing each task, the agent must:
1. Run all tests for that task and show the output
2. Print a summary of what was built
3. Wait for explicit user approval before proceeding to the next task

---

## Project Structure

```
/Users/siphesihlekhumalo/Desktop/Research/
├── data/
│   ├── raw/                        # symlink or reference — source CSVs live in ~/Downloads/archive/
│   ├── processed/                  # train.parquet, val.parquet, test.parquet
│   └── anonymised/                 # 11 anonymised training parquets
├── src/
│   ├── __init__.py
│   ├── config.py                   # all paths, constants, hyperparameters
│   ├── preprocessing.py            # data loading, cleaning, splitting
│   ├── eda.py                      # all 7 EDA steps
│   ├── anonymise.py                # noise, suppression, DP implementations
│   ├── metrics.py                  # JSD, Wasserstein, KL, Spearman ρ
│   └── visualise.py                # all result figures
├── results/
│   ├── gini_importance.csv         # produced by EDA step 6, reused downstream
│   ├── metrics.csv                 # 858 rows: 11 variants × 78 features × 3 metrics
│   └── spearman.csv                # 33 rows: 11 variants × 3 metrics
├── figures/
│   ├── eda/                        # eda_01_data_quality.png … eda_07_readiness.png
│   └── experiment/                 # fig_01_jsd_noise.png … fig_12_summary.png
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_anonymise.py
│   └── test_metrics.py
├── run_pipeline.py                 # top-level runner, calls each module's main() in sequence
├── requirements.txt
└── PLAN.md                         # this file
```

---

## Constants (`src/config.py`)

```python
RANDOM_STATE       = 42
TEST_SIZE          = 0.15
VAL_SIZE           = 0.15   # of the remaining 85% after test split

NOISE_SIGMAS       = [0.1, 0.5]
SUPPRESSION_LEVELS = [0.25, 0.50, 0.75]
DP_EPSILONS        = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]

N_BINS             = 100
KL_SMOOTH          = 1e-10

N_FEATURES         = 78    # assert this after loading

LABEL_COL          = 'Label'   # after stripping whitespace from column names

PLOT_STYLE = {
    "bg_color":   "#1a1a2e",
    "fg_color":   "#e0e0e0",
    "grid_color": "#2a2a4a",
    "dpi":        300,
    "font_size":  11,
    "title_size": 13,
}
```

All path constants defined using `pathlib.Path` relative to the project root,
resolved at import time so the project is portable.

---

## Task 1 — Project Scaffold and Configuration

### Objective
Create the full directory structure, `requirements.txt`, `README.md`, `config.py`,
and `run_pipeline.py` shell. Every subsequent task drops cleanly into this structure.

### Deliverables
- All directories created
- `src/config.py` — all constants and paths as specified above
- `requirements.txt` — pinned versions:
  ```
  pandas>=2.0
  numpy>=1.26
  scipy>=1.11
  scikit-learn>=1.4
  matplotlib>=3.8
  seaborn>=0.13
  pyarrow>=14.0
  tqdm>=4.66
  pytest>=7.4
  ```
- `README.md` — project description, setup instructions, how to run each step,
  how to run tests
- `run_pipeline.py` — imports each module, calls `main()` in sequence with logging.
  Each call wrapped in a step banner so it is clear which task is running
- All `src/` module files as stubs: each has a docstring, all intended functions
  declared with `pass` bodies, and a `main()` that prints `"[module] not yet implemented"`

### Tests (`tests/test_scaffold.py`)
```python
def test_directories_exist():
    # assert all required directories exist

def test_config_imports():
    # assert all constants are defined and correct types

def test_stubs_importable():
    # assert all src modules import without error
```

### Acceptance criteria
- `python run_pipeline.py` runs without any import or runtime errors
- All stubs print their "not yet implemented" messages in order
- `pytest tests/` passes all scaffold tests

---

## Task 2 — Data Loading and Preprocessing (`src/preprocessing.py`)

### Objective
Load all 8 CICIDS-2017 CSVs, clean them per the proposal spec, and produce
reproducible stratified train/val/test parquet splits.

### Functions to implement

```python
def load_raw(raw_dir: Path) -> pd.DataFrame:
    """
    Glob all *.pcap_ISCX.csv files from raw_dir.
    Read each with pd.read_csv, strip whitespace from ALL column names.
    Concatenate into a single DataFrame. Print file names and row counts.
    """

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    1. Replace np.inf and -np.inf with NaN.
    2. Drop all rows containing any NaN.
    3. Drop exact duplicate rows.
    Print rows dropped at each stage and total surviving rows.
    """

def identify_numeric_features(df: pd.DataFrame, label_col: str) -> list[str]:
    """
    Select columns with dtype float64 or int64, excluding label_col.
    Assert len == N_FEATURES (78).
    Return sorted list of feature names.
    """

def split(df, label_col) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified 70/15/15 split.
    Step 1: split off 30% for val+test (stratify=label).
    Step 2: split the 30% into two equal halves (stratify=label).
    Use random_state=RANDOM_STATE throughout.
    Return (train, val, test).
    """

def save_splits(train, val, test, processed_dir: Path) -> None:
    """Save as train.parquet, val.parquet, test.parquet using pyarrow engine."""

def main() -> None:
    """
    Orchestrate: load → clean → identify features → split → save.
    Print full preprocessing report.
    """
```

### Tests (`tests/test_preprocessing.py`)
```python
def test_no_nan_inf_in_splits(): ...
def test_split_sizes_sum_to_total(): ...
def test_class_proportions_consistent(): ...
def test_78_numeric_features(): ...
def test_label_column_present(): ...
def test_parquets_readable(): ...
```

### Acceptance criteria
- All 6 tests pass
- Preprocessing report shows CICIDS-2017 characteristics

---

## Task 3 — Exploratory Data Analysis (`src/eda.py`)

### Objective
Characterise all 78 features and the label distribution. Produce 7 figures and
save `results/gini_importance.csv`. All analysis on training set only.

### Figures

| # | Filename | Description |
|---|----------|-------------|
| 1 | `eda_01_data_quality.png` | Missing/zero-rate per feature + skewness bar chart |
| 2 | `eda_02_class_distribution.png` | Class counts linear + log scale |
| 3 | `eda_03_feature_distributions.png` | 2×4 KDE grid for 8 representative features by class |
| 4 | `eda_04_correlation_matrix.png` | 78×78 Pearson r heatmap (upper triangle) |
| 5 | `eda_05_flag_analysis.png` | TCP flag mean rate heatmap + grouped bar chart |
| 6 | `eda_06_feature_importance.png` | Top-20 RF Gini importance + cumulative curve |
| 7 | `eda_07_readiness.png` | 4-panel: distortion sensitivity, feature counts, variance, scope table |

**EDA 6 also saves:** `results/gini_importance.csv`
Columns: `[rank, feature, importance, cumulative_importance]`, 78 rows, rank 1 = most important.

### Tests
```python
def test_gini_csv_has_78_rows(): ...
def test_gini_importances_sum_to_one(): ...
def test_all_eda_figures_exist(): ...
def test_eda_figures_non_empty(): ...
```

### Acceptance criteria
- All 4 tests pass
- All 7 figures generated and visually coherent
- `gini_importance.csv` saved with 78 rows

---

## Task 4 — Anonymisation Pipeline (`src/anonymise.py`)

### Objective
Apply all three techniques at all intensity levels to the training set only.
Produce 11 anonymised parquet files. Label column unchanged in all outputs.

### Techniques and output files

| Technique | Level | Output filename |
|-----------|-------|----------------|
| Noise | σ=0.1 | `train_noise_0.1.parquet` |
| Noise | σ=0.5 | `train_noise_0.5.parquet` |
| Suppression | 25% | `train_suppression_0.25.parquet` |
| Suppression | 50% | `train_suppression_0.50.parquet` |
| Suppression | 75% | `train_suppression_0.75.parquet` |
| DP | ε=0.01 | `train_dp_0.01.parquet` |
| DP | ε=0.1 | `train_dp_0.1.parquet` |
| DP | ε=0.5 | `train_dp_0.5.parquet` |
| DP | ε=1.0 | `train_dp_1.0.parquet` |
| DP | ε=5.0 | `train_dp_5.0.parquet` |
| DP | ε=10.0 | `train_dp_10.0.parquet` |

### Key implementation details
- **Noise:** `N(0, σ × feature_std)` added per feature. Seed set once before all features.
- **Suppression:** Features sorted by ascending Gini importance. Bottom N replaced with
  training mean. `n_suppress = int(np.floor(78 * level))` → 19, 39, 58 features.
- **DP:** Per feature: clip to `[Q1 − 1.5×IQR, Q3 + 1.5×IQR]`, recompute IQR on clipped
  values, `sensitivity = IQR_clipped / epsilon`, add `Laplace(0, sensitivity)` noise.

### Tests (`tests/test_anonymise.py`)
```python
def test_all_11_files_exist(): ...
def test_same_shape_as_train(): ...
def test_label_column_unchanged(): ...
def test_suppressed_features_are_constant(): ...
def test_dp_noise_scale_inversely_proportional(): ...
def test_noise_mean_near_zero(): ...
def test_suppression_75_greater_distortion_than_25(): ...
```

### Acceptance criteria
- All 7 tests pass
- Summary table printed with distortion ordering intuitive

---

## Task 5 — Divergence Metric Computation (`src/metrics.py`)

### Objective
Compute JSD, Wasserstein, KL for all 78 features × 11 variants.
Compute Spearman ρ per (technique, level, metric). Save both result tables.

### Metric definitions
- **JSD:** `jensenshannon(p, q)² ∈ [0, 1]` — primary metric, symmetric, bounded
- **Wasserstein:** `wasserstein_distance(orig, anon) / orig.std()` — normalised, scale-free
- **KL:** `Σ rel_entr(p + ε, q + ε)` — diagnostic, unbounded, signals catastrophic distortion

### Output files
- `results/metrics.csv` — 858 rows (11 × 78), columns: `[technique, level, feature, jsd, wasserstein_norm, kl]`
- `results/spearman.csv` — 33 rows (11 × 3), columns: `[technique, level, metric, rho, pvalue]`

### Tests (`tests/test_metrics.py`)
```python
def test_metrics_csv_has_858_rows(): ...
def test_spearman_csv_has_33_rows(): ...
def test_jsd_in_unit_interval(): ...
def test_jsd_zero_for_identity(): ...
def test_suppression_monotonic_jsd(): ...
def test_dp_monotonic_jsd(): ...
def test_spearman_suppression_monotonic_rho(): ...
```

### Acceptance criteria
- All 7 tests pass
- Printed summary shows mean JSD ranking and Spearman ρ table

---

## Task 6 — Visualisation (`src/visualise.py`)

### Objective
Produce all 12 publication-ready result figures. Dark theme throughout.
Load from `results/metrics.csv`, `results/spearman.csv`, `results/gini_importance.csv`.

### Figures

| # | Filename | Description |
|---|----------|-------------|
| 1 | `fig_01_jsd_noise.png` | JSD heatmap — noise (78×2) |
| 2 | `fig_02_jsd_suppression.png` | JSD heatmap — suppression (78×3) |
| 3 | `fig_03_jsd_dp.png` | JSD heatmap — DP (78×6) |
| 4 | `fig_04_wass_noise.png` | Wasserstein heatmap — noise |
| 5 | `fig_05_wass_suppression.png` | Wasserstein heatmap — suppression |
| 6 | `fig_06_wass_dp.png` | Wasserstein heatmap — DP |
| 7 | `fig_07_kl_noise.png` | KL heatmap — noise (log colour scale) |
| 8 | `fig_08_kl_suppression.png` | KL heatmap — suppression (log colour scale) |
| 9 | `fig_09_kl_dp.png` | KL heatmap — DP (log colour scale) |
| 10 | `fig_10_dp_epsilon_curve.png` | DP privacy-utility curve, JSD + Wasserstein vs ε |
| 11 | `fig_11_spearman.png` | Grouped bar chart of Spearman ρ with significance stars |
| 12 | `fig_12_summary.png` | Master 3-panel summary: mean JSD, Wasserstein, Spearman ρ |

### Heatmap spec (all 9 heatmaps share this structure)
- Rows: 78 features sorted by descending Gini rank (rank 1 at top)
- Row labels: `"{feature_name} (#{rank})"`
- Red dashed separator after row 8 with annotation "← Top-8 features (90% Gini)"
- Figure size: 10in × 20in
- Colourmaps: JSD=viridis, Wasserstein=plasma, KL=magma (LogNorm)

### Tests (`tests/test_visualise.py`)
```python
def test_all_12_figures_exist(): ...
def test_figures_non_empty(): ...        # > 10KB each
def test_heatmap_column_counts(): ...    # noise=2, sup=3, dp=6
def test_epsilon_curve_exists(): ...
```

### Acceptance criteria
- All 4 tests pass
- All 12 figures generated, non-empty, publication-ready
- Figure 12 standalone tells the full story

---

## Running the Pipeline

```bash
# Full pipeline
cd /Users/siphesihlekhumalo/Desktop/Research
python run_pipeline.py

# Individual steps
python -m src.preprocessing
python -m src.eda
python -m src.anonymise
python -m src.metrics
python -m src.visualise

# Tests
pytest tests/ -v
```

---

## Key Numbers to Verify at Each Stage

| Stage | Expected value |
|-------|----------------|
| Raw rows (all 8 CSVs) | ~2,830,743 |
| After cleaning | ~2,270,000–2,400,000 |
| Numeric features | exactly 78 |
| Attack classes | 14 (including BENIGN) |
| Train rows | ~70% of cleaned total |
| Anonymised files | 11 |
| `metrics.csv` rows | 858 (11 × 78) |
| `spearman.csv` rows | 33 (11 × 3) |
| EDA figures | 7 |
| Result figures | 12 |
