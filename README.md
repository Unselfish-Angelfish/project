# COS700: Impact of Data Anonymisation on the Statistical Utility of Cybersecurity Datasets
 
**Student:** Siphesihle Khumalo (u25759257)
**Supervisor:** Prof. Eloff
**Module:** COS700, University of Pretoria
 
## Overview
 
This repository contains the code and results for a feature-level analysis of how three
anonymisation techniques distort the statistical properties of the CICIDS-2017 dataset:
 
- **Noise perturbation:** additive Gaussian noise scaled by α·σⱼ, α ∈ {0.1, 0.5}
- **Suppression:** mean-imputation of 25 %, 50 % and 75 % of features, least important first
- **Differential privacy:** per-value Laplace mechanism with data-independent clipping bounds,
  ε ∈ {0.01, 0.1, 0.5, 1, 5, 10}, plus a clip-only baseline (ε → ∞)
That gives 12 anonymised variants of the cleaned analysis set (2,520,798 flows, d = 68 features).
For every feature and variant, the notebooks compute Jensen–Shannon divergence (JSD), normalised
Wasserstein distance (W₁) and KL divergence. A Spearman rank correlation between each technique's
distortion scores and Random Forest Gini importance tests whether a technique preferentially
distorts the features that matter most for intrusion detection.
 
## Repository structure
 
```
src/
  NB1_preprocessing_eda_colab.ipynb   # cleaning, feature selection, DP bounds, Gini importance, EDA
  NB2_anonymisation_colab.ipynb       # builds the 12 anonymised variants
  NB3_metrics_colab.ipynb             # JSD, W1 and KL per feature; Spearman rho
  NB4_visualisation_colab.ipynb       # heatmaps, epsilon-curve, Spearman plot, summary tables
results/
  feature_meta.csv            # per-feature category and DP clipping bounds [a_j, b_j], delta_f, source
  gini_importance.csv         # Random Forest Gini (MDI) importance, 68 features, ranked
  metrics.csv                 # JSD, W1 and KL for every feature x variant (816 rows)
  spearman.csv                # Spearman rho and p-value per variant x metric (36 rows)
  feature_metrics_table.csv   # wide table: one row per feature, all variants (written by NB4)
  top15_features_by_jsd.csv   # the 15 features with the highest mean JSD (written by NB4)
figures/
  eda/                        # exploratory figures
  experiment/                 # result figures: JSD/W1/KL heatmaps, DP epsilon-curve, Spearman rho (written by NB4)
```
 
## Data (not included)
 
The datasets are **not in this repository because they are too large for GitHub**. That covers:
 
- the raw CICIDS-2017 CSV files;
- `data/analysis.parquet`, the cleaned analysis set written by NB1;
- `data/anonymised/*.parquet`, the 12 anonymised variants written by NB2.
The files in `results/` are small and are included, so the reported numbers can be checked
without rerunning anything.
 
To reproduce the pipeline, download the CICIDS-2017 **MachineLearningCSV** files from the
Canadian Institute for Cybersecurity: <https://www.unb.ca/cic/datasets/ids-2017.html>.
NB1 reads every file matching `*.pcap_ISCX.csv`
(e.g. `Monday-WorkingHours.pcap_ISCX.csv`) from the raw-data folder described below.
 
## How to run the notebooks
 
The notebooks are written for **Google Colab** and read and write files on Google Drive.
 
1. **Create the folder layout on Google Drive.**
```
   MyDrive/COS700/
     data/raw/        <- put the CICIDS-2017 *.pcap_ISCX.csv files here
```
   The notebooks create `data/anonymised/`, `results/`, `figures/eda/` and
   `figures/experiment/` themselves.
   If you use a different folder, change `DRIVE_ROOT` in the first cell of **each** notebook
   (default: `/content/drive/MyDrive/COS700`).
 
2. **Open each notebook in Colab** (File → Upload notebook, or open it from GitHub) and run all
   cells in order: **NB1 → NB2 → NB3 → NB4**. Each notebook mounts Google Drive on its first
   cell; approve the access prompt.
   | Notebook | Reads | Writes |
   |---|---|---|
   | NB1 | `data/raw/*.pcap_ISCX.csv` | `data/analysis.parquet`, `results/feature_meta.csv`, `results/gini_importance.csv`, `figures/eda/` |
   | NB2 | `analysis.parquet`, `feature_meta.csv`, `gini_importance.csv` | `data/anonymised/*.parquet` (12 files) |
   | NB3 | `analysis.parquet`, the 12 variants, `feature_meta.csv`, `gini_importance.csv` | `results/metrics.csv`, `results/spearman.csv` |
   | NB4 | the four CSVs above, `analysis.parquet` | `figures/experiment/`, `results/feature_metrics_table.csv`, `results/top15_features_by_jsd.csv` |
3. **Check the gate cells.** The last cell of each notebook runs checks (row counts, no NaN
   values, metric ranges) and stops with an error if anything is wrong. Fix any failure before
   running the next notebook.
### Requirements
 
The notebooks use packages that Colab provides by default: numpy, pandas, scipy, scikit-learn,
matplotlib and seaborn. `pyarrow` is installed automatically by the first cell if missing.
A high-RAM runtime is recommended, because the analysis set has about 2.5 million rows and the
Random Forest is fitted on all of it.
 
### Reproducibility
 
All random steps use a fixed seed (42): the Random Forest, the Gaussian and Laplace noise draws,
and the 10,000-row subsample used for W₁. Rerunning the notebooks on the same input files
reproduces the files in `results/`.
