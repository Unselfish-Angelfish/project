# COS700 — Impact of Data Anonymisation on the Statistical Utility of Cybersecurity Datasets

**Student:** Siphesihle Khumalo (u25759257)
**Supervisor:** Prof. Eloff
**Module:** COS700, University of Pretoria

## Overview

This project implements a systematic, feature-level empirical analysis of how three
anonymisation techniques — noise perturbation, suppression, and differential privacy —
distort the statistical utility of the CICIDS-2017 benchmark cybersecurity dataset across
all 78 numerical features.

For each technique and intensity level, Jensen–Shannon Divergence (JSD), Wasserstein
Distance, and KL Divergence are computed per feature. Spearman rank correlation between
per-feature distortion scores and Random Forest Gini importance quantifies whether each
technique preferentially damages the features most critical to IDS classification.

## Project Structure

```
Research/
├── data/
│   ├── raw/            # source CSVs (point to ~/Downloads/archive/)
│   ├── processed/      # train.parquet, val.parquet, test.parquet
│   └── anonymised/     # 11 anonymised training parquets
├── src/
│   ├── config.py       # all constants and paths
│   ├── preprocessing.py
│   ├── eda.py
│   ├── anonymise.py
│   ├── metrics.py
│   └── visualise.py
├── results/            # gini_importance.csv, metrics.csv, spearman.csv
├── figures/
│   ├── eda/            # 7 EDA figures
│   └── experiment/     # 12 result figures
├── tests/
├── run_pipeline.py
└── PLAN.md
```

## Setup

```bash
cd /Users/siphesihlekhumalo/Desktop/Research
pip install -r requirements.txt
```

The CICIDS-2017 dataset must be present at:
`/Users/siphesihlekhumalo/Downloads/archive/` (8 `.pcap_ISCX.csv` files)

## Running the Pipeline

Run the full pipeline (each step runs sequentially):
```bash
python run_pipeline.py
```

Run individual steps:
```bash
python -m src.preprocessing   # Step 1: load, clean, split → data/processed/
python -m src.eda              # Step 2: EDA figures + gini_importance.csv
python -m src.anonymise        # Step 3: 11 anonymised training sets
python -m src.metrics          # Step 4: metrics.csv + spearman.csv
python -m src.visualise        # Step 5: 12 result figures
```

Run all tests:
```bash
pytest tests/ -v
```

## Key Expected Values

| Stage | Expected value |
|-------|----------------|
| Raw rows (8 CSVs) | ~2,830,743 |
| After cleaning | ~2,270,000–2,400,000 |
| Numeric features | 78 |
| Attack classes | 14 (including BENIGN) |
| Anonymised variants | 11 |
| `metrics.csv` rows | 858 (11 × 78) |
| `spearman.csv` rows | 33 (11 × 3) |
| EDA figures | 7 |
| Result figures | 12 |

## Anonymisation Techniques

| Technique | Levels | Output files |
|-----------|--------|-------------|
| Noise perturbation | σ ∈ {0.1, 0.5} | `train_noise_0.1.parquet`, `train_noise_0.5.parquet` |
| Suppression | 25%, 50%, 75% | `train_suppression_0.25/0.50/0.75.parquet` |
| Differential Privacy | ε ∈ {0.01, 0.1, 0.5, 1.0, 5.0, 10.0} | `train_dp_{ε}.parquet` |
