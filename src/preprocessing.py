"""
preprocessing.py — Data loading, cleaning, and splitting for CICIDS-2017.

Pipeline:
  load_raw()                  → load all 8 CSVs, strip column names, concatenate
  clean()                     → remove inf, NaN, duplicates; report dropped rows
  identify_numeric_features() → list the 78 numeric feature column names
  split()                     → stratified 70 / 15 / 15 train / val / test
  save_splits()               → write parquet files to data/processed/
  main()                      → orchestrate all steps and print a full report
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    CICIDS_SOURCE_DIR,
    LABEL_COL,
    N_FEATURES,
    PROCESSED_DIR,
    RANDOM_STATE,
    TEST_SIZE,
)


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------

def load_raw(raw_dir: Path = CICIDS_SOURCE_DIR) -> pd.DataFrame:
    """
    Glob all *.pcap_ISCX.csv files from raw_dir, read each with pd.read_csv,
    strip leading/trailing whitespace from ALL column names, concatenate into
    a single DataFrame.

    Strips whitespace from the Label column values as well to normalise
    the known encoding artefacts in some CICIDS-2017 files (e.g. 'Web Attack \\xc2\\xa0...').

    Returns the concatenated DataFrame.
    """
    csv_files = sorted(Path(raw_dir).glob("*.pcap_ISCX.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No .pcap_ISCX.csv files found in {raw_dir}.\n"
            "Expected path: /Users/siphesihlekhumalo/Downloads/archive/"
        )

    frames = []
    print(f"\nLoading {len(csv_files)} CSV files from {raw_dir}")
    print("-" * 60)
    for f in csv_files:
        df = pd.read_csv(f, low_memory=False)
        # Strip whitespace from ALL column names (known CICIDS-2017 issue)
        df.columns = [c.strip() for c in df.columns]
        # Strip Label values
        if LABEL_COL in df.columns:
            df[LABEL_COL] = df[LABEL_COL].str.strip()
        print(f"  {f.name:55s}  {len(df):>8,} rows")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    print("-" * 60)
    print(f"  {'TOTAL':55s}  {len(combined):>8,} rows")
    return combined


# ---------------------------------------------------------------------------
# 2. Clean
# ---------------------------------------------------------------------------

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the three-step cleaning procedure from the proposal:
      1. Replace np.inf / -np.inf with NaN
      2. Drop all rows containing any NaN
      3. Drop exact duplicate rows

    Prints a row-count report at each stage.
    Returns the cleaned DataFrame.
    """
    n_start = len(df)
    print(f"\nCleaning  (start: {n_start:,} rows)")
    print("-" * 60)

    # Step 1: inf → NaN
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    n_inf_before = np.isinf(df[numeric_cols]).sum().sum()
    df = df.replace([np.inf, -np.inf], np.nan)
    print(f"  Inf values replaced with NaN: {n_inf_before:,}")

    # Step 2: drop NaN rows
    n_before_nan = len(df)
    df = df.dropna()
    n_nan_dropped = n_before_nan - len(df)
    print(f"  Rows dropped (contained NaN): {n_nan_dropped:,}")

    # Step 3: drop duplicates
    n_before_dup = len(df)
    df = df.drop_duplicates()
    n_dup_dropped = n_before_dup - len(df)
    print(f"  Rows dropped (exact duplicates): {n_dup_dropped:,}")

    n_end = len(df)
    n_total_dropped = n_start - n_end
    print("-" * 60)
    print(f"  Surviving rows: {n_end:,}  "
          f"(dropped {n_total_dropped:,} = "
          f"{100 * n_total_dropped / n_start:.1f}%)")

    df = df.reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 3. Identify numeric features
# ---------------------------------------------------------------------------

def identify_numeric_features(
    df: pd.DataFrame,
    label_col: str = LABEL_COL,
) -> list[str]:
    """
    Return a sorted list of all numeric (float64 / int64) column names,
    excluding label_col.

    Asserts that exactly N_FEATURES (78) numeric feature columns are found.
    """
    numeric_cols = [
        c for c in df.columns
        if c != label_col and pd.api.types.is_numeric_dtype(df[c])
    ]
    numeric_cols = sorted(numeric_cols)

    if len(numeric_cols) != N_FEATURES:
        raise ValueError(
            f"Expected {N_FEATURES} numeric feature columns, "
            f"found {len(numeric_cols)}.\n"
            f"Columns found: {numeric_cols}"
        )
    return numeric_cols


# ---------------------------------------------------------------------------
# 4. Split
# ---------------------------------------------------------------------------

def split(
    df: pd.DataFrame,
    label_col: str = LABEL_COL,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified 70 / 15 / 15 split.

    Step 1: hold out 30% (val + test) with stratify=label.
    Step 2: split the 30% pool in half (50/50) to get val and test,
            each 15% of the total.

    Uses random_state=RANDOM_STATE throughout for reproducibility.
    Returns (train, val, test).
    """
    labels = df[label_col]

    # Step 1: 70% train, 30% temp
    train, temp = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    # Step 2: 50% val, 50% test from the 30% temp pool
    val, test = train_test_split(
        temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp[label_col],
    )

    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5. Save
# ---------------------------------------------------------------------------

def save_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    processed_dir: Path = PROCESSED_DIR,
) -> None:
    """
    Save the three splits as parquet files using the pyarrow engine.
    Files: train.parquet, val.parquet, test.parquet
    """
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    train.to_parquet(processed_dir / "train.parquet", engine="pyarrow", index=False)
    val.to_parquet(processed_dir / "val.parquet",   engine="pyarrow", index=False)
    test.to_parquet(processed_dir / "test.parquet",  engine="pyarrow", index=False)

    print(f"\n  Saved to {processed_dir}")
    print(f"    train.parquet : {len(train):>8,} rows")
    print(f"    val.parquet   : {len(val):>8,} rows")
    print(f"    test.parquet  : {len(test):>8,} rows")


# ---------------------------------------------------------------------------
# 6. Report helpers
# ---------------------------------------------------------------------------

def _class_distribution(df: pd.DataFrame, label_col: str = LABEL_COL) -> pd.DataFrame:
    """Return a DataFrame with class counts and percentages, sorted descending."""
    counts = df[label_col].value_counts()
    pct = (counts / len(df) * 100).round(2)
    return pd.DataFrame({"count": counts, "pct": pct})


def print_report(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
) -> None:
    """Print the full preprocessing report."""
    total = len(train) + len(val) + len(test)

    print("\n" + "=" * 60)
    print("  PREPROCESSING REPORT")
    print("=" * 60)

    print(f"\n  Split sizes:")
    print(f"    Train : {len(train):>8,}  ({100*len(train)/total:.1f}%)")
    print(f"    Val   : {len(val):>8,}  ({100*len(val)/total:.1f}%)")
    print(f"    Test  : {len(test):>8,}  ({100*len(test)/total:.1f}%)")
    print(f"    Total : {total:>8,}")

    print(f"\n  Numeric features : {len(feature_cols)}")
    print(f"  Feature names    :")
    for i, col in enumerate(feature_cols):
        print(f"    [{i+1:2d}] {col}")

    print(f"\n  Class distribution (train):")
    dist = _class_distribution(train)
    for cls, row in dist.iterrows():
        print(f"    {cls:<45s}  {int(row['count']):>7,}  ({row['pct']:5.2f}%)")

    print("\n" + "=" * 60)


# ---------------------------------------------------------------------------
# 7. main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Orchestrate the full preprocessing pipeline:
    load → clean → identify features → split → save → report
    """
    print("\n[preprocessing] Starting...")

    # Load
    df_raw = load_raw()

    # Clean
    df_clean = clean(df_raw)
    del df_raw  # free memory

    # Identify numeric features
    feature_cols = identify_numeric_features(df_clean)
    print(f"\n  Identified {len(feature_cols)} numeric features ✓")

    # Split
    print("\n  Splitting (stratified 70/15/15)...")
    train, val, test = split(df_clean)
    del df_clean

    # Save
    save_splits(train, val, test)

    # Report
    print_report(train, val, test, feature_cols)

    print("\n[preprocessing] Done ✓")


if __name__ == "__main__":
    main()
