"""
run_pipeline.py — Top-level runner for the COS700 experiment.

Executes all pipeline steps in sequence:
  Step 1: Preprocessing  — load, clean, split CICIDS-2017
  Step 2: EDA            — 7 exploratory figures + gini_importance.csv
  Step 3: Anonymisation  — 11 anonymised training set variants
  Step 4: Metrics        — JSD / Wasserstein / KL + Spearman ρ
  Step 5: Visualisation  — 12 publication-ready result figures

Run: python run_pipeline.py
"""

import logging
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _banner(step: int, name: str) -> None:
    line = "=" * 60
    log.info(line)
    log.info(f"  STEP {step}: {name}")
    log.info(line)


def main() -> None:
    steps = [
        (1, "Preprocessing",  "src.preprocessing"),
        (2, "EDA",            "src.eda"),
        (3, "Anonymisation",  "src.anonymise"),
        (4, "Metrics",        "src.metrics"),
        (5, "Visualisation",  "src.visualise"),
    ]

    total_start = time.time()

    for step_num, step_name, module_path in steps:
        _banner(step_num, step_name)
        t0 = time.time()

        # Dynamically import and call main()
        import importlib
        module = importlib.import_module(module_path)
        module.main()

        elapsed = time.time() - t0
        log.info(f"  Step {step_num} completed in {elapsed:.1f}s")
        print()

    total_elapsed = time.time() - total_start
    log.info("=" * 60)
    log.info(f"  PIPELINE COMPLETE — total time: {total_elapsed:.1f}s")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
