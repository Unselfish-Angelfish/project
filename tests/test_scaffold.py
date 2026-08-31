"""
test_scaffold.py — Task 1 acceptance tests.

Verifies:
  1. All required directories exist
  2. src/config.py defines all required constants with correct types
  3. All src stub modules import without error
"""

import importlib
from pathlib import Path

import pytest

# Project root is two levels up from this file (tests/ → Research/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# 1. Directory structure
# ---------------------------------------------------------------------------
REQUIRED_DIRS = [
    "data/raw",
    "data/processed",
    "data/anonymised",
    "src",
    "results",
    "figures/eda",
    "figures/experiment",
    "tests",
]


@pytest.mark.parametrize("rel_path", REQUIRED_DIRS)
def test_directory_exists(rel_path: str) -> None:
    """Every required directory must exist."""
    target = PROJECT_ROOT / rel_path
    assert target.is_dir(), f"Missing directory: {target}"


# ---------------------------------------------------------------------------
# 2. config.py constants
# ---------------------------------------------------------------------------
def test_config_imports_without_error() -> None:
    """config.py must import cleanly."""
    import src.config  # noqa: F401


def test_config_random_state() -> None:
    from src.config import RANDOM_STATE
    assert isinstance(RANDOM_STATE, int)
    assert RANDOM_STATE == 42


def test_config_n_features() -> None:
    from src.config import N_FEATURES
    assert isinstance(N_FEATURES, int)
    assert N_FEATURES == 78


def test_config_label_col() -> None:
    from src.config import LABEL_COL
    assert isinstance(LABEL_COL, str)
    assert LABEL_COL == "Label"


def test_config_noise_sigmas() -> None:
    from src.config import NOISE_SIGMAS
    assert isinstance(NOISE_SIGMAS, list)
    assert NOISE_SIGMAS == [0.1, 0.5]


def test_config_suppression_levels() -> None:
    from src.config import SUPPRESSION_LEVELS
    assert isinstance(SUPPRESSION_LEVELS, list)
    assert SUPPRESSION_LEVELS == [0.25, 0.50, 0.75]


def test_config_dp_epsilons() -> None:
    from src.config import DP_EPSILONS
    assert isinstance(DP_EPSILONS, list)
    assert len(DP_EPSILONS) == 6
    assert 0.01 in DP_EPSILONS
    assert 10.0 in DP_EPSILONS


def test_config_n_bins() -> None:
    from src.config import N_BINS
    assert isinstance(N_BINS, int)
    assert N_BINS == 100


def test_config_kl_smooth() -> None:
    from src.config import KL_SMOOTH
    assert isinstance(KL_SMOOTH, float)
    assert KL_SMOOTH == 1e-10


def test_config_plot_style_keys() -> None:
    from src.config import PLOT_STYLE
    required_keys = {"bg_color", "fg_color", "grid_color", "dpi", "font_size", "title_size"}
    assert required_keys.issubset(set(PLOT_STYLE.keys()))
    assert PLOT_STYLE["dpi"] == 300


def test_config_paths_are_pathlib() -> None:
    from src.config import (
        ANONYMISED_DIR,
        CICIDS_SOURCE_DIR,
        FIGURES_EDA_DIR,
        FIGURES_EXP_DIR,
        PROCESSED_DIR,
        PROJECT_ROOT,
        RESULTS_DIR,
    )
    for p in [PROJECT_ROOT, PROCESSED_DIR, ANONYMISED_DIR,
              RESULTS_DIR, FIGURES_EDA_DIR, FIGURES_EXP_DIR, CICIDS_SOURCE_DIR]:
        assert isinstance(p, Path), f"{p} is not a Path object"


def test_config_output_dirs_created() -> None:
    """Importing config should auto-create output directories."""
    import src.config as cfg  # noqa: F401 — importing triggers mkdir
    from src.config import ANONYMISED_DIR, FIGURES_EDA_DIR, FIGURES_EXP_DIR, PROCESSED_DIR, RESULTS_DIR
    for d in [PROCESSED_DIR, ANONYMISED_DIR, RESULTS_DIR, FIGURES_EDA_DIR, FIGURES_EXP_DIR]:
        assert d.is_dir(), f"config failed to create: {d}"


# ---------------------------------------------------------------------------
# 3. Stub modules importable
# ---------------------------------------------------------------------------
STUB_MODULES = [
    "src.preprocessing",
    "src.eda",
    "src.anonymise",
    "src.metrics",
    "src.visualise",
]


@pytest.mark.parametrize("module_name", STUB_MODULES)
def test_stub_imports_without_error(module_name: str) -> None:
    """Every stub module must import cleanly (no syntax errors, no missing deps)."""
    mod = importlib.import_module(module_name)
    assert mod is not None


@pytest.mark.parametrize("module_name", STUB_MODULES)
def test_stub_has_main_function(module_name: str) -> None:
    """Every stub module must have a callable main() function."""
    mod = importlib.import_module(module_name)
    assert hasattr(mod, "main"), f"{module_name} missing main()"
    assert callable(mod.main), f"{module_name}.main is not callable"


# ---------------------------------------------------------------------------
# 4. run_pipeline.py importable
# ---------------------------------------------------------------------------
def test_run_pipeline_importable() -> None:
    """run_pipeline.py must import cleanly."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_pipeline",
        PROJECT_ROOT / "run_pipeline.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "main")


# ---------------------------------------------------------------------------
# 5. Key files exist
# ---------------------------------------------------------------------------
REQUIRED_FILES = [
    "PLAN.md",
    "README.md",
    "requirements.txt",
    "run_pipeline.py",
    "src/__init__.py",
    "src/config.py",
    "src/preprocessing.py",
    "src/eda.py",
    "src/anonymise.py",
    "src/metrics.py",
    "src/visualise.py",
    "tests/__init__.py",
    "tests/test_scaffold.py",
]


@pytest.mark.parametrize("rel_path", REQUIRED_FILES)
def test_file_exists(rel_path: str) -> None:
    """Every required file must exist."""
    target = PROJECT_ROOT / rel_path
    assert target.is_file(), f"Missing file: {target}"
