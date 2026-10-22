"""
r2_utils.py — small shared helpers for the R2 notebooks (and anyone else's).

Keeps paths, plot style and data loading in ONE place, so every notebook
loads the data the same way and the figures look consistent.

Use it from a notebook in notebooks/ with:
    import sys; sys.path.insert(0, "../scripts")
    import r2_utils as u
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def find_repo_root(start=None):
    """Walk up from the current folder until we find environment.yml (the repo root)."""
    here = Path(start or Path.cwd()).resolve()
    for folder in [here, *here.parents]:
        if (folder / "environment.yml").exists():
            return folder
    raise FileNotFoundError("Could not find the repository root (environment.yml).")


REPO_ROOT = find_repo_root(Path(__file__).parent)
RAW_DIR = REPO_ROOT / "data" / "raw" / "CUB_200_2011"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
FIGURE_DIR = REPO_ROOT / "documentation" / "figures"   # small PNGs, committed for the README

RANDOM_STATE = 42   # one seed for every random step, so results are reproducible

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------
# A small, colour-blind-friendly categorical palette (used in a fixed order).
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
SEQ_CMAP = "Blues"        # one hue, light -> dark, for magnitudes
DIV_CMAP = "RdBu_r"       # two hues around a neutral middle, for +/- correlations


def set_plot_style():
    """Consistent, quiet styling for every figure in the project."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.titleweight": "bold",
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "font.size": 9,
        "legend.frameon": False,
    })


def save_figure(fig, name):
    """Save a figure as documentation/figures/<name>.png (for the README/report)."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{name}.png")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
META_COLUMNS = ["image_id", "class_id", "species", "name_group", "is_train",
                "conflicting_duplicate"]


def load_attributes(clean=True):
    """
    Load Dataset 1 (tabular).
      clean=True  -> data/processed/cub_attributes_clean.csv (after notebook 01)
      clean=False -> data/processed/cub_attributes.csv       (straight from the pipeline)
    """
    name = "cub_attributes_clean.csv" if clean else "cub_attributes.csv"
    path = PROCESSED_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} not found. Run scripts/prepare_data.py"
            + (" and notebooks/01_data_cleaning.ipynb" if clean else "") + " first.")
    return pd.read_csv(path).copy()   # .copy() keeps the 300+ columns in one memory block (faster)


def attribute_columns(df):
    """The 312 attribute columns are the ones named like 'has_xxx::value'."""
    return [c for c in df.columns if "::" in c]


def attribute_group(column):
    """'has_bill_shape::dagger' -> 'bill_shape'."""
    return column.split("::")[0].replace("has_", "")


def attribute_value(column):
    """'has_bill_shape::dagger' -> 'dagger'."""
    return column.split("::")[1]


def load_images():
    """Load Dataset 2 (image metadata: paths, bounding boxes, sizes)."""
    path = PROCESSED_DIR / "cub_images.csv"
    if not path.exists():
        raise FileNotFoundError("cub_images.csv not found. Run scripts/prepare_data.py first.")
    return pd.read_csv(path)
