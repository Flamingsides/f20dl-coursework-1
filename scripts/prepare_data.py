"""
prepare_data.py — Step 2 of the data pipeline (R2).

Reads the raw CUB-200-2011 annotation files (data/raw/CUB_200_2011/) and builds
two clean, analysis-ready tables in data/processed/:

  1. cub_attributes.csv  -> Dataset 1 (tabular)
     one row per image: ids, species label, train/test flag,
     and the 312 binary visual attributes (1 = present, 0 = not present).

  2. cub_images.csv      -> Dataset 2 (images)
     one row per image: ids, species label, train/test flag, image path,
     bounding box, image width/height and colour mode.

Both tables share the same image_id and the SAME official train/test split, so
the tabular and image models can be compared fairly (no data leakage).

The cleaning steps below only fix problems in the raw files. Decisions that
need evidence (e.g. what to do with "not visible" attributes, rare attributes,
duplicate rows) are investigated and justified in
notebooks/01_data_cleaning.ipynb.

Usage (from the repository root, conda env activated):
    python scripts/prepare_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# ---------------------------------------------------------------------------
# Paths (relative to the repository root, so this works on every machine)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
RAW = REPO_ROOT / "data" / "raw" / "CUB_200_2011"
OUT = REPO_ROOT / "data" / "processed"

N_IMAGES = 11788      # from the dataset README
N_ATTRIBUTES = 312    # from attributes.txt


# ---------------------------------------------------------------------------
# 1. Small loading helpers
# ---------------------------------------------------------------------------
def read_space_separated(rel_path, column_names):
    """Read one of the simple '<id> <value> ...' text files of the dataset."""
    return pd.read_csv(RAW / rel_path, sep=" ", header=None, names=column_names)


def find_attribute_names_file():
    """
    RAW DATA QUIRK: the dataset README says the attribute names are in
    CUB_200_2011/attributes/attributes.txt, but the official Caltech archive
    puts this file one level up, NEXT TO the CUB_200_2011 folder
    (data/raw/attributes.txt). Some mirrors (e.g. Kaggle) moved it inside.
    We accept both locations and never move or edit the raw files.
    """
    for path in [RAW / "attributes" / "attributes.txt", RAW.parent / "attributes.txt"]:
        if path.exists():
            return path
    raise FileNotFoundError("attributes.txt not found in data/raw/ or "
                            "data/raw/CUB_200_2011/attributes/. Re-run scripts/download_data.py")


def load_attribute_names():
    """
    attributes.txt lines look like: '1 has_bill_shape::curved_(up_or_down)'.
    Returns a DataFrame with attribute_id, attribute_name, and the attribute
    group (the part before '::', e.g. 'has_bill_shape').
    """
    names = pd.read_csv(find_attribute_names_file(), sep=" ", header=None,
                        names=["attribute_id", "attribute_name"])
    names["group"] = names["attribute_name"].str.split("::").str[0]
    return names


def load_image_attribute_labels():
    """
    Read attributes/image_attribute_labels.txt (3.68 million rows: one row per
    image x attribute).

    Expected format (5 columns):
        <image_id> <attribute_id> <is_present> <certainty_id> <time>

    RAW DATA PROBLEM: 606 lines (all belonging to images 2275 and 9364) have an
    extra '0' column, e.g. '2275 10 0 1 0  1.509'. A normal pd.read_csv fails
    or mis-aligns on these. We therefore parse line by line and keep the first
    four fields plus the LAST field (time), which handles both cases.
    """
    rows = []
    n_malformed = 0
    with open(RAW / "attributes" / "image_attribute_labels.txt") as f:
        for line in f:
            parts = line.split()
            if len(parts) != 5:
                n_malformed += 1
            rows.append((int(parts[0]), int(parts[1]), int(parts[2]),
                         int(parts[3]), float(parts[-1])))

    labels = pd.DataFrame(rows, columns=["image_id", "attribute_id",
                                         "is_present", "certainty_id", "time"])
    print(f"  attribute labels: {len(labels):,} rows read, "
          f"{n_malformed} malformed lines repaired")
    return labels


# ---------------------------------------------------------------------------
# 2. Build the shared "image index" (ids, species, split)
# ---------------------------------------------------------------------------
def build_image_index():
    images = read_space_separated("images.txt", ["image_id", "image_path"])
    labels = read_space_separated("image_class_labels.txt", ["image_id", "class_id"])
    split = read_space_separated("train_test_split.txt", ["image_id", "is_train"])
    classes = read_space_separated("classes.txt", ["class_id", "class_folder"])

    # '001.Black_footed_Albatross' -> 'Black footed Albatross'
    classes["species"] = (classes["class_folder"].str.split(".", n=1).str[1]
                          .str.replace("_", " "))
    # Last word of the species name, e.g. 'Albatross', 'Warbler'.
    # NOTE: used ONLY to help interpret EDA/clustering plots. It is NOT a
    # biological family and NOT a prediction target (the lecturer recommended
    # classifying the 200 species).
    classes["name_group"] = classes["species"].str.split().str[-1]

    index = (images.merge(labels, on="image_id")
                   .merge(split, on="image_id")
                   .merge(classes[["class_id", "species", "name_group"]], on="class_id"))

    assert len(index) == N_IMAGES, "image count does not match the README"
    return index


# ---------------------------------------------------------------------------
# 3. Dataset 1: tabular attributes
# ---------------------------------------------------------------------------
def build_attribute_table(index):
    attr_names = load_attribute_names()
    labels = load_image_attribute_labels()

    # Sanity checks: exactly one label per (image, attribute) pair.
    assert labels.duplicated(["image_id", "attribute_id"]).sum() == 0
    assert len(labels) == N_IMAGES * N_ATTRIBUTES

    # The 'time' column is how long the annotator took, not a property of the
    # bird, so it is not used as a feature.
    # Long format -> wide format: one row per image, one column per attribute.
    wide = labels.pivot(index="image_id", columns="attribute_id", values="is_present")
    wide.columns = attr_names.set_index("attribute_id").loc[wide.columns, "attribute_name"]
    wide.columns.name = None
    wide = wide.astype("int8").reset_index()

    table = index[["image_id", "class_id", "species", "name_group", "is_train"]].merge(
        wide, on="image_id")

    # Flag images whose 312 attributes are IDENTICAL to an image of a DIFFERENT
    # species. No model can separate these, so 01_data_cleaning.ipynb decides
    # what to do with them. Here we only flag them (no rows removed).
    attr_cols = list(attr_names["attribute_name"])
    signature = table[attr_cols].astype(str).agg("".join, axis=1)
    n_species_per_signature = table.groupby(signature)["class_id"].transform("nunique")
    table.insert(5, "conflicting_duplicate", (n_species_per_signature > 1).astype("int8"))

    print(f"  attribute table: {table.shape[0]:,} images x {len(attr_cols)} attributes, "
          f"{table['conflicting_duplicate'].sum()} images flagged as conflicting duplicates")
    return table


# ---------------------------------------------------------------------------
# 4. Dataset 2: image metadata
# ---------------------------------------------------------------------------
def build_image_table(index):
    boxes = read_space_separated("bounding_boxes.txt",
                                 ["image_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h"])
    table = index.merge(boxes, on="image_id")

    # Open every image once to record its size and colour mode.
    # RAW DATA PROBLEM: a few images are greyscale (mode 'L') instead of RGB.
    # CNNs expect 3 channels, so R4 must convert them with .convert("RGB").
    widths, heights, modes = [], [], []
    for i, rel_path in enumerate(table["image_path"], start=1):
        with Image.open(RAW / "images" / rel_path) as img:
            widths.append(img.width)
            heights.append(img.height)
            modes.append(img.mode)
        if i % 2000 == 0 or i == len(table):
            print(f"\r  checking images... {i:,}/{len(table):,}", end="")
    print()

    table["width"], table["height"], table["mode"] = widths, heights, modes

    # RAW DATA PROBLEM: a few bounding boxes stick out past the image edge
    # (by up to ~30 px). Cropping with such a box would fail or add padding,
    # so we clip every box to the image borders (1 px tolerance for rounding).
    inside = ((table.bbox_x >= 0) & (table.bbox_y >= 0) &
              (table.bbox_x + table.bbox_w <= table.width + 1) &
              (table.bbox_y + table.bbox_h <= table.height + 1))
    table["bbox_clipped"] = (~inside).astype("int8")
    table["bbox_x"] = table["bbox_x"].clip(lower=0)
    table["bbox_y"] = table["bbox_y"].clip(lower=0)
    table["bbox_w"] = np.minimum(table["bbox_w"], table["width"] - table["bbox_x"])
    table["bbox_h"] = np.minimum(table["bbox_h"], table["height"] - table["bbox_y"])

    print(f"  image table: {len(table):,} images, "
          f"{(table['mode'] != 'RGB').sum()} not RGB, "
          f"{int(table['bbox_clipped'].sum())} bounding boxes clipped to the image border")
    return table


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if not (RAW / "images.txt").exists():
        raise SystemExit("Raw data not found. Run: python scripts/download_data.py")

    OUT.mkdir(parents=True, exist_ok=True)
    print("[1/3] Building image index...")
    index = build_image_index()

    print("[2/3] Building Dataset 1 (tabular attributes)...")
    build_attribute_table(index).to_csv(OUT / "cub_attributes.csv", index=False)

    print("[3/3] Building Dataset 2 (image metadata)...")
    build_image_table(index).to_csv(OUT / "cub_images.csv", index=False)

    print(f"\nDone. Files written to {OUT}:")
    for f in sorted(OUT.glob("*.csv")):
        print(f"  {f.name:22s} {f.stat().st_size / 1e6:6.1f} MB")
