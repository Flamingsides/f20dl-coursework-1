"""
prepare_images.py — Step 2 of the image pipeline (R2, for R4).

Turns Dataset 2 (data/cub_images.csv, made by notebooks/imagePrep.ipynb) into
model-ready image files, applying the findings of that notebook:
  (a) convert the 8 greyscale images to RGB,
  (b) crop to the (already clipped) bounding box, plus a small margin,
  (c) resize so the SHORTER side is TARGET_SHORT px (aspect ratio kept).

This step is deterministic (same output every run), so it is applied to train
AND test. Random augmentation is NOT done here: it happens on the fly, on the
training split only, in scripts/augmentation.py.

Output: data/processed/images_cropped/<same sub-folders as the raw images>
        data/processed/cub_images_cropped.csv  (cub_images.csv + cropped_path)

Usage (repo root, environment active, after running notebooks/imagePrep.ipynb):
    python scripts/prepare_images.py
    python scripts/prepare_images.py --short-side 512   # larger copy for 448 px models (resnet50.ipynb)
        -> data/processed/images_cropped_512/ and data/processed/cub_images_cropped_512.csv
"""
import argparse
from pathlib import Path

import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
# Dataset folder, found the same way as in notebooks/dataPrep.ipynb: inside data/raw/, or CUB_200_2011/ at the repo root
CUB = next((p.parent for p in (REPO_ROOT / "data" / "raw").rglob("image_class_labels.txt")), REPO_ROOT / "CUB_200_2011")
RAW_IMAGES = CUB / "images"
IMAGE_TABLE = REPO_ROOT / "data" / "cub_images.csv"   # written by notebooks/imagePrep.ipynb
PROCESSED = REPO_ROOT / "data" / "processed"
OUT_DIR = PROCESSED / "images_cropped"

MARGIN = 0.10        # add 10% of the box size on each side: keeps bill/tail tips that the
                     # box sometimes cuts, and leaves room for RandomResizedCrop later
TARGET_SHORT = 256   # store slightly larger than the model input (224) so random crops have room


def crop_box(row, margin=MARGIN):
    """Bounding box + margin, clipped to the image border. Returns (left, top, right, bottom)."""
    dx, dy = row.bbox_w * margin, row.bbox_h * margin
    left = max(0, row.bbox_x - dx)
    top = max(0, row.bbox_y - dy)
    right = min(row.width, row.bbox_x + row.bbox_w + dx)
    bottom = min(row.height, row.bbox_y + row.bbox_h + dy)
    return tuple(int(round(v)) for v in (left, top, right, bottom))


def output_paths(short_side):
    """Image folder and table for a stored size. 256 px keeps the original names used by the other notebooks."""
    suffix = "" if short_side == TARGET_SHORT else f"_{short_side}"
    return PROCESSED / f"images_cropped{suffix}", PROCESSED / f"cub_images_cropped{suffix}.csv"


def process_one(row, out_dir=OUT_DIR, target_short=TARGET_SHORT):
    with Image.open(RAW_IMAGES / row.image_path) as img:
        img = img.convert("RGB")                       # (a) greyscale -> 3 channels
        img = img.crop(crop_box(row))                  # (b) crop to the bird
        scale = target_short / min(img.size)           # (c) shorter side -> target_short
        new_size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
        img = img.resize(new_size, Image.BICUBIC)
        out = out_dir / row.image_path
        out.parent.mkdir(parents=True, exist_ok=True)
        img.save(out, quality=95)
    return out.relative_to(PROCESSED).as_posix()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--short-side", type=int, default=TARGET_SHORT,
                        help=f"stored size of the shorter side in px (default {TARGET_SHORT})")
    short_side = parser.parse_args().short_side
    out_dir, out_table = output_paths(short_side)

    table = pd.read_csv(IMAGE_TABLE)
    paths = []
    for i, row in enumerate(table.itertuples(index=False), start=1):
        paths.append(process_one(row, out_dir, short_side))
        if i % 1000 == 0 or i == len(table):
            print(f"\r  cropping images... {i:,}/{len(table):,}", end="", flush=True)
    print()
    table["cropped_path"] = paths
    table.to_csv(out_table, index=False)
    print(f"Done: {len(table):,} images written to {out_dir}")
