"""
prepare_images.py — Step 3 of the data pipeline (R2, image side).

Turns Dataset 2 (cub_images.csv, made by prepare_data.py) into model-ready
image files, applying Decision 8 of notebooks/01_data_cleaning.ipynb:
  (a) convert the 8 greyscale images to RGB,
  (b) crop to the (already clipped) bounding box, plus a small margin,
  (c) resize so the SHORTER side is TARGET_SHORT px (aspect ratio kept).

This step is deterministic (same output every run), so it is applied to train
AND test. Random augmentation is NOT done here: it happens on the fly, on the
training split only, in scripts/augmentation.py.

Output: data/processed/images_cropped/<same sub-folders as the raw images>
        data/processed/cub_images_cropped.csv  (cub_images.csv + cropped_path)

Usage (repo root, conda env active):
    python scripts/prepare_images.py
"""
from pathlib import Path

import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_IMAGES = REPO_ROOT / "data" / "raw" / "CUB_200_2011" / "images"
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


def process_one(row):
    with Image.open(RAW_IMAGES / row.image_path) as img:
        img = img.convert("RGB")                       # (a) greyscale -> 3 channels
        img = img.crop(crop_box(row))                  # (b) crop to the bird
        scale = TARGET_SHORT / min(img.size)           # (c) shorter side -> TARGET_SHORT
        new_size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
        img = img.resize(new_size, Image.BICUBIC)
        out = OUT_DIR / row.image_path
        out.parent.mkdir(parents=True, exist_ok=True)
        img.save(out, quality=95)
    return out.relative_to(PROCESSED).as_posix()


if __name__ == "__main__":
    table = pd.read_csv(PROCESSED / "cub_images.csv")
    paths = []
    for i, row in enumerate(table.itertuples(index=False), start=1):
        paths.append(process_one(row))
        if i % 1000 == 0 or i == len(table):
            print(f"\r  cropping images... {i:,}/{len(table):,}", end="")
    print()
    table["cropped_path"] = paths
    table.to_csv(PROCESSED / "cub_images_cropped.csv", index=False)
    print(f"Done: {len(table):,} images written to {OUT_DIR}")
