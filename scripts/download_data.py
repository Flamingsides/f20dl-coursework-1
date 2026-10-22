"""
download_data.py — Step 1 of the data pipeline (R2).

Downloads the ORIGINAL CUB-200-2011 dataset from its official source
(Caltech Data repository) and extracts it into data/raw/.

Why the official source and not Kaggle?
  * The CW Handbook asks for the "true source" of the data, and every
    teammate (and marker) must get an identical copy.
  * The Kaggle mirror needs an API key and contains extra, non-original files
    (e.g. a pre-"cleaned" attribute file). We do our own cleaning instead,
    so every step is documented and reproducible.

Usage (run from the repository root, with the conda env activated):
    python scripts/download_data.py

Result:
    data/raw/CUB_200_2011.tgz        (the downloaded archive, ~1.1 GB)
    data/raw/CUB_200_2011/...        (the extracted dataset)

Running it again is safe: it skips anything that is already done.

Dataset citation:
    Wah, C., Branson, S., Welinder, P., Perona, P. and Belongie, S. (2011).
    The Caltech-UCSD Birds-200-2011 Dataset. Technical Report CNS-TR-2011-001,
    California Institute of Technology.
Licence: images are for non-commercial research and educational use only
(see https://www.vision.caltech.edu/datasets/cub_200_2011/).
"""

import sys
import tarfile
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
# Official download link (Caltech Data). It redirects (HTTP 302) to a storage
# server; urllib follows that redirect automatically.
DATA_URL = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"

# Paths are built relative to the repository root, so the script works on any
# machine (Windows / macOS / Linux) no matter where the repo was cloned.
REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
ARCHIVE_PATH = RAW_DIR / "CUB_200_2011.tgz"
DATASET_DIR = RAW_DIR / "CUB_200_2011"

# Numbers stated in the dataset's README, used to check the download is complete.
EXPECTED_IMAGES = 11788
EXPECTED_CLASSES = 200

# Files the rest of the pipeline needs (relative to DATASET_DIR).
REQUIRED_FILES = [
    "images.txt",
    "image_class_labels.txt",
    "train_test_split.txt",
    "classes.txt",
    "bounding_boxes.txt",
    "attributes/image_attribute_labels.txt",
    "attributes/certainties.txt",
    "attributes/class_attribute_labels_continuous.txt",
    "parts/parts.txt",
    "parts/part_locs.txt",
]


# The Caltech server is behind Cloudflare, which rejects Python's default
# identity ("Python-urllib", HTTP 403 Forbidden). Sending a normal browser-style
# User-Agent header avoids this.
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) f20dl-coursework-download"}

CHUNK_SIZE = 1024 * 1024   # download 1 MB at a time


def print_manual_instructions():
    """What to do if the automatic download is blocked on your network."""
    print("\n      Automatic download failed. Download it manually instead:")
    print(f"        1. Open this link in your browser: {DATA_URL}")
    print(f"        2. Save the file as: {ARCHIVE_PATH}")
    print("        3. Run this script again (it will skip the download and extract the file).")


def download_archive():
    """Download the .tgz archive, unless it (or the extracted dataset) is already there."""
    if (DATASET_DIR / "images.txt").exists():
        print(f"[1/3] Dataset folder already present, no download needed: {DATASET_DIR}")
        return
    if ARCHIVE_PATH.exists() and ARCHIVE_PATH.stat().st_size > 1e9:
        print(f"[1/3] Archive already downloaded: {ARCHIVE_PATH}")
        return

    print(f"[1/3] Downloading CUB-200-2011 (~1.1 GB) from\n      {DATA_URL}")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Download to a temporary name first, so an interrupted download is never
    # mistaken for a complete one.
    tmp_path = ARCHIVE_PATH.with_suffix(".tgz.part")
    request = urllib.request.Request(DATA_URL, headers=HEADERS)
    try:
        with urllib.request.urlopen(request) as response, open(tmp_path, "wb") as out:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            while chunk := response.read(CHUNK_SIZE):
                out.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(f"\r      {downloaded * 100 / total:5.1f}% "
                          f"({downloaded / 1e6:,.0f} / {total / 1e6:,.0f} MB)", end="")
                else:
                    print(f"\r      {downloaded / 1e6:,.0f} MB", end="")
    except Exception as error:      # network blocked, server error, ...
        print(f"\n      ERROR: {error}")
        print_manual_instructions()
        sys.exit(1)

    tmp_path.replace(ARCHIVE_PATH)
    print(f"\n      saved to {ARCHIVE_PATH}")


def extract_archive():
    """Extract the archive into data/raw/, unless already extracted."""
    if (DATASET_DIR / "images.txt").exists():
        print(f"[2/3] Already extracted: {DATASET_DIR}")
        return

    print(f"[2/3] Extracting to {RAW_DIR} (this can take a few minutes)...")
    with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
        # The 'data' filter (Python >= 3.11.4) blocks unsafe paths inside the
        # archive. Older versions do not have it, so we fall back gracefully.
        if hasattr(tarfile, "data_filter"):
            tar.extractall(RAW_DIR, filter="data")
        else:
            tar.extractall(RAW_DIR)
    print("      done")


def verify_dataset():
    """Check that all expected files exist and the counts match the README."""
    print("[3/3] Verifying the dataset...")
    problems = []

    for rel_path in REQUIRED_FILES:
        if not (DATASET_DIR / rel_path).exists():
            problems.append(f"missing file: {rel_path}")

    # The official archive stores attributes.txt next to the CUB_200_2011 folder
    # (not inside attributes/ as its README says); some mirrors move it inside.
    if not ((DATASET_DIR / "attributes" / "attributes.txt").exists()
            or (RAW_DIR / "attributes.txt").exists()):
        problems.append("missing file: attributes.txt")

    class_dirs = [d for d in (DATASET_DIR / "images").iterdir() if d.is_dir()]
    if len(class_dirs) != EXPECTED_CLASSES:
        problems.append(f"expected {EXPECTED_CLASSES} class folders, found {len(class_dirs)}")

    n_images = sum(1 for d in class_dirs for f in d.iterdir() if f.suffix.lower() == ".jpg")
    if n_images != EXPECTED_IMAGES:
        problems.append(f"expected {EXPECTED_IMAGES} images, found {n_images}")

    if problems:
        print("      PROBLEMS FOUND:")
        for p in problems:
            print("       -", p)
        print("      Delete data/raw/ and run this script again.")
        sys.exit(1)

    print(f"      OK: {EXPECTED_CLASSES} classes, {n_images} images, all annotation files present")


if __name__ == "__main__":
    download_archive()
    extract_archive()
    verify_dataset()
    print("\nRaw data ready. Next step: python scripts/prepare_data.py")
