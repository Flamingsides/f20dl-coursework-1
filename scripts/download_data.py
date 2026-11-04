"""
Download the CUB-200-2011 dataset from Kaggle into data/raw/.

Usage (from the repo root):
    python scripts/download_data.py

Requires a Kaggle API token (see README: "Getting the data").
The data is NOT committed to Git (data/raw/ is in .gitignore); everyone runs this
script so the whole team uses an identical copy, as required by the CW Handbook (4.3.1).
"""
# --- GenAI disclosure (F20DL coursework spec, Section 8) ---
# Code generated with Claude (Anthropic) and reviewed by <YOUR NAME>.
# Prompt used, in full: "the data is too big to be put in the repo"

import sys
from pathlib import Path

DATASET = "wenewone/cub2002011"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def main():
    if any(RAW_DIR.rglob("image_class_labels.txt")):
        print(f"Dataset already present in {RAW_DIR}. Delete that folder to re-download.")
        return

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        sys.exit("The 'kaggle' package is missing. Run: pip install kaggle")

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as e:
        sys.exit(f"Kaggle login failed ({e}).\n"
                 "Create a token at kaggle.com > Settings > API > Create New Token,\n"
                 "then place kaggle.json in ~/.kaggle/ (Windows: C:\\Users\\<you>\\.kaggle\\).")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATASET} to {RAW_DIR} (about 1 GB, this can take a while)...")
    api.dataset_download_files(DATASET, path=str(RAW_DIR), unzip=True, quiet=False)

    if not any(RAW_DIR.rglob("image_class_labels.txt")):
        sys.exit("Download finished but the expected files were not found. Check the folder contents.")
    print("Done.")


if __name__ == "__main__":
    main()
