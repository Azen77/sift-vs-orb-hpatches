"""
Downloads and extracts the HPatches "sequences" dataset (full images + ground-truth
homographies), which is what we need for wide-baseline matching evaluation.

Usage:
    python download_data.py

This will populate ./data/hpatches-sequences-release/ with one folder per sequence,
each containing:
    1.ppm ... 6.ppm   -- the images (1 = reference)
    H_1_2 ... H_1_6    -- 3x3 homographies mapping image 1 -> image N
"""

import os
import zipfile
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
# Official source, linked from https://github.com/hpatches/hpatches-dataset
URL = "https://huggingface.co/datasets/vbalnt/hpatches/resolve/main/hpatches-sequences-release.zip"
ARCHIVE_PATH = os.path.join(DATA_DIR, "hpatches-sequences-release.zip")
EXTRACT_DIR = os.path.join(DATA_DIR, "hpatches-sequences-release")


def download():
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(EXTRACT_DIR):
        print(f"Already extracted at {EXTRACT_DIR}, skipping.")
        return

    if not os.path.exists(ARCHIVE_PATH):
        print(f"Downloading HPatches sequences from {URL} ...")
        try:
            urllib.request.urlretrieve(URL, ARCHIVE_PATH)
        except Exception as e:
            print(f"\nDownload failed: {e}")
            print(
                "\nIf this link is down, grab the file manually from the official "
                "repo's 'Full image sequences' link instead:\n"
                "  https://github.com/hpatches/hpatches-dataset\n"
                f"and save it as:\n  {ARCHIVE_PATH}\n"
                "then re-run this script (it will pick up the local file and extract it)."
            )
            return

    print("Extracting (this is ~1.3GB, may take a minute)...")
    with zipfile.ZipFile(ARCHIVE_PATH, "r") as zf:
        zf.extractall(DATA_DIR)
    print(f"Done. Sequences extracted to {EXTRACT_DIR}")


if __name__ == "__main__":
    download()
