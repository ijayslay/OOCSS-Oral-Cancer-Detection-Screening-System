"""
download_dataset.py
====================
Downloads the oral cancer image dataset from Kaggle and organises it
into the folder structure that the rest of OOCSS expects.

Expected final structure
------------------------
data/
  train/
    Normal/
    Benign/
    OPMD/
    OSCC/
  val/
    Normal/
    Benign/
    OPMD/
    OSCC/
  test/
    Normal/
    Benign/
    OPMD/
    OSCC/

How to set up Kaggle API (do this ONCE before running this script)
------------------------------------------------------------------
1. Go to https://www.kaggle.com  →  Your Account  →  Settings
2. Scroll to "API" and click "Create New Token"
3. A file called kaggle.json is downloaded — it looks like:
       {"username": "yourname", "key": "abcdef1234567890"}
4. Put that file in:
       Windows : C:\\Users\\<YourName>\\.kaggle\\kaggle.json
       Mac/Linux: ~/.kaggle/kaggle.json
5. Run this script.  That's it.
"""

import os
import shutil
import random
import zipfile
from pathlib import Path

# ── We use kaggle's Python API to pull the dataset ──────────────────────────
try:
    import kaggle
except ImportError:
    raise SystemExit(
        "\n[ERROR] kaggle package not found.\n"
        "Run:  pip install kaggle\n"
    )

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent          # project root  (oocss/)
DATA_DIR   = BASE_DIR / "data"
RAW_DIR    = DATA_DIR / "raw"               # where the zip lands
SPLITS     = ["train", "val", "test"]
CLASSES    = ["Normal", "Benign", "OPMD", "OSCC"]

# Train / val / test proportions
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

# Kaggle dataset slug  (owner/dataset-name)
# "zaidpy/oral-cancer-dataset" is a public dataset with ~3 000 labelled images
KAGGLE_DATASET = "zaidpy/oral-cancer-dataset"


# ── Helpers ──────────────────────────────────────────────────────────────────
def make_split_dirs():
    """Create train/val/test/<class> folders."""
    for split in SPLITS:
        for cls in CLASSES:
            (DATA_DIR / split / cls).mkdir(parents=True, exist_ok=True)
    print("[✓] Split directories created.")


def download_raw():
    """Download the Kaggle dataset zip into data/raw/."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[↓] Downloading dataset: {KAGGLE_DATASET} …")
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        KAGGLE_DATASET,
        path=str(RAW_DIR),
        unzip=True,
        quiet=False,
    )
    print("[✓] Download complete.")


def discover_images(raw_root: Path) -> dict[str, list[Path]]:
    """
    Walk raw_root and return {class_name: [image_paths]}.

    The Kaggle dataset uses folder names that may differ from our four
    class names.  We map them here — edit if your download looks different.
    """
    # Mapping: folder name on disk  →  our canonical class name
    FOLDER_MAP = {
        "normal"        : "Normal",
        "benign"        : "Benign",
        "opmd"          : "OPMD",
        "oscc"          : "OSCC",
        "malignant"     : "OSCC",      # some datasets use this label
        "pre-malignant" : "OPMD",
        "premalignant"  : "OPMD",
        "cancer"        : "OSCC",
    }

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    class_images: dict[str, list[Path]] = {c: [] for c in CLASSES}

    for folder in raw_root.rglob("*"):
        if not folder.is_dir():
            continue
        key = folder.name.lower().strip()
        mapped = FOLDER_MAP.get(key)
        if mapped is None:
            continue
        imgs = [
            p for p in folder.iterdir()
            if p.suffix.lower() in image_extensions
        ]
        class_images[mapped].extend(imgs)
        print(f"  Found {len(imgs):>4} images in '{folder.name}' → {mapped}")

    return class_images


def split_and_copy(class_images: dict[str, list[Path]]):
    """Randomly split each class and copy files into split folders."""
    random.seed(42)

    for cls, paths in class_images.items():
        if not paths:
            print(f"[!] WARNING: No images found for class '{cls}'. "
                  "Check folder mapping in discover_images().")
            continue

        random.shuffle(paths)
        n      = len(paths)
        n_tr   = int(n * SPLIT_RATIOS["train"])
        n_val  = int(n * SPLIT_RATIOS["val"])

        splits_assign = {
            "train": paths[:n_tr],
            "val"  : paths[n_tr : n_tr + n_val],
            "test" : paths[n_tr + n_val :],
        }

        for split, imgs in splits_assign.items():
            dest_dir = DATA_DIR / split / cls
            for img in imgs:
                shutil.copy2(img, dest_dir / img.name)
            print(f"  {cls}/{split}: {len(imgs)} images copied.")


def print_summary():
    """Print a table of image counts per split and class."""
    print("\n── Dataset Summary ──────────────────────────────────")
    header = f"{'Class':<12}" + "".join(f"{s:>8}" for s in SPLITS) + f"{'Total':>8}"
    print(header)
    print("─" * len(header))

    for cls in CLASSES:
        counts = [len(list((DATA_DIR / s / cls).glob("*"))) for s in SPLITS]
        total  = sum(counts)
        row    = f"{cls:<12}" + "".join(f"{c:>8}" for c in counts) + f"{total:>8}"
        print(row)

    print("─" * len(header))
    totals = [
        sum(len(list((DATA_DIR / s / c).glob("*"))) for c in CLASSES)
        for s in SPLITS
    ]
    print(f"{'TOTAL':<12}" + "".join(f"{t:>8}" for t in totals) +
          f"{sum(totals):>8}")
    print("────────────────────────────────────────────────────\n")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n╔══════════════════════════════════════════╗")
    print("║   OOCSS — Dataset Preparation Script     ║")
    print("╚══════════════════════════════════════════╝\n")

    make_split_dirs()
    download_raw()

    print("\n[→] Scanning downloaded files …")
    class_images = discover_images(RAW_DIR)

    total_found = sum(len(v) for v in class_images.values())
    if total_found == 0:
        raise SystemExit(
            "\n[ERROR] No images were discovered in the downloaded folder.\n"
            "Open data/raw/ and check the folder names.\n"
            "Then edit the FOLDER_MAP dictionary in discover_images().\n"
        )

    print(f"\n[→] Total images found: {total_found}")
    print("[→] Splitting and copying …\n")
    split_and_copy(class_images)
    print_summary()
    print("[✓] Dataset ready.  You can now run:  python train.py\n")
