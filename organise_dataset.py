"""
organise_dataset.py
===================
Organises the downloaded oral cancer dataset into the correct
train / val / test folder structure for OOCSS.

This dataset has two classes: CANCER and NON CANCER.
We map them to our four-class structure like this:
  NON CANCER  →  Normal   (split 50/50 between Normal and Benign)
  CANCER      →  OSCC     (split 50/50 between OPMD and OSCC)

Run this once:
  poetry run python organise_dataset.py
"""

import shutil
import random
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

SPLITS       = ["train", "val", "test"]
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED         = 42

# All source folders to scan
SOURCE_ROOTS = [
    DATA_DIR / "raw" / "Oral Cancer" / "Oral Cancer Dataset",
    DATA_DIR / "raw" / "Oral cancer Dataset 2.0" / "OC Dataset kaggle new",
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def collect_images():
    """Collect all CANCER and NON CANCER images from both source roots."""
    cancer     = []
    non_cancer = []

    for root in SOURCE_ROOTS:
        for folder in root.iterdir():
            if not folder.is_dir():
                continue
            imgs = [p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
            name = folder.name.upper().strip()
            if "NON" in name:
                non_cancer.extend(imgs)
                print(f"  NON CANCER ← {folder}  ({len(imgs)} images)")
            elif "CANCER" in name:
                cancer.extend(imgs)
                print(f"  CANCER     ← {folder}  ({len(imgs)} images)")

    return cancer, non_cancer


def split_list(items):
    """Split a list into train/val/test."""
    random.shuffle(items)
    n     = len(items)
    n_tr  = int(n * SPLIT_RATIOS["train"])
    n_val = int(n * SPLIT_RATIOS["val"])
    return {
        "train": items[:n_tr],
        "val"  : items[n_tr : n_tr + n_val],
        "test" : items[n_tr + n_val:],
    }


def copy_images(split_dict, class_a, class_b):
    """
    Splits images 50/50 between class_a and class_b,
    then copies them into the correct split folders.
    """
    for split, imgs in split_dict.items():
        mid   = len(imgs) // 2
        half_a = imgs[:mid]
        half_b = imgs[mid:]

        for img in half_a:
            dest = DATA_DIR / split / class_a
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, dest / img.name)

        for img in half_b:
            dest = DATA_DIR / split / class_b
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, dest / img.name)

    total_a = sum(len(v)//2 for v in split_dict.values())
    total_b = sum(len(v) - len(v)//2 for v in split_dict.values())
    print(f"  → {class_a}: {total_a} images  |  {class_b}: {total_b} images")


def print_summary():
    print("\n── Final Dataset Summary ────────────────────────────────")
    classes = ["Normal", "Benign", "OPMD", "OSCC"]
    splits  = ["train", "val", "test"]
    header  = f"{'Class':<10}" + "".join(f"{s:>8}" for s in splits) + f"{'Total':>8}"
    print(header)
    print("─" * len(header))
    for cls in classes:
        counts = [len(list((DATA_DIR / s / cls).glob("*"))) for s in splits]
        total  = sum(counts)
        print(f"{cls:<10}" + "".join(f"{c:>8}" for c in counts) + f"{total:>8}")
    print("─" * len(header))
    totals = [
        sum(len(list((DATA_DIR / s / c).glob("*"))) for c in classes)
        for s in splits
    ]
    print(f"{'TOTAL':<10}" + "".join(f"{t:>8}" for t in totals) +
          f"{sum(totals):>8}")
    print()


if __name__ == "__main__":
    print("\n╔══════════════════════════════════════════╗")
    print("║   OOCSS — Dataset Organiser               ║")
    print("╚══════════════════════════════════════════╝\n")

    random.seed(SEED)

    print("[→] Scanning source folders …")
    cancer, non_cancer = collect_images()

    print(f"\n  Total CANCER images    : {len(cancer)}")
    print(f"  Total NON CANCER images: {len(non_cancer)}")

    if not cancer or not non_cancer:
        raise SystemExit("\n[ERROR] No images found. Check the SOURCE_ROOTS paths.")

    print("\n[→] Splitting and copying NON CANCER → Normal + Benign …")
    copy_images(split_list(non_cancer), "Normal", "Benign")

    print("[→] Splitting and copying CANCER → OPMD + OSCC …")
    copy_images(split_list(cancer), "OPMD", "OSCC")

    print_summary()
    print("[✓] Done! Now run:  poetry run python train.py\n")
