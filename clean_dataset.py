"""
clean_dataset.py
================
Scans all images in the dataset and removes any that are corrupted
or unreadable. Run this once before train.py.

Usage:
  poetry run python clean_dataset.py
"""

from pathlib import Path
from PIL import Image
from tqdm import tqdm

DATA_DIR = Path(__file__).parent / "data"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

all_images = [
    p for p in DATA_DIR.rglob("*")
    if p.suffix.lower() in IMAGE_EXTS
]

print(f"\n[→] Scanning {len(all_images)} images for corruption …\n")

removed = 0
for img_path in tqdm(all_images, unit="img"):
    try:
        with Image.open(img_path) as img:
            img.verify()          # checks file integrity without fully decoding
    except Exception:
        try:
            # Second attempt: try actually loading it
            with Image.open(img_path) as img:
                img.convert("RGB")
        except Exception as e:
            print(f"  [REMOVING] {img_path.name}  — {e}")
            img_path.unlink()     # delete the bad file
            removed += 1

print(f"\n[✓] Scan complete.")
print(f"    Removed {removed} corrupted file(s).")
print(f"    Clean images remaining: {len(all_images) - removed}")
print(f"\n[→] Now run:  poetry run python train.py\n")
