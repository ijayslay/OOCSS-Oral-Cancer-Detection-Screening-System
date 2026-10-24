"""
test_setup.py
=============
Run this FIRST before anything else.
It checks that all packages are installed and the model can do a
forward pass — without needing the dataset.

Usage:
  python test_setup.py

All checks should print [PASS].  Fix any [FAIL] before running train.py.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

PASS = "[PASS]"
FAIL = "[FAIL]"
all_ok = True


def check(label: str, fn):
    global all_ok
    try:
        result = fn()
        print(f"  {PASS}  {label}" + (f"  →  {result}" if result else ""))
    except Exception as e:
        print(f"  {FAIL}  {label}  →  {e}")
        all_ok = False


print("\n╔══════════════════════════════════════════════════════╗")
print("║          OOCSS — Environment Setup Check             ║")
print("╚══════════════════════════════════════════════════════╝\n")

# ── Package imports ────────────────────────────────────────────────────────────
print("── Checking package imports ──────────────────────────────")
check("torch",        lambda: __import__("torch").__version__)
check("torchvision",  lambda: __import__("torchvision").__version__)
check("sklearn",      lambda: __import__("sklearn").__version__)
check("numpy",        lambda: __import__("numpy").__version__)
check("pandas",       lambda: __import__("pandas").__version__)
check("matplotlib",   lambda: __import__("matplotlib").__version__)
check("seaborn",      lambda: __import__("seaborn").__version__)
check("PIL",          lambda: __import__("PIL").__version__)
check("tqdm",         lambda: __import__("tqdm").__version__)
check("joblib",       lambda: __import__("joblib").__version__)
check("cv2",          lambda: __import__("cv2").__version__)

# ── Config ─────────────────────────────────────────────────────────────────────
print("\n── Checking config ───────────────────────────────────────")
import config
check("config imports",  lambda: "OK")
check("CLASS_NAMES",     lambda: config.CLASS_NAMES)
check("DEVICE",          lambda: config.DEVICE)
check("MODEL_DIR exists",lambda: str(config.MODEL_DIR))

# ── ResNet-50 forward pass ─────────────────────────────────────────────────────
print("\n── Checking ResNet-50 model ──────────────────────────────")
import torch

def _test_model():
    from utils.model import build_resnet50
    model = build_resnet50()
    dummy = torch.randn(1, 3, 224, 224)      # fake image batch of 1
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (1, 4), f"Expected (1,4), got {out.shape}"
    return f"Output shape: {tuple(out.shape)}  ✓"

check("ResNet-50 forward pass", _test_model)

# ── Random Forest ──────────────────────────────────────────────────────────────
print("\n── Checking Random Forest module ─────────────────────────")

def _test_rf():
    from utils.risk_model import generate_synthetic_rf_data, train_rf
    df = generate_synthetic_rf_data(n_samples=200)
    rf = train_rf(df)
    return f"RF trained on {len(df)} synthetic samples  ✓"

check("Random Forest train", _test_rf)

# ── Dataset folders ────────────────────────────────────────────────────────────
print("\n── Checking data directories ─────────────────────────────")
import config

def _check_data_dir(split, cls):
    d = config.DATA_DIR / split / cls
    if d.exists():
        n = len(list(d.glob("*")))
        return f"{n} files" if n > 0 else "EXISTS but empty (run download_dataset.py)"
    return f"NOT FOUND (run download_dataset.py)"

for split in ["train", "val", "test"]:
    for cls in config.CLASS_NAMES:
        check(f"data/{split}/{cls}", lambda s=split, c=cls: _check_data_dir(s, c))

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n══════════════════════════════════════════════════════════")
if all_ok:
    print("  All checks passed ✓")
    print("  Next: python download_dataset.py   (set up Kaggle API first)")
    print("  Then: python train.py")
else:
    print("  Some checks FAILED.  Fix the errors above before continuing.")
    print("  Tip: pip install -r requirements.txt")
print("══════════════════════════════════════════════════════════\n")
