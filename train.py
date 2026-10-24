"""
train.py
========
MAIN TRAINING SCRIPT — run this file to train the full OOCSS pipeline.

What happens when you run this:
  1.  Load the dataset from data/train, data/val, data/test
  2.  Build the ResNet-50 model
  3.  Train the CNN for up to NUM_EPOCHS epochs (with early stopping)
  4.  Evaluate on the test set → print metrics, save plots
  5.  Extract image features for the Random Forest training set
  6.  Train the Random Forest module on synthetic + image-derived features
  7.  Save both models to models/

How to run (in VS Code terminal):
  python train.py

Expected total time on CPU: 2–6 hours depending on dataset size.
Tip: Start with a small subset first to confirm everything runs.
"""

import sys
import csv
import time
import random
import numpy as np
import torch
from pathlib import Path

# ── Add project root to Python path ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

import config
from utils.dataset   import get_dataloaders, compute_class_weights
from utils.model     import build_resnet50, save_model
from utils.trainer   import fit
from utils.evaluate  import full_evaluation_report, plot_training_history
from utils.risk_model import (
    generate_synthetic_rf_data,
    train_rf,
    print_feature_importance,
    save_rf,
)


# ── Reproducibility ───────────────────────────────────────────────────────────
def set_seed(seed: int = config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║        OOCSS — Oral Cancer Screening System          ║")
    print("║              Training Pipeline  v1.0                 ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    set_seed()

    # ── Step 1: Load data ─────────────────────────────────────────────────
    print("─── Step 1/6 : Loading dataset ──────────────────────────")
    train_loader, val_loader, test_loader, class_names = get_dataloaders()
    class_weights = compute_class_weights(train_loader)

    # ── Step 2: Build model ───────────────────────────────────────────────
    print("\n─── Step 2/6 : Building ResNet-50 model ─────────────────")
    model = build_resnet50()

    # ── Step 3: Train CNN ─────────────────────────────────────────────────
    print("\n─── Step 3/6 : Training ResNet-50 ───────────────────────")
    t0 = time.time()
    history = fit(model, train_loader, val_loader, class_weights)
    elapsed = (time.time() - t0) / 60
    print(f"[✓] Training finished in {elapsed:.1f} minutes.")

    # ── Step 4: Save training history ────────────────────────────────────
    print("\n─── Step 4/6 : Saving history & plots ───────────────────")
    plot_training_history(history)

    # Write CSV so you can inspect numbers in Excel
    with open(config.HISTORY_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history.keys())
        writer.writeheader()
        rows = [
            {k: history[k][i] for k in history}
            for i in range(len(history["train_loss"]))
        ]
        writer.writerows(rows)
    print(f"[✓] History CSV saved → {config.HISTORY_FILE}")

    # ── Step 5: Evaluate on test set ──────────────────────────────────────
    print("\n─── Step 5/6 : Evaluating on test set ───────────────────")
    # Load the BEST weights (saved by the trainer when val loss improved)
    model.load_state_dict(
        torch.load(config.CNN_WEIGHTS_FILE, map_location=config.DEVICE)
    )
    metrics = full_evaluation_report(model, test_loader, class_names)

    # ── Step 6: Train Random Forest module ───────────────────────────────
    print("\n─── Step 6/6 : Training Random Forest module ────────────")
    print("[→] Generating synthetic clinical training data …")
    df_rf = generate_synthetic_rf_data(n_samples=4000)

    print("[→] Fitting Random Forest …")
    rf = train_rf(df_rf)
    print_feature_importance(rf)
    save_rf(rf)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║                   Training Complete                  ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  CNN weights  →  models/resnet50_oocss.pth           ║")
    print(f"║  RF  model    →  models/random_forest_oocss.joblib   ║")
    print(f"║  Plots        →  outputs/                            ║")
    print("║                                                      ║")
    print("║  Next step: python predict.py --image <path>         ║")
    print("╚══════════════════════════════════════════════════════╝\n")


if __name__ == "__main__":
    main()
