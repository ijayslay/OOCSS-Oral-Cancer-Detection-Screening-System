"""
utils/model.py
==============
Defines the OOCSS CNN — a ResNet-50 with a custom classification head,
fine-tuned for 4-class oral cancer classification.

What is ResNet-50?
------------------
ResNet-50 is a deep neural network with 50 layers, originally trained on
1.2 million images across 1,000 categories (ImageNet competition).
Because it has already learned powerful general visual features
(edges, textures, shapes), we don't train from scratch.  We:
  1. Load the pre-trained weights  (download ~100 MB once)
  2. Freeze the early layers        (keep their learned features)
  3. Replace the final layer        (swap 1,000-class → 4-class output)
  4. Fine-tune the last few layers  (adapt to oral cavity images)

This is called Transfer Learning and it is the standard approach
when your domain-specific dataset is smaller than ImageNet.
"""

from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


# ── Model builder ─────────────────────────────────────────────────────────────

def build_resnet50() -> nn.Module:
    """
    Loads a pre-trained ResNet-50, freezes early layers, and replaces
    the classifier head with a 4-class output.

    Returns the model (on CPU — as configured).
    """
    # ── Step 1: Load ResNet-50 with ImageNet weights ──────────────────────
    # weights=DEFAULT downloads the best available pre-trained weights.
    # First run: downloads ~100 MB.  Subsequent runs: uses cache.
    print("[→] Loading ResNet-50 with ImageNet pre-trained weights …")
    weights = models.ResNet50_Weights.DEFAULT
    model   = models.resnet50(weights=weights)

    # ── Step 2: Freeze layers we don't want to update ─────────────────────
    # ResNet-50's parameters are listed in order from input → output.
    # We freeze the first FREEZE_LAYERS parameters so gradient updates
    # don't overwrite the low-level feature detectors.
    all_params = list(model.parameters())
    for i, param in enumerate(all_params):
        if i < config.FREEZE_LAYERS:
            param.requires_grad = False    # frozen — no gradient
        else:
            param.requires_grad = True     # trainable

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"    Trainable parameters : {trainable:,}")
    print(f"    Frozen parameters    : {frozen:,}")

    # ── Step 3: Replace the classification head ────────────────────────────
    # ResNet-50's original head: model.fc  → Linear(2048, 1000)
    # We replace it with our own 4-class head.
    in_features = model.fc.in_features    # = 2048 for ResNet-50

    model.fc = nn.Sequential(
        # Dense layer: 2048 → 512 units with ReLU activation
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),

        # Dropout: randomly zeroes 40% of neurons during training.
        # This prevents overfitting (memorising training images).
        nn.Dropout(p=config.DROPOUT_RATE),

        # Final output: 512 → 4  (one score per class)
        # We do NOT apply Softmax here because PyTorch's CrossEntropyLoss
        # includes Softmax internally.  Adding it twice would be wrong.
        nn.Linear(512, config.NUM_CLASSES),
    )

    # ── Step 4: Move model to the configured device ────────────────────────
    model = model.to(config.DEVICE)
    print(f"[✓] ResNet-50 model built and moved to: {config.DEVICE}")
    return model


# ── Save / load helpers ────────────────────────────────────────────────────────

def save_model(model: nn.Module, path: Path = config.CNN_WEIGHTS_FILE):
    """Saves only the model weights (state_dict), not the full object."""
    torch.save(model.state_dict(), path)
    print(f"[✓] Model weights saved → {path}")


def load_model(path: Path = config.CNN_WEIGHTS_FILE) -> nn.Module:
    """
    Rebuilds the model architecture and loads saved weights into it.
    Call this at inference time.
    """
    model = build_resnet50()
    model.load_state_dict(
        torch.load(path, map_location=config.DEVICE)
    )
    model.eval()    # switch to evaluation mode (disables Dropout)
    print(f"[✓] Weights loaded from → {path}")
    return model
