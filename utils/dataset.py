"""
utils/dataset.py
================
Defines the PyTorch Dataset class and builds DataLoaders for
train / val / test splits.

What is a Dataset?
-------------------
PyTorch needs a Dataset object that knows:
  1. How many images there are  → __len__()
  2. How to load image #N        → __getitem__(n)

What is a DataLoader?
---------------------
A DataLoader wraps a Dataset and automatically:
  - Shuffles images (training only)
  - Groups them into mini-batches (e.g., 16 at a time)
  - Loads them in parallel so training doesn't wait on disk I/O
"""

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


# ── Build transforms (image pre-processing pipelines) ────────────────────────

def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns two transform pipelines:
      train_transform — includes random augmentation to make the model
                        more robust (it sees slightly different versions
                        of each image every epoch)
      eval_transform  — deterministic: just resize + normalise.
                        Used for val, test, and inference.
    """
    train_transform = transforms.Compose([
        # 1. Resize every image to 224×224 pixels
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),

        # 2. Randomly flip left-right (50% chance) — a lesion looks the same
        #    whether it's on the left or right side of the frame
        transforms.RandomHorizontalFlip(p=config.AUG_HFLIP_PROB),

        # 3. Randomly rotate up to ±15 degrees — the camera angle varies
        transforms.RandomRotation(degrees=config.AUG_ROTATION),

        # 4. Randomly adjust brightness and contrast — lighting varies in
        #    real clinical photographs
        transforms.ColorJitter(
            brightness=config.AUG_BRIGHTNESS,
            contrast=config.AUG_CONTRAST,
        ),

        # 5. Convert PIL Image to a PyTorch tensor (values 0.0 – 1.0)
        transforms.ToTensor(),

        # 6. Normalise using ImageNet statistics — this is important because
        #    our ResNet weights were learned with these exact normalisation
        #    values.  Without this, the pre-trained features don't transfer.
        transforms.Normalize(
            mean=config.IMAGENET_MEAN,
            std=config.IMAGENET_STD,
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config.IMAGENET_MEAN,
            std=config.IMAGENET_STD,
        ),
    ])

    return train_transform, eval_transform


# ── Build DataLoaders ─────────────────────────────────────────────────────────

def get_dataloaders() -> Tuple[DataLoader, DataLoader, DataLoader, list]:
    """
    Creates and returns three DataLoaders (train, val, test) plus the
    class names list.

    torchvision.datasets.ImageFolder expects data laid out like:
        train/
            Normal/   ← folder name = class name
            Benign/
            OPMD/
            OSCC/
    It automatically assigns integer labels: 0, 1, 2, 3 (alphabetical).
    """
    train_tf, eval_tf = get_transforms()

    train_dataset = datasets.ImageFolder(
        root=str(config.TRAIN_DIR),
        transform=train_tf,
    )
    val_dataset = datasets.ImageFolder(
        root=str(config.VAL_DIR),
        transform=eval_tf,
    )
    test_dataset = datasets.ImageFolder(
        root=str(config.TEST_DIR),
        transform=eval_tf,
    )

    # num_workers=0 is safest on Windows + CPU
    # If you're on Linux/Mac you can set num_workers=2 for a small speed boost
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,       # shuffle so model doesn't memorise the order
        num_workers=0,
        pin_memory=False,   # pin_memory only helps with GPU
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    class_names = train_dataset.classes   # ['Benign', 'Normal', 'OPMD', 'OSCC']
    print(f"[✓] DataLoaders ready.")
    print(f"    Train: {len(train_dataset)} images | "
          f"Val: {len(val_dataset)} | "
          f"Test: {len(test_dataset)}")
    print(f"    Classes detected: {class_names}")

    return train_loader, val_loader, test_loader, class_names


def compute_class_weights(train_loader: DataLoader) -> torch.Tensor:
    """
    Computes inverse-frequency class weights.

    Why do we need this?
    The dataset is slightly imbalanced (more Normal images than OSCC).
    Without correction, the model learns to favour the majority class.
    We penalise mistakes on rare classes more heavily during training.

    Returns a 1-D tensor of shape [NUM_CLASSES].
    """
    counts = torch.zeros(config.NUM_CLASSES)
    for _, labels in train_loader:
        for label in labels:
            counts[label] += 1

    # Weight = total_images / (num_classes × class_count)
    total = counts.sum()
    weights = total / (config.NUM_CLASSES * counts)
    weights = weights / weights.sum()   # normalise so they sum to 1

    print(f"[✓] Class weights computed: "
          + ", ".join(
              f"{config.CLASS_NAMES[i]}={weights[i]:.3f}"
              for i in range(config.NUM_CLASSES)
          ))
    return weights
