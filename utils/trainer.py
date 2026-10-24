"""
utils/trainer.py
================
Contains the core training and evaluation loops.

Every epoch of training does two things:
  1. TRAIN LOOP  — forward pass → compute loss → backprop → update weights
  2. EVAL LOOP   — forward pass only → measure accuracy (no weight updates)

Key concepts explained inline.
"""

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm   # progress bar — shows a nice [=====>  ] bar in terminal

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


# ── Single epoch: train ────────────────────────────────────────────────────────

def train_one_epoch(
    model      : nn.Module,
    loader     : DataLoader,
    criterion  : nn.Module,     # loss function
    optimizer  : torch.optim.Optimizer,
) -> Tuple[float, float]:
    """
    Runs ONE full pass through the training set.
    Returns (average_loss, accuracy_percent).
    """
    model.train()   # tells BatchNorm and Dropout to use training behaviour

    running_loss    = 0.0
    correct         = 0
    total           = 0

    # tqdm wraps the loader and prints a progress bar automatically
    pbar = tqdm(loader, desc="  Train", leave=False, unit="batch")

    for images, labels in pbar:
        # Move data to the same device as the model (CPU in our case)
        images = images.to(config.DEVICE)
        labels = labels.to(config.DEVICE)

        # ── Forward pass ──────────────────────────────────────────────────
        # The model takes a batch of images and produces raw scores (logits)
        # for each class.  Shape: [batch_size, 4]
        optimizer.zero_grad()   # clear gradients from the previous batch
        logits = model(images)

        # ── Compute loss ──────────────────────────────────────────────────
        # CrossEntropyLoss measures how wrong the predictions are.
        # It applies Softmax internally, then computes log loss.
        loss = criterion(logits, labels)

        # ── Backward pass ─────────────────────────────────────────────────
        # PyTorch computes how much each weight contributed to the loss
        loss.backward()

        # ── Weight update ─────────────────────────────────────────────────
        # Adam adjusts each weight in the direction that reduces loss
        optimizer.step()

        # ── Bookkeeping ───────────────────────────────────────────────────
        running_loss += loss.item() * images.size(0)

        # The predicted class is the one with the highest score
        preds    = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# ── Single epoch: evaluate ─────────────────────────────────────────────────────

@torch.no_grad()    # decorator: disables gradient computation → faster + less RAM
def evaluate(
    model     : nn.Module,
    loader    : DataLoader,
    criterion : nn.Module,
    desc      : str = "Val",
) -> Tuple[float, float]:
    """
    Evaluates the model on a DataLoader (val or test).
    No weight updates happen here.
    Returns (average_loss, accuracy_percent).
    """
    model.eval()    # tells Dropout to pass all neurons (no zeroing)

    running_loss = 0.0
    correct      = 0
    total        = 0

    pbar = tqdm(loader, desc=f"  {desc}", leave=False, unit="batch")

    for images, labels in pbar:
        images = images.to(config.DEVICE)
        labels = labels.to(config.DEVICE)

        logits = model(images)
        loss   = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        preds    = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# ── Full training loop ─────────────────────────────────────────────────────────

def fit(
    model      : nn.Module,
    train_loader: DataLoader,
    val_loader  : DataLoader,
    class_weights: torch.Tensor,
) -> dict:
    """
    Runs the complete training loop for NUM_EPOCHS epochs, with:
      - Learning rate reduction on plateau
      - Early stopping
      - Saving the best model weights (lowest validation loss)

    Returns a history dict with lists of losses and accuracies per epoch.
    """
    # ── Loss function ──────────────────────────────────────────────────────
    # weight= penalises the model more for misclassifying rare classes
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(config.DEVICE)
    )

    # ── Optimiser ──────────────────────────────────────────────────────────
    # Adam is the go-to optimiser.  It adapts the learning rate per weight.
    # We only pass the TRAINABLE parameters (frozen ones would be ignored
    # by requires_grad anyway, but being explicit is good practice).
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        trainable_params,
        lr=config.LEARNING_RATE,
    )

    # ── Learning rate scheduler ────────────────────────────────────────────
    # If validation loss doesn't improve for LR_PATIENCE epochs,
    # the learning rate is halved.  This helps escape plateaus.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode     = "min",
        factor   = 0.5,
        patience = config.LR_PATIENCE,
        verbose  = True,
    )

    # ── History tracking ───────────────────────────────────────────────────
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss"  : [], "val_acc"  : [],
    }

    best_val_loss   = float("inf")
    epochs_no_impv  = 0             # counter for early stopping

    print(f"\n{'═'*60}")
    print(f"  Starting training for up to {config.NUM_EPOCHS} epochs")
    print(f"  Device: {config.DEVICE}  |  Batch size: {config.BATCH_SIZE}")
    print(f"{'═'*60}\n")

    for epoch in range(1, config.NUM_EPOCHS + 1):
        print(f"Epoch {epoch:>3}/{config.NUM_EPOCHS}")

        # ── Train ──────────────────────────────────────────────────────────
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer
        )

        # ── Validate ───────────────────────────────────────────────────────
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, desc="Val")

        # ── Scheduler step ────────────────────────────────────────────────
        scheduler.step(vl_loss)

        # ── Save history ───────────────────────────────────────────────────
        history["train_loss"].append(tr_loss)
        history["train_acc" ].append(tr_acc)
        history["val_loss"  ].append(vl_loss)
        history["val_acc"   ].append(vl_acc)

        print(f"  Train → loss: {tr_loss:.4f}  acc: {tr_acc:.1f}%")
        print(f"  Val   → loss: {vl_loss:.4f}  acc: {vl_acc:.1f}%")

        # ── Save best model ────────────────────────────────────────────────
        if vl_loss < best_val_loss:
            best_val_loss  = vl_loss
            epochs_no_impv = 0
            torch.save(model.state_dict(), config.CNN_WEIGHTS_FILE)
            print(f"  ★ New best val loss ({best_val_loss:.4f}) — weights saved.")
        else:
            epochs_no_impv += 1
            print(f"  No improvement for {epochs_no_impv} epoch(s).")

        # ── Early stopping ─────────────────────────────────────────────────
        if epochs_no_impv >= config.EARLY_STOP_PAT:
            print(f"\n[→] Early stopping triggered at epoch {epoch}.")
            break

        print()

    print(f"\n{'═'*60}")
    print(f"  Training complete.  Best val loss: {best_val_loss:.4f}")
    print(f"  Best weights saved to: {config.CNN_WEIGHTS_FILE}")
    print(f"{'═'*60}\n")

    return history
