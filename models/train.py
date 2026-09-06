# -*- coding: utf-8 -*-
"""MobileNetV3-Small Training Script for AgriDoc-PK.

Fine-tunes a MobileNetV3-Small backbone on the agridoc-pk-dataset
with Albumentations augmentation pipeline designed for Pakistani
field conditions (PRD Section 7.2).

Usage:
    python models/train.py --data_dir data/processed --epochs 25 --batch_size 32

Expects data_dir structured as:
    data/processed/
    ├── train/
    │   ├── wheat_leaf_rust/
    │   ├── cotton_healthy/
    │   └── ...
    ├── val/
    └── test/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from sklearn.utils.class_weight import compute_class_weight

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.classifier import CLASS_NAMES, initialize_model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Albumentations Augmentation Pipeline (PRD Section 7.2)
# ---------------------------------------------------------------------------
def get_train_transforms() -> A.Compose:
    """Pakistani field-condition augmentation pipeline."""
    return A.Compose([
        A.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        # Simulate harsh Pakistani field sunlight & deep shadows
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3, hue=0.1, p=0.5),
        # Simulate camera defocus and moving hands
        A.OneOf([
            A.MotionBlur(blur_limit=5, p=0.5),
            A.GaussianBlur(blur_limit=5, p=0.5),
        ], p=0.3),
        # Simulate field dust & sensor noise (std_range is 0-1 scale in v2.0)
        A.GaussNoise(std_range=(0.02, 0.1), p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms() -> A.Compose:
    """Validation/test transforms — resize + normalize only."""
    return A.Compose([
        A.Resize(height=224, width=224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class CropDiseaseDataset(Dataset):
    """PyTorch dataset for crop disease images organized in class folders."""

    def __init__(self, root_dir: str, transform: A.Compose | None = None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples: list[tuple[str, int]] = []
        self.class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}

        for class_name in CLASS_NAMES:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                logger.warning("Class directory not found: %s", class_dir)
                continue
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))

        logger.info("Loaded %d samples from %s", len(self.samples), root_dir)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, label


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (batch_idx + 1) % 20 == 0:
            logger.info(
                "Epoch %d [%d/%d] Loss: %.4f Acc: %.2f%%",
                epoch, batch_idx + 1, len(loader),
                loss.item(), 100.0 * correct / total,
            )

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Validate the model. Returns (avg_loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def train(args: argparse.Namespace):
    """Main training routine."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)

    # Datasets
    train_dataset = CropDiseaseDataset(
        os.path.join(args.data_dir, "train"),
        transform=get_train_transforms(),
    )
    val_dataset = CropDiseaseDataset(
        os.path.join(args.data_dir, "val"),
        transform=get_val_transforms(),
    )

    if len(train_dataset) == 0:
        logger.error("No training samples found in %s/train/", args.data_dir)
        logger.error("Expected subdirectories: %s", ", ".join(CLASS_NAMES))
        return

    # Class weights for imbalanced data (PRD Section 7.1)
    all_labels = [label for _, label in train_dataset.samples]
    class_weights = compute_class_weight(
        "balanced", classes=np.arange(len(CLASS_NAMES)), y=np.array(all_labels)
    )
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    logger.info("Class weights: %s", dict(zip(CLASS_NAMES, class_weights.round(3))))

    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Model
    model = initialize_model(num_classes=len(CLASS_NAMES), pretrained=True)
    model = model.to(device)

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_val_acc = 0.0
    weights_dir = Path(__file__).resolve().parent / "weights"
    weights_dir.mkdir(exist_ok=True)
    best_path = weights_dir / "mobilenetv3_best.pth"

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        logger.info(
            "Epoch %d/%d (%.1fs) — Train: loss=%.4f acc=%.2f%% | Val: loss=%.4f acc=%.2f%%",
            epoch, args.epochs, elapsed, train_loss, train_acc, val_loss, val_acc,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)
            logger.info("New best model saved: %.2f%% -> %s", val_acc, best_path)

    # Save training history
    history_path = weights_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    logger.info("Training complete. Best val accuracy: %.2f%%", best_val_acc)
    logger.info("Best model: %s", best_path)
    logger.info("History: %s", history_path)


def main():
    parser = argparse.ArgumentParser(description="AgriDoc-PK MobileNetV3 Training")
    parser.add_argument("--data_dir", type=str, default="data/processed",
                        help="Root directory with train/val/test splits")
    parser.add_argument("--epochs", type=int, default=25,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Initial learning rate")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="DataLoader worker processes")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
