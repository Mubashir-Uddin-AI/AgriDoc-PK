# -*- coding: utf-8 -*-
"""Model Evaluation Script for AgriDoc-PK.

Generates confusion matrix, per-class precision/recall/F1, and
overall accuracy metrics on the test split.

Usage:
    python models/evaluate.py --data_dir data/processed --weights models/weights/mobilenetv3_best.pth
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.classifier import CLASS_NAMES, CropDiseaseClassifier
from models.train import CropDiseaseDataset, get_val_transforms

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate(args: argparse.Namespace):
    """Run full evaluation on the test set."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    weights_path = Path(args.weights)
    if not weights_path.exists():
        logger.error("Model weights not found: %s", weights_path)
        logger.error("Train the model first: python models/train.py")
        return

    classifier = CropDiseaseClassifier(weights_path=weights_path, device=str(device))
    model = classifier.model

    # Load test dataset
    test_dir = os.path.join(args.data_dir, "val")
    test_dataset = CropDiseaseDataset(test_dir, transform=get_val_transforms())

    if len(test_dataset) == 0:
        logger.error("No test samples found in %s", test_dir)
        return

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    # Collect predictions
    all_preds = []
    all_labels = []
    all_probs = []

    model.eval()
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, preds = outputs.max(1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Overall accuracy
    accuracy = (all_preds == all_labels).mean() * 100
    logger.info("Overall Test Accuracy: %.2f%%", accuracy)

    # Per-class metrics
    report = classification_report(
        all_labels, all_preds,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)
    print(report)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("CONFUSION MATRIX")
    print("-" * 70)

    # Header
    header = "Predicted ->  " + "  ".join(f"{name[:8]:>8}" for name in CLASS_NAMES)
    print(header)
    print("-" * len(header))
    for i, row in enumerate(cm):
        row_str = f"{CLASS_NAMES[i][:12]:<12} " + "  ".join(f"{v:>8d}" for v in row)
        print(row_str)

    # OOD statistics
    max_probs = all_probs.max(axis=1)
    ood_count = (max_probs < 0.65).sum()
    logger.info("\nOOD detections (confidence < 65%%): %d / %d (%.1f%%)",
                ood_count, len(max_probs), 100.0 * ood_count / len(max_probs))

    # Save report
    report_path = Path(args.weights).parent / "evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Overall Accuracy: {accuracy:.2f}%\n\n")
        f.write(report)
        f.write(f"\nOOD detections: {ood_count}/{len(max_probs)}\n")
    logger.info("Report saved to %s", report_path)


def main():
    parser = argparse.ArgumentParser(description="AgriDoc-PK Model Evaluation")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--weights", type=str, default="models/weights/mobilenetv3_best.pth")
    parser.add_argument("--batch_size", type=int, default=32)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
