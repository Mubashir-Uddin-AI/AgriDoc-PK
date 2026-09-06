# -*- coding: utf-8 -*-
"""MobileNetV3-Small Disease Classifier (Module 2).

Implements FR-2.1 through FR-2.3 from the AgriDoc-PK PRD:
  - MobileNetV3-Small backbone with 10-class crop disease head
  - ImageNet-normalized 224×224 tensor preprocessing
  - Softmax confidence with OOD guard (< 65% → reject)
  - Urdu class label mapping

The classifier loads trained weights from disk. If weights are not
yet available (pre-training), use `initialize_model()` to create
a fresh architecture ready for fine-tuning.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Class definitions (FR-2.1 — 10 classes)
# ---------------------------------------------------------------------------
CLASS_NAMES = [
    "cotton_bacterial_blight",
    "cotton_healthy",
    "cotton_leaf_curl_virus",
    "rice_blast",
    "rice_brown_spot",
    "rice_healthy",
    "wheat_healthy",
    "wheat_leaf_rust",
    "wheat_stripe_rust",
]
# NOTE: wheat_loose_smut was removed — not present in agridoc-pk-dataset.
# Dataset mappings: cotton_curl_virus→cotton_leaf_curl_virus,
#                   wheat_brown_rust→wheat_leaf_rust,
#                   wheat_yellow_rust→wheat_stripe_rust

CLASS_LABELS_UR = {
    "cotton_bacterial_blight": "کپاس کا جھلساؤ",
    "cotton_healthy": "صحت مند کپاس",
    "cotton_leaf_curl_virus": "مڑوریا وائرس (CLCuV)",
    "rice_blast": "دھان کا بلاسٹ / جھلساؤ",
    "rice_brown_spot": "چاول کے بھورے دھبے",
    "rice_healthy": "صحت مند چاول",
    "wheat_healthy": "صحت مند گندم",
    "wheat_leaf_rust": "پتوں کی بھوری زنگاری",
    "wheat_stripe_rust": "زرد زنگاری",
}

CROP_FROM_CLASS = {
    "cotton_bacterial_blight": ("Cotton", "کپاس"),
    "cotton_healthy": ("Cotton", "کپاس"),
    "cotton_leaf_curl_virus": ("Cotton", "کپاس"),
    "rice_blast": ("Rice", "چاول"),
    "rice_brown_spot": ("Rice", "چاول"),
    "rice_healthy": ("Rice", "چاول"),
    "wheat_healthy": ("Wheat", "گندم"),
    "wheat_leaf_rust": ("Wheat", "گندم"),
    "wheat_stripe_rust": ("Wheat", "گندم"),
}

PATHOGEN_MAP = {
    "wheat_leaf_rust": "Puccinia triticina (Fungal)",
    "wheat_stripe_rust": "Puccinia striiformis (Fungal)",
    "cotton_leaf_curl_virus": "Begomovirus via Bemisia tabaci (Viral)",
    "cotton_bacterial_blight": "Xanthomonas citri pv. malvacearum (Bacterial)",
    "rice_blast": "Magnaporthe oryzae (Fungal)",
    "rice_brown_spot": "Bipolaris oryzae (Fungal)",
}

# ImageNet normalization constants (PRD Section 6.2)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# OOD confidence threshold (FR-2.2)
OOD_THRESHOLD = 0.65

# Default model weights path
DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "models" / "weights" / "mobilenetv3_best.pth"

# Urdu rejection messages
MSG_OOD = "پتے کی بیماری کی تصدیق نہیں ہو سکی۔ براہ کرم صاف تصویر دوبارہ لیں۔"


def initialize_model(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """Create a MobileNetV3-Small model with a custom classification head.

    Args:
        num_classes: Number of output classes (default 10).
        pretrained: Whether to load ImageNet pretrained weights for
                    the backbone (recommended for fine-tuning).

    Returns:
        MobileNetV3-Small PyTorch model ready for training or inference.
    """
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)

    # Replace the classifier head for our 10 crop disease classes
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 1024),
        nn.Hardswish(),
        nn.Dropout(p=0.2),
        nn.Linear(1024, num_classes),
    )

    logger.info("Initialized MobileNetV3-Small with %d-class head (pretrained=%s)",
                num_classes, pretrained)
    return model


class CropDiseaseClassifier:
    """Production inference wrapper for the MobileNetV3 crop disease model.

    Handles preprocessing, inference, softmax confidence extraction,
    and OOD detection in a single predict() call.

    Args:
        weights_path: Path to the trained .pth checkpoint file.
        device: PyTorch device ('cpu' or 'cuda').
        ood_threshold: Minimum confidence to accept a prediction.
    """

    def __init__(
        self,
        weights_path: str | Path | None = None,
        device: str = "cpu",
        ood_threshold: float = OOD_THRESHOLD,
    ):
        self.device = torch.device(device)
        self.ood_threshold = ood_threshold
        self.model = initialize_model(num_classes=len(CLASS_NAMES), pretrained=False)

        weights_path = Path(weights_path) if weights_path else DEFAULT_WEIGHTS_PATH
        if weights_path.exists():
            state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            logger.info("Loaded model weights from %s", weights_path)
        else:
            logger.warning(
                "Model weights not found at %s. Classifier will use uninitialized "
                "weights — train the model first using models/train.py.",
                weights_path,
            )

        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, img_bgr: np.ndarray) -> torch.Tensor:
        """Preprocess a BGR image for MobileNetV3 inference.

        Pipeline (PRD Section 6.2):
          1. Resize to 224×224
          2. Convert BGR → RGB
          3. Scale to [0, 1]
          4. Normalize with ImageNet mean/std
          5. Reshape to (1, 3, 224, 224)

        Args:
            img_bgr: Input BGR image (any size).

        Returns:
            Preprocessed tensor ready for inference.
        """
        # Resize to 224×224
        resized = cv2.resize(img_bgr, (224, 224), interpolation=cv2.INTER_AREA)
        # BGR → RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # To float [0, 1]
        tensor = torch.from_numpy(rgb).float() / 255.0
        # HWC → CHW
        tensor = tensor.permute(2, 0, 1)
        # Normalize with ImageNet stats
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        tensor = (tensor - mean) / std
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        return tensor.to(self.device)

    @torch.no_grad()
    def predict(self, img_bgr: np.ndarray) -> Dict:
        """Run inference on a leaf image and return structured diagnosis.

        Args:
            img_bgr: BGR NumPy image of the leaf.

        Returns:
            Dictionary with keys:
              - disease_id (str): Predicted class identifier
              - disease_name_ur (str): Urdu disease label
              - crop (str): English crop name
              - crop_ur (str): Urdu crop name
              - confidence (float): Softmax probability (0.0-1.0)
              - pathogen (str): Pathogen scientific name
              - is_ood (bool): True if below OOD threshold
              - ood_message_ur (str | None): Urdu message if OOD
        """
        tensor = self.preprocess(img_bgr)
        logits = self.model(tensor)
        probabilities = F.softmax(logits, dim=1).squeeze(0)

        confidence, predicted_idx = torch.max(probabilities, dim=0)
        confidence = float(confidence)
        predicted_idx = int(predicted_idx)

        disease_id = CLASS_NAMES[predicted_idx]
        crop_en, crop_ur = CROP_FROM_CLASS[disease_id]

        is_ood = confidence < self.ood_threshold

        result = {
            "disease_id": disease_id,
            "disease_name": disease_id.replace("_", " ").title(),
            "disease_name_ur": CLASS_LABELS_UR[disease_id],
            "crop": crop_en,
            "crop_ur": crop_ur,
            "confidence": round(confidence, 4),
            "pathogen": PATHOGEN_MAP.get(disease_id, "N/A"),
            "is_ood": is_ood,
            "ood_message_ur": MSG_OOD if is_ood else None,
        }

        logger.info(
            "Prediction: %s (%.1f%%) — OOD=%s",
            disease_id, confidence * 100, is_ood,
        )
        return result
