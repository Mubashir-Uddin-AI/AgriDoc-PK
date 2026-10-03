# -*- coding: utf-8 -*-
"""ONNX INT8 Quantization & Export for AgriDoc-PK.

Exports the trained MobileNetV3-Small to ONNX format and applies
post-training dynamic quantization for edge deployment (PRD Section 7.3).

Target: ~16 MB PyTorch -> ~4 MB quantized ONNX (75% reduction).

Usage:
    python models/quantize.py --weights models/weights/mobilenetv3_best.pth
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic

# Import ONNX Runtime Quantization Utilities
from onnxruntime.quantization import quantize_dynamic as onnx_quantize_dynamic
from onnxruntime.quantization import QuantType

import onnx
from onnx import shape_inference

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.classifier import CLASS_NAMES, initialize_model

logger = logging.getLogger(__name__)


def export_onnx(model: nn.Module, output_path: str, opset_version: int = 17):
    """Export PyTorch model to ONNX format.

    Args:
        model: Trained PyTorch model.
        output_path: Path to save the ONNX file.
        opset_version: ONNX opset version.
    """
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)

    # Use fixed batch size (no dynamic_axes) to avoid shape inference
    # conflicts during ONNX quantization. Batch=1 is appropriate for
    # single-image edge inference on mobile devices.
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    logger.info("ONNX model exported: %s (%.2f MB)", output_path, file_size)


def quantize_model(model: nn.Module) -> nn.Module:
    """Apply dynamic quantization to reduce model size.

    Quantizes Linear layers to INT8 for significant size reduction
    and faster CPU inference.

    Args:
        model: Trained PyTorch model.

    Returns:
        Quantized model.
    """
    quantized = quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8,
    )
    logger.info("Dynamic quantization applied (Linear -> INT8)")
    return quantized


def preprocess_onnx_for_quantization(input_path: str, output_path: str):
    """Pre-process ONNX model to fix shape metadata before quantization.

    Loads the model in-memory and runs shape inference with strict_mode=False
    to resolve any shape mismatches from the custom classifier head, then
    saves the cleaned model.

    Args:
        input_path: Path to the exported ONNX model.
        output_path: Path to save the pre-processed model.
    """
    model = onnx.load(input_path)

    # Run shape inference with strict_mode=False to tolerate shape differences
    # in the custom classifier head (576 -> 1024 Linear layer)
    inferred = shape_inference.infer_shapes(model, check_type=True, strict_mode=False)

    # Clear existing value_info and re-populate from inference to avoid stale shapes
    del inferred.graph.value_info[:]
    inferred = shape_inference.infer_shapes(inferred, check_type=True, strict_mode=False)

    onnx.save(inferred, output_path)
    logger.info("Pre-processed ONNX model saved: %s", output_path)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="AgriDoc-PK ONNX Export & Quantization")
    parser.add_argument("--weights", type=str, default="models/weights/mobilenetv3_best.pth",
                        help="Path to trained PyTorch weights")
    parser.add_argument("--output_dir", type=str, default="models/weights",
                        help="Directory to save exported models")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not weights_path.exists():
        logger.error("Weights not found: %s. Train the model first.", weights_path)
        return

    # Load trained model
    model = initialize_model(num_classes=len(CLASS_NAMES), pretrained=False)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    original_size = os.path.getsize(weights_path) / (1024 * 1024)
    logger.info("Original PyTorch model: %.2f MB", original_size)

    # Step 1: Export full-precision ONNX (fixed batch=1 for edge deployment)
    onnx_fp32_path = output_dir / "mobilenetv3_fp32.onnx"
    export_onnx(model, str(onnx_fp32_path))

    # Step 2: Dynamic quantization (PyTorch version)
    quantized_model = quantize_model(model)

    # Step 3: Save quantized PyTorch model
    quantized_pth = output_dir / "mobilenetv3_quantized.pth"
    torch.save(quantized_model.state_dict(), quantized_pth)
    quantized_size = os.path.getsize(quantized_pth) / (1024 * 1024)
    logger.info("Quantized PyTorch: %.2f MB (%.1f%% reduction)",
                quantized_size, (1 - quantized_size / original_size) * 100)

    # Step 4: Pre-process the ONNX model to fix shape metadata
    onnx_preprocessed_path = output_dir / "mobilenetv3_fp32_preprocessed.onnx"
    preprocess_onnx_for_quantization(str(onnx_fp32_path), str(onnx_preprocessed_path))

    # Step 5: Quantize the pre-processed ONNX model using ONNX Runtime
    onnx_quant_path = output_dir / "mobilenetv3_quantized.onnx"
    logger.info("Applying ONNX Runtime dynamic quantization...")
    onnx_quantize_dynamic(
        model_input=str(onnx_preprocessed_path),
        model_output=str(onnx_quant_path),
        weight_type=QuantType.QUInt8,
    )
    logger.info("Quantized ONNX model exported: %s", onnx_quant_path)

    # Summary
    print("\n" + "=" * 50)
    print("EXPORT SUMMARY")
    print("=" * 50)
    print(f"Original PyTorch:   {original_size:.2f} MB")
    print(f"FP32 ONNX:          {os.path.getsize(onnx_fp32_path) / (1024*1024):.2f} MB")
    print(f"Quantized PyTorch:  {quantized_size:.2f} MB")
    print(f"Quantized ONNX:     {os.path.getsize(onnx_quant_path) / (1024*1024):.2f} MB")
    print(f"Compression ratio:  {original_size / quantized_size:.1f}x")


if __name__ == "__main__":
    main()
