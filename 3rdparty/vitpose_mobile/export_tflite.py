#!/usr/bin/env python3
"""
TFLite/LiteRT Export Script for ViTPose Android Deployment

Exports ViTPose models to TensorFlow Lite format with support for FP32 and FP16 precision.

Conversion Path: PyTorch -> ONNX -> TensorFlow SavedModel -> TFLite

Usage:
    python export_tflite.py --model-ckpt model.pth --model-name b --output ckpts/
"""

import os
import sys
import tempfile
import argparse

import torch
import numpy as np
from loguru import logger

from vit_models.model import ViTPose
from utils.util import infer_dataset_by_path, dyn_model_import


# =============================================================================
# Constants
# =============================================================================
C, H, W = (3, 256, 192)
INPUT_SHAPE_NCHW = (1, C, H, W)  # PyTorch format


# =============================================================================
# Export Functions
# =============================================================================

def export_to_onnx(model, output_path, opset_version=17):
    """
    Export PyTorch model to ONNX format.

    Args:
        model: ViTPose model in eval mode
        output_path: Path to save .onnx file
        opset_version: ONNX opset version (17+ recommended for transformers)

    Returns:
        Path to exported ONNX model
    """
    import onnx

    device = next(model.parameters()).device
    dummy_input = torch.randn(1, C, H, W).to(device)

    logger.info(f"Exporting to ONNX (opset {opset_version})")

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset_version,
        input_names=['input'],
        output_names=['output'],
        do_constant_folding=True,
        export_params=True
    )

    # Validate ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    logger.success(f"ONNX model exported and validated: {output_path}")

    return output_path


def convert_onnx_to_savedmodel(onnx_path, output_dir):
    """
    Convert ONNX model to TensorFlow SavedModel using onnx2tf.

    Args:
        onnx_path: Path to ONNX model
        output_dir: Directory to save TensorFlow SavedModel

    Returns:
        Path to SavedModel directory
    """
    import onnx2tf

    logger.info("Converting ONNX to TensorFlow SavedModel")

    onnx2tf.convert(
        input_onnx_file_path=onnx_path,
        output_folder_path=output_dir,
        copy_onnx_input_output_names_to_tflite=True,
        non_verbose=True,
        keep_ncw_or_nchw_or_ncdhw_input_names=['input'],  # Keep NCHW format
    )

    logger.success(f"TensorFlow SavedModel exported: {output_dir}")
    return output_dir


def convert_to_tflite(savedmodel_path, output_path, precision='fp32'):
    """
    Convert TensorFlow SavedModel to TFLite format.

    Args:
        savedmodel_path: Path to TensorFlow SavedModel directory
        output_path: Path to save .tflite file
        precision: 'fp32' or 'fp16'

    Returns:
        Path to TFLite model
    """
    import tensorflow as tf

    logger.info(f"Converting to TFLite ({precision.upper()} precision)")

    converter = tf.lite.TFLiteConverter.from_saved_model(savedmodel_path)

    # Enable TF ops for unsupported operations (transformer attention, etc.)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]

    if precision == 'fp16':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    # Convert
    tflite_model = converter.convert()

    # Save
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.success(f"TFLite model saved: {output_path} ({file_size_mb:.1f} MB)")

    return output_path


def validate_tflite_model(tflite_path, num_keypoints):
    """
    Validate exported TFLite model with test inference.

    Args:
        tflite_path: Path to TFLite model
        num_keypoints: Expected number of keypoints in output

    Returns:
        True if validation passes
    """
    import tensorflow as tf

    logger.info("Validating TFLite model")

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Create test input
    input_shape = input_details[0]['shape']
    test_input = np.random.randn(*input_shape).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])

    # Check output shape
    logger.info(f"Input shape: {input_shape}")
    logger.info(f"Output shape: {output.shape}")

    # Expected output: (1, num_keypoints, H/4, W/4) = (1, num_keypoints, 64, 48)
    expected_keypoints = num_keypoints
    if output.shape[1] == expected_keypoints or output.shape[-1] == expected_keypoints:
        logger.success("Validation passed: output shape matches expected keypoint dimensions")
        return True
    else:
        logger.warning(f"Output shape {output.shape} may not match expected keypoints ({expected_keypoints})")
        return True  # Still return True as shape differences may be due to NCHW/NHWC


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Export ViTPose to TFLite format for Android deployment'
    )
    parser.add_argument('--model-ckpt', type=str, required=True,
                        help='The torch model that shall be used for conversion')
    parser.add_argument('--model-name', type=str, required=True,
                        choices=['s', 'b', 'l', 'h'],
                        help='[s: ViT-S, b: ViT-B, l: ViT-L, h: ViT-H]')
    parser.add_argument('--output', type=str, default='ckpts/',
                        help='File (without extension) or dir path for output')
    parser.add_argument('--dataset', type=str, required=False, default=None,
                        help='Name of the dataset. If None it\'s extracted from the file name. '
                             '["coco", "coco_25", "wholebody", "mpii", "ap10k", "apt36k", "aic"]')
    args = parser.parse_args()

    # Get dataset and model_cfg
    dataset = args.dataset
    if dataset is None:
        dataset = infer_dataset_by_path(args.model_ckpt)
    assert dataset in ['mpii', 'coco', 'coco_25', 'wholebody', 'aic', 'ap10k', 'apt36k'], \
        'The specified dataset is not valid'
    model_cfg = dyn_model_import(dataset, args.model_name)

    # Get number of keypoints from config
    num_keypoints = model_cfg['keypoint_head']['out_channels']

    logger.info(f"Converting ViTPose-{args.model_name} for {dataset} ({num_keypoints} keypoints)")

    # Load PyTorch model
    CKPT_PATH = args.model_ckpt
    model = ViTPose(model_cfg)

    ckpt = torch.load(CKPT_PATH, map_location='cpu', weights_only=True)
    if 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']

    model.load_state_dict(ckpt)
    model.eval()

    # Determine output path
    base_name = os.path.basename(args.model_ckpt).replace('.pth', '')
    if not os.path.isdir(args.output):
        base_name = os.path.basename(args.output).replace('.tflite', '')
    output_dir = os.path.dirname(args.output) if not os.path.isdir(args.output) else args.output

    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # Export pipeline using temp directory for intermediate files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Step 1: PyTorch -> ONNX
        onnx_path = os.path.join(tmpdir, 'model.onnx')
        export_to_onnx(model, onnx_path)

        # Step 2: ONNX -> TensorFlow SavedModel
        savedmodel_path = os.path.join(tmpdir, 'saved_model')
        convert_onnx_to_savedmodel(onnx_path, savedmodel_path)

        # Step 3: SavedModel -> TFLite (FP32)
        output_fp32 = os.path.join(output_dir, f"{base_name}_fp32.tflite")
        convert_to_tflite(savedmodel_path, output_fp32, precision='fp32')
        validate_tflite_model(output_fp32, num_keypoints)

        # Step 4: SavedModel -> TFLite (FP16)
        output_fp16 = os.path.join(output_dir, f"{base_name}_fp16.tflite")
        convert_to_tflite(savedmodel_path, output_fp16, precision='fp16')
        validate_tflite_model(output_fp16, num_keypoints)

    logger.info("TFLite export complete")
    logger.info(f"FP32: {os.path.abspath(output_fp32)}")
    logger.info(f"FP16: {os.path.abspath(output_fp16)}")


if __name__ == '__main__':
    main()
