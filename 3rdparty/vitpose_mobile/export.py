import os
import sys
import torch
import argparse
import coremltools as ct
from loguru import logger

from vit_models.model import ViTPose
from utils.util import infer_dataset_by_path, dyn_model_import


parser = argparse.ArgumentParser()
parser.add_argument('--model-ckpt', type=str, required=True,
                    help='The torch model that shall be used for conversion')
parser.add_argument('--model-name', type=str, required=True, choices=['s', 'b', 'l', 'h'],
                    help='[s: ViT-S, b: ViT-B, l: ViT-L, h: ViT-H]')
parser.add_argument('--output', type=str, default='ckpts/',
                    help='File (without extension) or dir path for checkpoint output')
parser.add_argument('--dataset', type=str, required=False, default=None,
                    help='Name of the dataset. If None it"s extracted from the file name. \
                          ["coco", "coco_25", "wholebody", "mpii", "ap10k", "apt36k", "aic"]')
args = parser.parse_args()


# Get dataset and model_cfg
dataset = args.dataset
if dataset is None:
    dataset = infer_dataset_by_path(args.model_ckpt)
assert dataset in ['mpii', 'coco', 'coco_25', 'wholebody', 'aic', 'ap10k', 'apt36k'], \
    'The specified dataset is not valid'
model_cfg = dyn_model_import(dataset, args.model_name)

# Convert to CoreML and save
logger.info("Converting to CoreML")
CKPT_PATH = args.model_ckpt
C, H, W = (3, 256, 192)

model = ViTPose(model_cfg)

ckpt = torch.load(CKPT_PATH, map_location='cpu', weights_only=True)
if 'state_dict' in ckpt:
    ckpt = ckpt['state_dict']

model.load_state_dict(ckpt)
model.eval()

device = next(model.parameters()).device
inputs = torch.randn(1, C, H, W).to(device)

# Trace the model
logger.info("Tracing model with TorchScript")
traced_model = torch.jit.trace(model, inputs)

# Determine output path
base_name = os.path.basename(args.model_ckpt).replace('.pth', '')
if not os.path.isdir(args.output):
    base_name = os.path.basename(args.output).replace('.mlpackage', '')
output_dir = os.path.dirname(args.output) if not os.path.isdir(args.output) else args.output

# Export FP32 version
logger.info("Converting to CoreML (Float32 precision)")
try:
    mlmodel_fp32 = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="input", shape=(1, C, H, W))],
        outputs=[ct.TensorType(name="output")],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT32
    )

    # Add metadata
    mlmodel_fp32.short_description = f"ViTPose-{args.model_name} for {dataset} (FP32)"
    mlmodel_fp32.author = "easy_ViTPose"
    mlmodel_fp32.license = "Apache 2.0"
    mlmodel_fp32.version = "1.0"

    output_fp32 = os.path.join(output_dir, f"{base_name}_fp32.mlpackage")
    mlmodel_fp32.save(output_fp32)
    logger.success(f"FP32 model saved at: {os.path.abspath(output_fp32)}")
except Exception as e:
    logger.error(f"CoreML FP32 conversion failed: {e}")
    sys.exit(1)

# Export FP16 version
logger.info("Converting to CoreML (Float16 precision)")
try:
    mlmodel_fp16 = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="input", shape=(1, C, H, W))],
        outputs=[ct.TensorType(name="output")],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16
    )

    # Add metadata
    mlmodel_fp16.short_description = f"ViTPose-{args.model_name} for {dataset} (FP16)"
    mlmodel_fp16.author = "easy_ViTPose"
    mlmodel_fp16.license = "Apache 2.0"
    mlmodel_fp16.version = "1.0"

    output_fp16 = os.path.join(output_dir, f"{base_name}_fp16.mlpackage")
    mlmodel_fp16.save(output_fp16)
    logger.success(f"FP16 model saved at: {os.path.abspath(output_fp16)}")
except Exception as e:
    logger.error(f"CoreML FP16 conversion failed: {e}")
    sys.exit(1)

logger.info("CoreML export complete")

try:
    import torch_tensorrt
except ModuleNotFoundError:
    logger.debug("TRT module not found, skipping TensorRT export")
    sys.exit()

# From yolo convert script, pytorch -> trt
logger.info("Converting to TensorRT")
trt_ts_module = torch_tensorrt.compile(model,
    # If the inputs to the module are plain Tensors, specify them via the `inputs` argument:
    inputs = [
        torch_tensorrt.Input( # Specify input object with shape and dtype
            shape=[1, C, H, W],
            dtype=torch.float32
        )
    ],

    # TODO: ADD Datatype for inference. Allowed options torch.(float|half|int8|int32|bool)
    enabled_precisions = {torch.float32}, # half Run with FP16
    workspace_size = 1 << 28
)

# Export
output_trt = os.path.join(output_dir, f"{base_name}.engine")

input_names = ["input_0"]
output_names = ["output_0"]

device = next(model.parameters()).device
torch.jit.save(trt_ts_module, output_trt) # save the TRT embedded Torchscript

logger.success(f"TensorRT model saved at: {os.path.abspath(output_trt)}")
