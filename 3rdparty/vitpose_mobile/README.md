# ViTPose-Mobile

Human pose estimation using ViTPose models with CoreML (macOS/iOS) and TFLite (Android) deployment.

## Directory Structure

```
vitpose/
├── export.py           # Convert PyTorch models to CoreML
├── export_tflite.py    # Convert PyTorch models to TFLite (Android)
├── inference.py        # Run pose estimation on videos (CoreML)
├── inference_tflite.py # Run pose estimation on videos (TFLite)
├── ckpts/
│   ├── torch/          # PyTorch checkpoints (.pth)
│   ├── coreml/         # CoreML models (.mlpackage)
│   ├── tflite/         # TFLite models (.tflite)
│   └── yolov8n.mlpackage/  # YOLO person detector
├── utils/              # Utility functions
├── vit_models/         # Model architecture
└── sample/             # Sample videos
```

## Installation

```bash
# Create conda environment
conda create -n vitpose python=3.10 -y
conda activate vitpose

conda install conda-forge::ffmpeg -y

# Install dependencies
pip install -r requirements.txt
```

## Model Checkpoints

Download the pre-trained model checkpoints from Google Drive:

1. Download the checkpoint zip file: [ckpts.zip](https://drive.google.com/file/d/1ECQ_BdL3KDP6S4Wf8lVlem1_mN4tX1yc/view?usp=drive_link)

2. Extract the zip file in the project root:
   ```bash
   unzip ckpts.zip -d .
   ```

3. Verify the folder structure:
   ```
   ckpts/
   ├── torch/          # PyTorch checkpoints (.pth)
   ├── coreml/         # CoreML models (.mlpackage)
   └── tflite/         # TFLite models (.tflite)
   ```

The checkpoints include models in multiple sizes (S, B, L, H) and formats for different deployment targets.

## Usage

### Export PyTorch to CoreML

Convert a PyTorch checkpoint to CoreML format:

```bash
python export.py \
    --model-ckpt ckpts/torch/vitpose-s-wholebody.pth \
    --model-name s \
    --output ckpts/coreml/
```

**Arguments:**
- `--model-ckpt`: Path to PyTorch .pth file
- `--model-name`: Model size (`s`, `b`, `l`, `h`)
- `--output`: Output directory for .mlpackage files
- `--dataset`: Dataset type (default: inferred from filename)

**Output:** Creates both FP32 and FP16 CoreML models.

### Export PyTorch to TFLite (Android)

Convert a PyTorch checkpoint to TFLite format for Android deployment:

```bash
python export_tflite.py \
    --model-ckpt ckpts/torch/vitpose-s-wholebody.pth \
    --model-name s \
    --output ckpts/tflite/
```

**Arguments:**
- `--model-ckpt`: Path to PyTorch .pth file
- `--model-name`: Model size (`s`, `b`, `l`, `h`)
- `--output`: Output directory for .tflite files
- `--dataset`: Dataset type (default: inferred from filename)

**Output:** Creates both FP32 and FP16 TFLite models.

**Note:** TFLite models use NHWC format `(1, 64, 48, keypoints)` for output heatmaps.

### Run Inference

Run pose estimation on a single video or directory of videos:

```bash
# Single video
python inference.py \
    --input sample/video.mp4 \
    --vitpose-model ckpts/coreml/vitpose-b-wholebody_fp32.mlpackage \
    --yolo-model ckpts/yolov8n.mlpackage

# Directory of videos
python inference.py \
    --input sample/ \
    --vitpose-model ckpts/coreml/vitpose-b-wholebody_fp32.mlpackage \
    --yolo-model ckpts/yolov8n.mlpackage

# With video output
python inference.py \
    --input sample/ \
    --vitpose-model ckpts/coreml/vitpose-b-wholebody_fp32.mlpackage \
    --save-video
```

**Arguments:**
- `--input`: Path to video file or directory of videos
- `--output-dir`: Output directory for results (default: `results/`)
- `--vitpose-model`: Path to ViTPose CoreML model
- `--yolo-model`: Path to YOLO CoreML model (optional)
- `--save-video`: Save annotated video output (flag)
- `--show`: Display video during processing
- `--show-bbox`: Show detection bounding boxes
- `--dataset`: Keypoint format (`wholebody`, `coco`, `mpii`, etc.)

### Run TFLite Inference

Run pose estimation on videos using TFLite models (for Android/cross-platform):

```bash
# Single video (full-image mode, no YOLO)
python inference_tflite.py \
    --input sample/video.mp4 \
    --vitpose-model ckpts/tflite/vitpose-s-wholebody_fp32.tflite

# With YOLO person detection
python inference_tflite.py \
    --input sample/video.mp4 \
    --vitpose-model ckpts/tflite/vitpose-s-wholebody_fp32.tflite \
    --yolo-model ckpts/tflite/yolov8n.tflite

# Directory of videos with video output
python inference_tflite.py \
    --input sample/ \
    --vitpose-model ckpts/tflite/vitpose-s-wholebody_fp32.tflite \
    --save-video
```

**Arguments:**
- `--input`: Path to video file or directory of videos
- `--output-dir`: Output directory for results (default: `results/`)
- `--vitpose-model`: Path to ViTPose TFLite model (.tflite)
- `--yolo-model`: Path to YOLO TFLite model (.tflite), optional
- `--save-video`: Save annotated video output
- `--show`: Display video during processing
- `--show-bbox`: Show detection bounding boxes
- `--dataset`: Keypoint format (`wholebody`, `coco`, `mpii`, etc.)
- `--yolo-step`: Run YOLO every N frames (use tracker in between)
- `--single-pose`: Single person mode (no tracking)

**Note:** Compatible with both full TensorFlow and lightweight `tflite-runtime` package.

## Models

| Model | Parameters | Keypoints |
|-------|------------|-----------|
| ViTPose-S | 24M | 133 (wholebody) |
| ViTPose-B | 86M | 133 (wholebody) |
| ViTPose-L | 307M | 133 (wholebody) |
| ViTPose-H | 632M | 133 (wholebody) |

Each model is available in FP32 and FP16 precision for both CoreML and TFLite formats.

| Format | Platform | FP32 Size (ViT-S) | FP16 Size (ViT-S) |
|--------|----------|-------------------|-------------------|
| CoreML | macOS/iOS | ~93 MB | ~47 MB |
| TFLite | Android | ~93 MB | ~47 MB |

