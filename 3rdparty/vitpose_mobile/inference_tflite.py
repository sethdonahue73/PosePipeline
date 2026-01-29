#!/usr/bin/env python3
"""
TFLite Video Inference Script for ViTPose

This script runs pose estimation on video files using TFLite models.
Supports single video or batch processing of a directory.
Compatible with both full TensorFlow and lightweight tflite-runtime.

Usage:
    # Single video (full-image mode)
    python inference_tflite.py --input video.mp4 --vitpose-model model.tflite

    # With YOLO detection
    python inference_tflite.py --input video.mp4 --vitpose-model model.tflite --yolo-model yolo.tflite

    # Directory of videos with video output
    python inference_tflite.py --input videos/ --vitpose-model model.tflite --save-video

Requirements:
    - tensorflow OR tflite-runtime
    - opencv-python
    - numpy
    - loguru
"""

import argparse
import os
import time
from datetime import datetime
from typing import Tuple, List, Dict, Any

import cv2
import numpy as np
import logging
try:
    from loguru import logger  # type: ignore
except Exception:
    logger = logging.getLogger(__name__)

# TFLite runtime - support both full TF and lightweight runtime
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

# Import utilities
from utils.inference import VideoReader, pad_image
from utils.top_down_eval import keypoints_from_heatmaps
from utils.visualization import draw_points_and_skeleton, joints_dict
from utils.sort import Sort

# Constants
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
VITPOSE_INPUT_SIZE = (192, 256)  # width, height
YOLO_HUMAN_CLASS = 0
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}


def find_videos(path: str) -> List[str]:
    """
    Find all video files in path (file or directory).

    Args:
        path: Path to a video file or directory containing videos

    Returns:
        List of video file paths, sorted alphabetically
    """
    if os.path.isfile(path):
        ext = os.path.splitext(path)[1].lower()
        if ext in VIDEO_EXTENSIONS:
            return [path]
        else:
            logger.warning(f"File {path} is not a recognized video format")
            return []
    elif os.path.isdir(path):
        videos = []
        for f in os.listdir(path):
            if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS:
                videos.append(os.path.join(path, f))
        return sorted(videos)
    else:
        logger.error(f"Path does not exist: {path}")
        return []


class TFLiteViTPoseInference:
    """
    TFLite-based ViTPose inference class.

    Args:
        vitpose_model_path: Path to ViTPose TFLite model (.tflite)
        yolo_model_path: Path to YOLO TFLite model (.tflite)
        dataset: Dataset name for keypoint visualization (default: 'wholebody')
        yolo_size: Input size for YOLO model (default: 640)
        is_video: Whether processing video (enables tracking)
        single_pose: Skip tracking for single-person videos
        yolo_step: Run YOLO every N frames (use tracker in between)
        conf_threshold: Confidence threshold for detections
    """

    def __init__(
        self,
        vitpose_model_path: str,
        yolo_model_path: str = None,
        dataset: str = 'wholebody',
        yolo_size: int = 640,
        is_video: bool = False,
        single_pose: bool = False,
        yolo_step: int = 1,
        conf_threshold: float = 0.35
    ):
        assert os.path.exists(vitpose_model_path), f"ViTPose model not found: {vitpose_model_path}"

        logger.info(f"Loading ViTPose TFLite model: {vitpose_model_path}")
        self.vitpose_interpreter = Interpreter(model_path=vitpose_model_path)
        self.vitpose_interpreter.allocate_tensors()
        self.vitpose_input = self.vitpose_interpreter.get_input_details()[0]
        self.vitpose_output = self.vitpose_interpreter.get_output_details()[0]

        # Detect input format (NCHW vs NHWC)
        input_shape = self.vitpose_input['shape']
        self.vitpose_is_nchw = input_shape[1] == 3  # (1, 3, 256, 192) vs (1, 256, 192, 3)
        logger.info(f"ViTPose input shape: {input_shape} ({'NCHW' if self.vitpose_is_nchw else 'NHWC'})")
        logger.info(f"ViTPose output shape: {self.vitpose_output['shape']}")

        # Load YOLO model if provided
        self.yolo_interpreter = None
        self.yolo_input = None
        self.yolo_output = None
        if yolo_model_path and os.path.exists(yolo_model_path):
            logger.info(f"Loading YOLO TFLite model: {yolo_model_path}")
            self.yolo_interpreter = Interpreter(model_path=yolo_model_path)
            self.yolo_interpreter.allocate_tensors()
            self.yolo_input = self.yolo_interpreter.get_input_details()[0]
            self.yolo_output = self.yolo_interpreter.get_output_details()
            logger.info(f"YOLO input shape: {self.yolo_input['shape']}")
        else:
            logger.warning("No YOLO model provided, will use entire image for pose estimation")

        self.dataset = dataset
        self.yolo_size = yolo_size
        self.is_video = is_video
        self.single_pose = single_pose
        self.yolo_step = yolo_step
        self.conf_threshold = conf_threshold

        # Initialize tracker
        self.reset()

        # State for visualization
        self._img = None
        self._keypoints = {}
        self._bboxes = []
        self._ids = []
        self._scores = []

    def reset(self):
        """Reset tracker for new video."""
        min_hits = 3 if self.yolo_step == 1 else 1
        use_tracker = self.is_video and not self.single_pose and self.yolo_interpreter is not None
        self.tracker = Sort(max_age=self.yolo_step, min_hits=min_hits, iou_threshold=0.3) if use_tracker else None
        self.frame_counter = 0

    def preprocess_vitpose_input(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocess image for ViTPose model.

        Args:
            img: Input image in RGB format

        Returns:
            Preprocessed image tensor
        """
        # Resize to model input size
        img_resized = cv2.resize(img, VITPOSE_INPUT_SIZE, interpolation=cv2.INTER_LINEAR)

        # Normalize to [0, 1]
        img_normalized = img_resized.astype(np.float32) / 255.0

        # Apply ImageNet normalization
        img_normalized = (img_normalized - MEAN) / STD

        # Add batch dimension
        if self.vitpose_is_nchw:
            # Convert to CHW format: (1, 3, H, W)
            img_tensor = img_normalized.transpose(2, 0, 1)[np.newaxis, :, :, :]
        else:
            # Keep HWC format: (1, H, W, 3)
            img_tensor = img_normalized[np.newaxis, :, :, :]

        return img_tensor.astype(np.float32)

    def preprocess_yolo_input(self, img: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Preprocess image for YOLO model.

        Args:
            img: Input image in RGB format

        Returns:
            Tuple of (preprocessed tensor, scale, (top_pad, left_pad))
        """
        h, w = img.shape[:2]
        scale = self.yolo_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to square
        pad_h = self.yolo_size - new_h
        pad_w = self.yolo_size - new_w
        top_pad, bottom = pad_h // 2, pad_h - pad_h // 2
        left_pad, right = pad_w // 2, pad_w - pad_w // 2

        img_padded = cv2.copyMakeBorder(
            img_resized, top_pad, bottom, left_pad, right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        # Normalize to [0, 1]
        img_normalized = img_padded.astype(np.float32) / 255.0

        # Add batch dimension: (1, H, W, 3) for TFLite YOLO
        img_tensor = img_normalized[np.newaxis, :, :, :]

        return img_tensor, scale, (top_pad, left_pad)

    def detect_persons(self, img: np.ndarray) -> Tuple[np.ndarray, List]:
        """
        Detect persons using YOLO TFLite model.

        Args:
            img: Input image in RGB format

        Returns:
            Tuple of (bboxes, scores) where bboxes is (N, 4) and scores is list of N floats
        """
        if self.yolo_interpreter is None:
            h, w = img.shape[:2]
            return np.array([[0, 0, w, h]]), [1.0]

        h, w = img.shape[:2]
        img_input, scale, (top_pad, left_pad) = self.preprocess_yolo_input(img)

        # Run YOLO inference
        self.yolo_interpreter.set_tensor(self.yolo_input['index'], img_input)
        self.yolo_interpreter.invoke()

        # Get outputs - TFLite YOLO output format varies
        bboxes = []
        scores = []

        # Try to parse output based on shape
        outputs = {}
        for detail in self.yolo_output:
            output = self.yolo_interpreter.get_tensor(detail['index'])
            outputs[detail['name']] = output

            if self.frame_counter == 0:
                logger.debug(f"YOLO output '{detail['name']}': shape={output.shape}")

        # Common TFLite YOLO output formats:
        # 1. YOLOv8: (1, 84, 8400) - transposed format [4 bbox + 80 classes, num_anchors]
        # 2. (1, N, 85) or (N, 85): [x, y, w, h, class_scores...]
        # 3. Multiple tensors: boxes, scores, classes, num_detections

        # Try to find detection data
        for name, output in outputs.items():
            # YOLOv8 transposed format: (1, 84, 8400)
            if len(output.shape) == 3 and output.shape[1] == 84:
                # Transpose to (1, 8400, 84) then remove batch -> (8400, 84)
                detections = output[0].T  # (84, 8400) -> (8400, 84)

                for det in detections:
                    x_center, y_center, width_det, height_det = det[:4]
                    class_scores = det[4:]  # 80 class scores
                    cls = np.argmax(class_scores)
                    conf = class_scores[cls]

                    # Filter: person class (0) and confidence
                    if cls == 0 and conf > self.conf_threshold:
                        # Coordinates are normalized (0-1), convert to pixels in padded image
                        x_c_px = x_center * self.yolo_size
                        y_c_px = y_center * self.yolo_size
                        w_px = width_det * self.yolo_size
                        h_px = height_det * self.yolo_size

                        x1_px = x_c_px - w_px / 2
                        y1_px = y_c_px - h_px / 2
                        x2_px = x_c_px + w_px / 2
                        y2_px = y_c_px + h_px / 2

                        # Convert to original image coords
                        x1 = (x1_px - left_pad) / scale
                        y1 = (y1_px - top_pad) / scale
                        x2 = (x2_px - left_pad) / scale
                        y2 = (y2_px - top_pad) / scale

                        # Clip to image bounds
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(w, x2)
                        y2 = min(h, y2)

                        bboxes.append([x1, y1, x2, y2])
                        scores.append(float(conf))
                break

            elif len(output.shape) == 3 and output.shape[2] >= 5:
                # Format: (1, N, 5+) - typical YOLOv8 output
                detections = output[0]  # Remove batch dim
                for det in detections:
                    x_center, y_center, width_det, height_det = det[:4]

                    if len(det) == 5:
                        conf = det[4]
                        cls = 0
                    elif len(det) > 5:
                        class_scores = det[4:]
                        cls = np.argmax(class_scores)
                        conf = class_scores[cls]
                    else:
                        continue

                    if cls == 0 and conf > self.conf_threshold:
                        x1_px = x_center - width_det / 2
                        y1_px = y_center - height_det / 2
                        x2_px = x_center + width_det / 2
                        y2_px = y_center + height_det / 2

                        x1 = (x1_px - left_pad) / scale
                        y1 = (y1_px - top_pad) / scale
                        x2 = (x2_px - left_pad) / scale
                        y2 = (y2_px - top_pad) / scale

                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(w, x2)
                        y2 = min(h, y2)

                        bboxes.append([x1, y1, x2, y2])
                        scores.append(float(conf))
                break

            elif len(output.shape) == 2 and output.shape[1] >= 5:
                # Format: (N, 5+) - flattened detections
                for det in output:
                    if len(det) >= 6:
                        x1, y1, x2, y2, conf, cls = det[:6]
                    else:
                        x1, y1, x2, y2, conf = det[:5]
                        cls = 0

                    if int(cls) == 0 and conf > self.conf_threshold:
                        # Convert from padded to original coords
                        x1 = (x1 - left_pad) / scale
                        y1 = (y1 - top_pad) / scale
                        x2 = (x2 - left_pad) / scale
                        y2 = (y2 - top_pad) / scale

                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(w, x2)
                        y2 = min(h, y2)

                        bboxes.append([x1, y1, x2, y2])
                        scores.append(float(conf))
                break

        if len(bboxes) == 0:
            return np.empty((0, 4)), []

        # Apply NMS
        bboxes = np.array(bboxes)
        scores = np.array(scores)
        indices = self._nms(bboxes, scores, iou_threshold=0.5)

        return bboxes[indices], scores[indices].tolist()

    def _nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> np.ndarray:
        """
        Non-Maximum Suppression.

        Args:
            boxes: Array of boxes (N, 4) in xyxy format
            scores: Array of scores (N,)
            iou_threshold: IoU threshold for suppression

        Returns:
            Indices of boxes to keep
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)

            if order.size == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return np.array(keep)

    def postprocess_heatmaps(self, heatmaps: np.ndarray, org_w: int, org_h: int) -> np.ndarray:
        """
        Convert heatmaps to keypoints.

        Args:
            heatmaps: Model output heatmaps (1, num_keypoints, H, W) in NCHW format
            org_w: Original image width
            org_h: Original image height

        Returns:
            Keypoints array (num_keypoints, 3) with [x, y, confidence]
        """
        points, prob = keypoints_from_heatmaps(
            heatmaps=heatmaps,
            center=np.array([[org_w // 2, org_h // 2]]),
            scale=np.array([[org_w, org_h]]),
            unbiased=True,
            use_udp=True
        )

        # Concatenate points and probabilities: (N, 2) + (N, 1) -> (N, 3)
        return np.concatenate([points[:, :, ::-1], prob], axis=2)

    def inference_vitpose(self, img_crop: np.ndarray) -> np.ndarray:
        """
        Run ViTPose inference on cropped person image.

        Args:
            img_crop: Cropped image containing person (RGB)

        Returns:
            Keypoints array (num_keypoints, 3) with [x, y, confidence]
        """
        org_h, org_w = img_crop.shape[:2]

        # Preprocess
        img_input = self.preprocess_vitpose_input(img_crop)

        if self.frame_counter == 1:
            logger.debug(f"ViTPose input tensor shape: {img_input.shape}")

        # Run TFLite inference
        self.vitpose_interpreter.set_tensor(self.vitpose_input['index'], img_input)
        self.vitpose_interpreter.invoke()
        heatmaps = self.vitpose_interpreter.get_tensor(self.vitpose_output['index'])

        if self.frame_counter == 1:
            logger.debug(f"ViTPose output shape: {heatmaps.shape}")

        # Convert NHWC to NCHW if needed for postprocessing
        # Expected output: (1, num_keypoints, 64, 48) in NCHW format
        # TFLite may output: (1, 64, 48, num_keypoints) in NHWC format
        if len(heatmaps.shape) == 4:
            # Check if output is NHWC by looking at dimensions
            # NCHW: (1, num_keypoints, 64, 48) - num_keypoints >> 64
            # NHWC: (1, 64, 48, num_keypoints) - 64 < num_keypoints
            if heatmaps.shape[1] < heatmaps.shape[-1]:
                # NHWC format detected: (1, H, W, num_keypoints) -> (1, num_keypoints, H, W)
                heatmaps = heatmaps.transpose(0, 3, 1, 2)
                if self.frame_counter == 1:
                    logger.debug(f"Transposed heatmaps to NCHW: {heatmaps.shape}")

        # Postprocess heatmaps to keypoints
        keypoints = self.postprocess_heatmaps(heatmaps, org_w, org_h)

        return keypoints[0]  # Remove batch dimension

    def inference(self, img: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Run full inference pipeline on an image.

        Args:
            img: Input image in RGB format

        Returns:
            Dictionary mapping person ID to keypoints array
        """
        # Detect persons
        res_pd = np.empty((0, 5))
        if self.yolo_interpreter and (self.tracker is None or
                                (self.frame_counter % self.yolo_step == 0 or self.frame_counter < 3)):
            bboxes, scores = self.detect_persons(img)
            if len(bboxes) > 0:
                res_pd = np.column_stack([bboxes, scores])

        self.frame_counter += 1

        # Apply tracker
        frame_keypoints = {}
        ids = None

        if self.tracker is not None and len(res_pd) > 0:
            res_pd = self.tracker.update(res_pd)
            ids = res_pd[:, 5].astype(int).tolist() if len(res_pd) > 0 else []

        # Prepare bboxes
        bboxes = res_pd[:, :4].round().astype(int) if len(res_pd) > 0 else np.array([])
        scores = res_pd[:, 4].tolist() if len(res_pd) > 0 else []

        # If no YOLO, use full image
        if self.yolo_interpreter is None:
            h, w = img.shape[:2]
            bboxes = np.array([[0, 0, w, h]])
            scores = [1.0]
            ids = [0]
        elif ids is None and len(bboxes) > 0:
            ids = list(range(len(bboxes)))
        elif len(bboxes) == 0:
            ids = []

        # Add padding to bboxes
        pad_bbox = 10

        for bbox, person_id, score in zip(bboxes, ids, scores):
            # Expand bbox slightly
            bbox[[0, 2]] = np.clip(bbox[[0, 2]] + [-pad_bbox, pad_bbox], 0, img.shape[1])
            bbox[[1, 3]] = np.clip(bbox[[1, 3]] + [-pad_bbox, pad_bbox], 0, img.shape[0])

            # Crop and pad to 3:4 aspect ratio
            img_crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]

            if img_crop.size == 0:
                continue

            img_padded, (left_pad, top_pad) = pad_image(img_crop, 3 / 4)

            # Run pose estimation
            try:
                keypoints = self.inference_vitpose(img_padded)

                if self.frame_counter == 1 and len(frame_keypoints) == 0:
                    logger.debug(f"Keypoints shape: {keypoints.shape}")

                # Transform keypoints back to original image coordinates
                keypoints[:, :2] += bbox[:2][::-1] - [top_pad, left_pad]

                frame_keypoints[person_id] = keypoints
            except Exception as e:
                logger.error(f"Pose estimation error for person {person_id}: {e}")
                if self.frame_counter <= 3:
                    logger.exception("Full traceback:")
                continue

        # Save state for visualization
        self._img = img
        self._keypoints = frame_keypoints
        self._bboxes = bboxes
        self._ids = ids
        self._scores = scores

        return frame_keypoints

    def draw(self, show_bbox: bool = True, confidence_threshold: float = 0.5) -> np.ndarray:
        """
        Draw keypoints and bboxes on the last processed image.

        Args:
            show_bbox: Whether to show bounding boxes
            confidence_threshold: Minimum confidence to draw keypoints

        Returns:
            Annotated image in RGB format
        """
        if self._img is None:
            return None

        img = self._img.copy()

        # Draw bboxes
        if show_bbox and len(self._bboxes) > 0:
            img_bgr = np.ascontiguousarray(img[..., ::-1])
            for bbox, person_id, score in zip(self._bboxes, self._ids, self._scores):
                x1, y1, x2, y2 = bbox
                color = (0, 255, 0)
                cv2.rectangle(img_bgr, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = f"ID:{person_id} {score:.2f}"
                cv2.putText(img_bgr, label, (int(x1), int(y1) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            img = img_bgr[..., ::-1].copy()

        # Draw skeleton
        img_bgr = np.ascontiguousarray(img[..., ::-1])
        skeleton = joints_dict()[self.dataset]['skeleton']

        for person_id, keypoints in self._keypoints.items():
            img_bgr = draw_points_and_skeleton(
                img_bgr.copy(),
                keypoints,
                skeleton,
                person_index=person_id,
                points_color_palette='gist_rainbow',
                skeleton_color_palette='jet',
                points_palette_samples=10,
                confidence_threshold=confidence_threshold
            )

        return img_bgr[..., ::-1]


def process_video(
    model: TFLiteViTPoseInference,
    video_path: str,
    output_dir: str,
    args: argparse.Namespace
) -> Dict[str, Any]:
    """
    Process a single video file and save results.

    Args:
        model: TFLiteViTPoseInference instance
        video_path: Path to input video file
        output_dir: Directory for output files
        args: Command line arguments

    Returns:
        Dictionary with processing results and metadata
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Reset tracker for new video
    model.reset()

    # Setup video reader
    logger.info(f"Loading video: {video_path}")
    reader = VideoReader(video_path, args.rotate)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    logger.info(f"Video info: {total_frames} frames, {fps:.2f} FPS, {width}x{height}")

    # Setup video writer if requested
    out_writer = None
    video_output_path = None
    if args.save_video:
        video_output_path = os.path.join(output_dir, f"{video_name}_pose.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))
        logger.info(f"Saving video to: {video_output_path}")

    # Process video
    all_keypoints = []
    inference_times = []
    total_time = 0.0

    try:
        for frame_idx, img in enumerate(reader):
            t0 = time.time()

            # Run inference
            frame_keypoints = model.inference(img)
            all_keypoints.append(frame_keypoints)

            t1 = time.time()
            inference_time = t1 - t0
            inference_times.append(inference_time)
            total_time += inference_time

            # Draw and show/save
            if args.show or args.save_video:
                img_draw = model.draw(show_bbox=args.show_bbox,
                                     confidence_threshold=args.conf_threshold)

                if img_draw is not None:
                    img_bgr = img_draw[..., ::-1]

                    # Add FPS overlay
                    current_fps = 1.0 / inference_time if inference_time > 0 else 0
                    cv2.putText(img_bgr, f"FPS: {current_fps:.1f}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    if args.save_video and out_writer:
                        out_writer.write(img_bgr)

                    if args.show:
                        cv2.imshow('TFLite ViTPose Inference', img_bgr)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            logger.info("Stopped by user")
                            break

            # Progress
            if (frame_idx + 1) % 30 == 0 or (frame_idx + 1) == total_frames:
                avg_fps = 1.0 / np.mean(inference_times[-30:]) if len(inference_times) > 0 else 0
                logger.info(f"Frame {frame_idx + 1}/{total_frames} | FPS: {avg_fps:.2f} | Persons: {len(frame_keypoints)}")

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        raise

    finally:
        if out_writer:
            out_writer.release()

    # Calculate statistics
    stats = {}
    if len(inference_times) > 0:
        total_poses = sum(len(kp) for kp in all_keypoints)
        stats = {
            'mean_fps': float(1.0 / np.mean(inference_times)),
            'total_time': float(total_time),
            'mean_inference_ms': float(np.mean(inference_times) * 1000),
            'min_inference_ms': float(np.min(inference_times) * 1000),
            'max_inference_ms': float(np.max(inference_times) * 1000),
            'total_poses': total_poses,
            'mean_poses_per_frame': float(total_poses / len(all_keypoints)) if all_keypoints else 0
        }

    # Build result
    result = {
        'metadata': {
            'video': os.path.basename(video_path),
            'video_path': video_path,
            'model': os.path.basename(args.vitpose_model),
            'dataset': args.dataset,
            'processed_at': datetime.now().isoformat(),
            'total_frames': len(all_keypoints),
            'fps': fps,
            'resolution': [width, height]
        },
        'statistics': stats,
        'keypoints': all_keypoints,
        'skeleton': joints_dict()[args.dataset]['keypoints']
    }

    if video_output_path:
        logger.success(f"Saved video to: {video_output_path}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description='TFLite ViTPose inference on video',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single video (full-image mode)
  python inference_tflite.py --input video.mp4 --vitpose-model model.tflite

  # With YOLO detection
  python inference_tflite.py --input video.mp4 --vitpose-model model.tflite --yolo-model yolo.tflite

  # Directory of videos with video output
  python inference_tflite.py --input videos/ --vitpose-model model.tflite --save-video
        """
    )
    parser.add_argument('--input', type=str, required=True,
                       help='Path to video file or directory of videos')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for JSON and video files (default: results)')
    parser.add_argument('--vitpose-model', type=str, required=True,
                       help='Path to ViTPose TFLite model (.tflite)')
    parser.add_argument('--yolo-model', type=str, default=None,
                       help='Path to YOLO TFLite model (.tflite), if None uses full image')
    parser.add_argument('--dataset', type=str, default='wholebody',
                       choices=['coco', 'coco_25', 'wholebody', 'mpii', 'ap10k', 'apt36k', 'aic'],
                       help='Dataset name for keypoint skeleton')
    parser.add_argument('--yolo-size', type=int, default=640,
                       help='YOLO input size')
    parser.add_argument('--yolo-step', type=int, default=1,
                       help='Run YOLO every N frames (use tracker in between)')
    parser.add_argument('--single-pose', action='store_true',
                       help='Single person mode (no tracking)')
    parser.add_argument('--conf-threshold', type=float, default=0.5,
                       help='Confidence threshold for visualization')
    parser.add_argument('--show', action='store_true',
                       help='Display video during processing')
    parser.add_argument('--save-video', action='store_true',
                       help='Save annotated video output')
    parser.add_argument('--show-bbox', action='store_true',
                       help='Show bounding boxes')
    parser.add_argument('--rotate', type=int, choices=[0, 90, 180, 270], default=0,
                       help='Rotate video counterclockwise')

    args = parser.parse_args()

    # Validate inputs
    assert os.path.exists(args.input), f"Input not found: {args.input}"
    assert os.path.exists(args.vitpose_model), f"ViTPose model not found: {args.vitpose_model}"

    if args.yolo_model:
        assert os.path.exists(args.yolo_model), f"YOLO model not found: {args.yolo_model}"

    # Find videos
    videos = find_videos(args.input)
    if not videos:
        logger.error(f"No video files found in: {args.input}")
        return

    logger.info("=" * 60)
    logger.info("TFLite ViTPose Video Inference")
    logger.info("=" * 60)
    logger.info(f"Found {len(videos)} video(s) to process")

    # Determine output directory
    if os.path.isdir(args.input):
        dir_name = os.path.basename(os.path.normpath(args.input))
        output_dir = os.path.join(args.output_dir, dir_name)
    else:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {os.path.abspath(output_dir)}")

    # Initialize model once
    model = TFLiteViTPoseInference(
        vitpose_model_path=args.vitpose_model,
        yolo_model_path=args.yolo_model,
        dataset=args.dataset,
        yolo_size=args.yolo_size,
        is_video=True,
        single_pose=args.single_pose,
        yolo_step=args.yolo_step,
        conf_threshold=0.35
    )

    # Process each video
    results = []
    try:
        for i, video_path in enumerate(videos):
            logger.info("-" * 60)
            logger.info(f"Processing video {i + 1}/{len(videos)}: {os.path.basename(video_path)}")
            logger.info("-" * 60)

            result = process_video(model, video_path, output_dir, args)
            results.append(result)

            # Print statistics for this video
            stats = result.get('statistics', {})
            if stats:
                logger.info(f"  Mean FPS: {stats.get('mean_fps', 0):.2f}")
                logger.info(f"  Total time: {stats.get('total_time', 0):.2f}s")
                logger.info(f"  Total poses: {stats.get('total_poses', 0)}")

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
    finally:
        cv2.destroyAllWindows()

    # Summary
    logger.info("=" * 60)
    logger.info("Processing Complete")
    logger.info("=" * 60)
    logger.info(f"Processed {len(results)}/{len(videos)} videos")
    logger.info(f"Results saved to: {os.path.abspath(output_dir)}")


if __name__ == '__main__':
    main()
