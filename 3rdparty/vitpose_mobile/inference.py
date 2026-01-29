#!/usr/bin/env python3
"""
CoreML Video Inference Script for ViTPose on macOS Desktop

This script runs pose estimation on video files using CoreML models.
Supports single video or batch processing of a directory.

Usage:
    # Single video
    python inference.py --input video.mp4 --vitpose-model model.mlpackage

    # Directory of videos
    python inference.py --input videos/ --vitpose-model model.mlpackage

    # With YOLO detection and video output
    python inference.py --input videos/ --vitpose-model model.mlpackage --yolo-model yolo.mlpackage --save-video

Requirements:
    - macOS (for CoreML support)
    - coremltools
    - opencv-python
    - numpy
    - pillow
"""

import argparse
import os
import time
from datetime import datetime
from typing import Tuple, List, Dict, Any

import cv2
import numpy as np
import coremltools as ct
from PIL import Image
import logging
try:
    from loguru import logger  # type: ignore
except Exception:
    logger = logging.getLogger(__name__)

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


class CoreMLViTPoseInference:
    """
    CoreML-based ViTPose inference class for macOS desktop.

    Args:
        vitpose_model_path: Path to ViTPose CoreML model (.mlpackage)
        yolo_model_path: Path to YOLO CoreML model (.mlpackage)
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

        logger.info(f"Loading ViTPose CoreML model: {vitpose_model_path}")
        self.vitpose_model = ct.models.MLModel(vitpose_model_path)

        # Load YOLO model if provided
        self.yolo_model = None
        if yolo_model_path and os.path.exists(yolo_model_path):
            logger.info(f"Loading YOLO CoreML model: {yolo_model_path}")
            self.yolo_model = ct.models.MLModel(yolo_model_path)
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
        use_tracker = self.is_video and not self.single_pose and self.yolo_model is not None
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

        # Convert to CHW format and add batch dimension: (1, 3, H, W)
        img_tensor = img_normalized.transpose(2, 0, 1)[np.newaxis, :, :, :]

        return img_tensor.astype(np.float32)

    def preprocess_yolo_input(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocess image for YOLO model.

        Args:
            img: Input image in RGB format

        Returns:
            Preprocessed image tensor
        """
        # Resize maintaining aspect ratio
        h, w = img.shape[:2]
        scale = self.yolo_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to square
        pad_h = self.yolo_size - new_h
        pad_w = self.yolo_size - new_w
        top, bottom = pad_h // 2, pad_h - pad_h // 2
        left, right = pad_w // 2, pad_w - pad_w // 2

        img_padded = cv2.copyMakeBorder(
            img_resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        # Normalize to [0, 1] and convert to float32
        img_normalized = img_padded.astype(np.float32) / 255.0

        # Add batch dimension: (1, H, W, 3)
        img_tensor = img_normalized[np.newaxis, :, :, :]

        return img_tensor, scale, (top, left)

    def detect_persons(self, img: np.ndarray) -> Tuple[np.ndarray, List]:
        """
        Detect persons using YOLO CoreML model.

        Args:
            img: Input image in RGB format

        Returns:
            Tuple of (bboxes, scores) where bboxes is (N, 4) and scores is list of N floats
        """
        if self.yolo_model is None:
            # No YOLO model, use entire image
            h, w = img.shape[:2]
            return np.array([[0, 0, w, h]]), [1.0]

        # Convert to PIL Image for CoreML YOLO
        # Resize to YOLO input size
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

        # Convert to PIL Image (CoreML YOLO expects PIL Image)
        pil_image = Image.fromarray(img_padded)

        # Run inference
        try:
            results = self.yolo_model.predict({'image': pil_image})

            # Debug: Print output keys once (first frame only)
            if self.frame_counter == 0:
                logger.debug(f"YOLO output keys: {list(results.keys())}")
                for key, value in results.items():
                    if isinstance(value, np.ndarray):
                        logger.debug(f"  {key}: shape={value.shape}, dtype={value.dtype}")

                # Debug: Show what's being detected
                if 'confidence' in results:
                    confs = results['confidence']
                    if len(confs.shape) == 2 and confs.shape[1] == 80:
                        logger.debug("Detection analysis:")
                        for i in range(len(confs)):
                            max_class = np.argmax(confs[i])
                            max_conf = confs[i, max_class]
                            person_conf = confs[i, 0]
                            logger.debug(f"  Det {i}: class={max_class}, conf={max_conf:.3f} | person_conf={person_conf:.3f}")

            # Extract bboxes and scores for person class (class 0)
            # YOLOv8 CoreML model typically outputs: 'coordinates' and 'confidence'
            bboxes = []
            scores = []

            # Try to find the correct output keys
            if 'coordinates' in results and 'confidence' in results:
                # Standard format
                coords = results['coordinates']
                confs = results['confidence']

                # coords shape: (N, 4) where N is number of detections
                # confs shape: (N, 80) where 80 is number of COCO classes
                # Each row: [x1, y1, x2, y2] in the resized image coordinates

                for i in range(len(coords)):
                    # Extract person class confidence (class 0 in COCO)
                    if len(confs.shape) == 2 and confs.shape[1] == 80:
                        # Multi-class confidence: shape (N, 80)
                        conf = float(confs[i, 0])  # Person is class 0
                    elif len(confs.shape) == 1:
                        # Single confidence value per detection
                        conf = float(confs[i])
                    else:
                        # Unknown format, take max
                        conf = float(np.max(confs[i]))

                    if conf > self.conf_threshold:
                        bbox = coords[i]

                        # Coordinates are normalized (0-1) relative to the padded image
                        # Convert to pixel coordinates in padded image, then to original
                        x_center, y_center, width_norm, height_norm = bbox

                        # Convert from normalized center+size to pixel xyxy in padded image
                        box_w = width_norm * self.yolo_size
                        box_h = height_norm * self.yolo_size
                        x_center_px = x_center * self.yolo_size
                        y_center_px = y_center * self.yolo_size

                        x1_px = x_center_px - box_w / 2
                        y1_px = y_center_px - box_h / 2
                        x2_px = x_center_px + box_w / 2
                        y2_px = y_center_px + box_h / 2

                        # Convert from padded image to original image coordinates
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
                        scores.append(conf)
            else:
                # Try alternative output format - look for any array outputs
                for key, value in results.items():
                    if isinstance(value, np.ndarray):
                        # YOLOv8 might output as single array with shape (N, 6)
                        # [x1, y1, x2, y2, conf, class]
                        if len(value.shape) == 2 and value.shape[1] >= 5:
                            for detection in value:
                                x1, y1, x2, y2, conf = detection[:5]
                                cls = detection[5] if value.shape[1] > 5 else 0

                                # Filter for person class (0) and confidence
                                if cls == 0 and conf > self.conf_threshold:
                                    # Convert back to original coordinates
                                    x1 = (x1 - left_pad) / scale
                                    y1 = (y1 - top_pad) / scale
                                    x2 = (x2 - left_pad) / scale
                                    y2 = (y2 - top_pad) / scale

                                    bboxes.append([x1, y1, x2, y2])
                                    scores.append(float(conf))
                            break

            if len(bboxes) == 0:
                return np.empty((0, 4)), []

            return np.array(bboxes), scores

        except Exception as e:
            logger.error(f"YOLO detection error: {e}")
            # Fallback to full image
            h, w = img.shape[:2]
            return np.array([[0, 0, w, h]]), [1.0]

    def postprocess_heatmaps(self, heatmaps: np.ndarray, org_w: int, org_h: int) -> np.ndarray:
        """
        Convert heatmaps to keypoints.

        Args:
            heatmaps: Model output heatmaps (1, num_keypoints, H, W)
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
        # Points come as (y, x), need to swap to (x, y)
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

        # Debug: Print input shape on first frame
        if self.frame_counter == 1:
            logger.debug(f"ViTPose input shape: {img_input.shape}")

        # Run CoreML inference
        # CoreML expects dictionary input
        results = self.vitpose_model.predict({'input': img_input})

        # Debug: Print output keys on first frame
        if self.frame_counter == 1:
            logger.debug(f"ViTPose output keys: {list(results.keys())}")
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    logger.debug(f"  {key}: shape={value.shape}, dtype={value.dtype}")

        # Extract heatmaps from output
        # The output key might vary, common names: 'output', 'var_xxx', etc.
        heatmaps = None
        for key in results.keys():
            output = results[key]
            if isinstance(output, np.ndarray) and len(output.shape) == 4:
                heatmaps = output
                break

        if heatmaps is None:
            raise ValueError(f"Could not find heatmap output in model results. Keys: {results.keys()}")

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
        if self.yolo_model and (self.tracker is None or
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
        if self.yolo_model is None:
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

                # Debug: Print keypoints info on first detection
                if self.frame_counter == 1 and len(frame_keypoints) == 0:
                    logger.debug(f"Keypoints shape: {keypoints.shape}")
                    logger.debug(f"Sample keypoints (first 3): {keypoints[:3]}")

                # Transform keypoints back to original image coordinates
                keypoints[:, :2] += bbox[:2][::-1] - [top_pad, left_pad]

                frame_keypoints[person_id] = keypoints
            except Exception as e:
                logger.error(f"Pose estimation error for person {person_id}: {e}")
                if self.frame_counter <= 3:  # Print full traceback for first few frames
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
            img_bgr = np.ascontiguousarray(img[..., ::-1])  # RGB to BGR, make contiguous
            for bbox, person_id, score in zip(self._bboxes, self._ids, self._scores):
                x1, y1, x2, y2 = bbox
                color = (0, 255, 0)
                cv2.rectangle(img_bgr, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = f"ID:{person_id} {score:.2f}"
                cv2.putText(img_bgr, label, (int(x1), int(y1) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            img = img_bgr[..., ::-1].copy()  # BGR back to RGB

        # Draw skeleton
        img_bgr = np.ascontiguousarray(img[..., ::-1])  # RGB to BGR, make contiguous
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

        return img_bgr[..., ::-1]  # BGR back to RGB


def process_video(
    model: CoreMLViTPoseInference,
    video_path: str,
    output_dir: str,
    args: argparse.Namespace
) -> Dict[str, Any]:
    """
    Process a single video file and save results.

    Args:
        model: CoreMLViTPoseInference instance
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
                    img_bgr = img_draw[..., ::-1]  # RGB to BGR

                    # Add FPS overlay
                    current_fps = 1.0 / inference_time if inference_time > 0 else 0
                    cv2.putText(img_bgr, f"FPS: {current_fps:.1f}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    if args.save_video and out_writer:
                        out_writer.write(img_bgr)

                    if args.show:
                        cv2.imshow('CoreML ViTPose Inference', img_bgr)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            logger.info("Stopped by user")
                            break

            # Progress
            if (frame_idx + 1) % 30 == 0 or (frame_idx + 1) == total_frames:
                avg_fps = 1.0 / np.mean(inference_times[-30:]) if len(inference_times) > 0 else 0
                logger.info(f"Frame {frame_idx + 1}/{total_frames} | FPS: {avg_fps:.2f} | Persons: {len(frame_keypoints)}")

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        raise  # Re-raise to stop processing other videos

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
        description='CoreML ViTPose inference on video (macOS)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single video
  python inference.py --input video.mp4 --vitpose-model model.mlpackage

  # Directory of videos
  python inference.py --input videos/ --vitpose-model model.mlpackage

  # With video output
  python inference.py --input videos/ --vitpose-model model.mlpackage --save-video
        """
    )
    parser.add_argument('--input', type=str, required=True,
                       help='Path to video file or directory of videos')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for JSON and video files (default: results)')
    parser.add_argument('--vitpose-model', type=str, required=True,
                       help='Path to ViTPose CoreML model (.mlpackage)')
    parser.add_argument('--yolo-model', type=str, default=None,
                       help='Path to YOLO CoreML model (.mlpackage), if None uses full image')
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
    logger.info("CoreML ViTPose Video Inference")
    logger.info("=" * 60)
    logger.info(f"Found {len(videos)} video(s) to process")

    # Determine output directory - mirror input dir name if directory provided
    if os.path.isdir(args.input):
        dir_name = os.path.basename(os.path.normpath(args.input))
        output_dir = os.path.join(args.output_dir, dir_name)
    else:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {os.path.abspath(output_dir)}")

    # Initialize model once
    model = CoreMLViTPoseInference(
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
