"""ViTPose wrapper linking inference with `posepipe.videos`.

This module provides a lightweight wrapper around `inference.CoreMLViTPoseInference`
to make it easy to run pose estimation from a notebook that stores video records
in a `posepipe.videos` table (DataJoint or similar).

The wrapper is deliberately flexible: it will use DataJoint when available and
when a `video_id` is supplied, or accept explicit filesystem paths / pandas
DataFrame rows. It exposes convenience methods intended for interactive use in
notebooks.

Notes:
- This wrapper does not add heavy dependencies; DataJoint is optional.
- The underlying ViTPose inference requires CoreML models and macOS support.
"""

from typing import Optional, Iterable, Dict, Any, List
import os
import logging

try:
    # DataJoint is optional; only import if available
    import datajoint as dj  # type: ignore
    _HAS_DJ = True
except Exception:
    dj = None  # type: ignore
    _HAS_DJ = False

import numpy as np

# Local imports (deferred imports happen in _ensure_model)
from typing import Any

logger = logging.getLogger(__name__)


class ViTPoseWrapper:
    """Notebook-friendly wrapper for ViTPose inference.

    Example usage in a notebook:

    wrapper = ViTPoseWrapper(vitpose_model='/path/to/model.mlpackage')
    # If using DataJoint and connected:
    result = wrapper.process_video_by_id(123, output_dir='results')

    Or provide a direct video path:
    result = wrapper.process_video_path('/data/video.mp4', output_dir='results')
    """

    def __init__(
        self,
        vitpose_model_path: str,
        yolo_model_path: Optional[str] = None,
        dataset: str = 'wholebody',
        yolo_size: int = 640,
        yolo_step: int = 1,
        single_pose: bool = False,
        conf_threshold: float = 0.35,
    ):
        assert os.path.exists(vitpose_model_path), f"ViTPose model not found: {vitpose_model_path}"

        self.vitpose_model_path = vitpose_model_path
        self.yolo_model_path = yolo_model_path
        self.dataset = dataset
        self.yolo_size = yolo_size
        self.yolo_step = yolo_step
        self.single_pose = single_pose
        self.conf_threshold = conf_threshold

        # Lazy-initialize inference model when first needed
        self._model: Optional[Any] = None
        # remember which backend we will use: 'coreml' or 'tflite'
        self._backend = 'tflite' if self.vitpose_model_path.lower().endswith('.tflite') else 'coreml'

    def _ensure_model(self):
        if self._model is None:
            if self._backend == 'tflite':
                logger.info('Initializing TFLite ViTPose model')
                # Import here to avoid importing TF/tflite unless needed
                from .inference_tflite import TFLiteViTPoseInference as BackendClass
            else:
                logger.info('Initializing CoreML ViTPose model')
                from .inference import CoreMLViTPoseInference as BackendClass

            self._model = BackendClass(
                vitpose_model_path=self.vitpose_model_path,
                yolo_model_path=self.yolo_model_path,
                dataset=self.dataset,
                yolo_size=self.yolo_size,
                is_video=True,
                single_pose=self.single_pose,
                yolo_step=self.yolo_step,
                conf_threshold=self.conf_threshold,
            )

    # ---- Data access helpers ----
    def _get_video_path_from_dj(self, video_id: int) -> str:
        """Fetch video path from a `posepipe.videos` DataJoint table row.

        The function attempts to read common column names: 'path', 'video_path',
        'file_path', or 'filepath'. If DataJoint isn't available or the column is
        missing, an informative error is raised.
        """
        if not _HAS_DJ:
            raise RuntimeError('DataJoint is not installed or failed to import')

        # Access the posepipe.videos table
        try:
            table = dj.Table('posepipe.videos')
        except Exception as e:
            raise RuntimeError(f"Could not access DataJoint table 'posepipe.videos': {e}")

        try:
            row = (table & {'video_id': int(video_id)}).fetch(as_dict=True)
        except Exception:
            # Try with 'id' key or primary key fallback
            try:
                row = (table & {list(table.primary_key)[0]: int(video_id)}).fetch(as_dict=True)
            except Exception as e:
                raise RuntimeError(f"Failed to query posepipe.videos for id {video_id}: {e}")

        if not row:
            raise KeyError(f'No row found in posepipe.videos for id {video_id}')

        row = row[0]

        for k in ('path', 'video_path', 'file_path', 'filepath', 'filename'):
            if k in row and row[k]:
                return row[k]

        # If there is a column containing a path-like string, pick first str field
        for k, v in row.items():
            if isinstance(v, str) and os.path.exists(v):
                return v

        raise KeyError(
            "Could not find a path field in posepipe.videos row. "
            "Expected columns like 'path', 'video_path', 'file_path'."
        )

    # ---- High level processing methods ----
    def process_video_path(self, video_path: str, output_dir: str, show: bool = False, save_video: bool = False) -> Dict[str, Any]:
        """Process a single video given by `video_path` and return the result dict.

        This is a thin wrapper around `process_video` in `inference.py`.
        """
        assert os.path.exists(video_path), f"Video not found: {video_path}"

        self._ensure_model()

        # Build args namespace compatible with process_video
        class _Args:
            pass

        args = _Args()
        args.rotate = 0
        args.show = show
        args.save_video = save_video
        args.show_bbox = True
        args.conf_threshold = self.conf_threshold
        args.vitpose_model = os.path.basename(self.vitpose_model_path)
        args.dataset = self.dataset

        # Import the correct process_video function based on backend
        if self._backend == 'tflite':
            from .inference_tflite import process_video
        else:
            from .inference import process_video

        result = process_video(self._model, video_path, output_dir, args)
        return result

    def process_video_by_id(self, video_id: int, output_dir: str, show: bool = False, save_video: bool = False) -> Dict[str, Any]:
        """Fetch video path from `posepipe.videos` and process it.

        Requires DataJoint to be importable and a connected database session.
        Pass `video_id` as stored in the table.
        """
        video_path = self._get_video_path_from_dj(video_id)
        return self.process_video_path(video_path, output_dir, show=show, save_video=save_video)

    def process_multiple_by_ids(self, video_ids: Iterable[int], output_dir: str, show: bool = False, save_video: bool = False) -> List[Dict[str, Any]]:
        """Process multiple videos by their `posepipe.videos` IDs.

        Returns a list of result dictionaries in the same order as `video_ids`.
        """
        results = []
        for vid in video_ids:
            try:
                res = self.process_video_by_id(vid, output_dir, show=show, save_video=save_video)
                results.append(res)
            except Exception as e:
                logger.error(f"Failed to process video id {vid}: {e}")
        return results


def example_notebook_usage():
    """Prints an example that can be copy-pasted into an interactive notebook.

    The example shows both DataJoint and direct-path usage.
    """
    print("Example usage:\n")
    print("# Direct video path:\nfrom vitpose_mobile.wrapper import ViTPoseWrapper")
    print("wrapper = ViTPoseWrapper(vitpose_model_path='/path/to/vit.mlpackage')")
    print("res = wrapper.process_video_path('/full/path/to/video.mp4', output_dir='results', save_video=True)")
    print("\n# Using DataJoint (ensure dj.Connection()):\nwrapper = ViTPoseWrapper(vitpose_model_path='/path/to/vit.mlpackage')")
    print("res = wrapper.process_video_by_id(123, output_dir='results')")


__all__ = ["ViTPoseWrapper", "example_notebook_usage"]
