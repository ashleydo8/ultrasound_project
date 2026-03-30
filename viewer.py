import logging
import os
import gc
import queue
import sys
import threading
import time 
from datetime import datetime
import csv
import base64
import json
from pathlib import Path
from functools import partial
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".." ))
if project_root not in sys.path:
    sys.path.append(project_root)
    sys.path.append(os.path.join(project_root, 'drivers'))

import drivers.imaging as ultrasoundImage
from drivers.tracking import NatNetClientWrapper

import napari
from napari.components.viewer_model import ViewerModel
from napari.layers import Image, Points, Vectors, Shapes, Surface
from napari.qt import QtViewer
import numpy as np
import psutil
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
from core.recon import reconstruct_volume_from_frames
from core.calibration import NeedlePivotCalibrator
from core.structures import Pose, Event, RenderPacket, RecordingBuffer
import ui.constants as const

# IMPORTANT: On Windows, importing Qt *before* hardware DLLs can cause crashes or device enumeration failures.
# Always import all hardware dependencies (e.g., ultrasound, NatNet) FIRST!
from qtpy.QtCore import QTimer, Qt

from ui.constants import ACQ_QUEUE_MAXSIZE, ARCHIVE_MAX_SAMPLES, RECORDER_QUEUE_MAXSIZE, ULTRASOUND_PLANE_VIS_SCALE, POSE_STALE_THRESHOLD_MS, DROP_LOG_INTERVAL
from ui.constants import VOLUME_SHAPE, ULTRASOUND_RB_ID,ASSUMED_US_PIXEL_SPACING_MM, TABLE_PLANE_NORMAL, TABLE_PLANE_OFFSET_M, GUI_FPS, SLICE_AXES,SLICE_POSITIONS, GROUND_OUTLINE_HALF_SIZE_MM, GROUND_THICKNESS_MM
from ui.constants import NEEDLE_RB_ID, LOCAL_ARROW_DIR, NEEDLE_ARROW_LENGTH_MM, RECORDINGS_DIR, PIXEL_SPACING_MM, SIMPLE_US_ROT_MAT_FWD, SIMPLE_US_ROT_MAT_REV
from ui.constants import SCALE, _needle_pivot_calibrator, CALIBRATIONS_DIR, SLICE_THICKNESS_MM, QUEUE_MAXSIZE, TRACKER_CLIENT_ADDRESS, TRACKER_SERVER_ADDRESS, TRACKER_USE_MULTICAST, TRACKER_PRINT_LEVEL

from qtpy.QtWidgets import (
    QApplication,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QCheckBox,
    QFileDialog,
    QMessageBox,
)
from napari._qt.qt_viewer import QtViewer
try:
    import qdarkstyle
except ImportError:
    qdarkstyle = None
# === CONFIGURE LOGGING ===
logging.basicConfig(
    format="%(asctime)s %(threadName)s %(levelname)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

@dataclass
class UltrasoundCalibrationSample:
    """Container for raw ultrasound frames and poses during calibration."""

    timestamp: float
    frame: np.ndarray
    pose_matrix: np.ndarray
    image_points_px: np.ndarray
def _extract_pose_rt(pose: Any) -> tuple[np.ndarray, np.ndarray]:
    if pose is None:
        raise ValueError("pose must not be None")
    if isinstance(pose, np.ndarray):
        if pose.shape == (4, 4):
            return pose[:3, :3].astype(np.float64), pose[:3, 3].astype(np.float64)
        raise TypeError("Unsupported numpy pose shape")
    if hasattr(pose, "T_W_H"):
        T = np.asarray(pose.T_W_H, dtype=np.float64)
        return T[:3, :3], T[:3, 3]
    if hasattr(pose, "R") and hasattr(pose, "t"):
        R = np.asarray(pose.R, dtype=np.float64)
        t = np.asarray(pose.t, dtype=np.float64).reshape(3)
        return R, t
    if hasattr(pose, "body_rot") and hasattr(pose, "pos_mm"):
        rot = pose.body_rot
        align_rot = getattr(pose, "align_rot", None)
        if align_rot is not None:
            rot = rot * align_rot
        R = rot.as_matrix().astype(np.float64)
        t = np.asarray(pose.pos_mm, dtype=np.float64)
        return R, t
    raise TypeError("Unsupported pose type for pivot calibration")
def on_pivot_start() -> None:
    """Reset the pivot calibration accumulator."""
    _needle_pivot_calibrator.reset()
def on_pivot_capture(pose: Any, quality: float | None = None, ts: float | None = None) -> None:
    """Record a pose sample for pivot calibration."""
    R_i, t_i = _extract_pose_rt(pose)
    if quality is None and hasattr(pose, "quality"):
        quality = getattr(pose, "quality")
    if ts is None and hasattr(pose, "timestamp"):
        ts = getattr(pose, "timestamp")
    _needle_pivot_calibrator.add_pose(R_i, t_i, quality=quality, ts=ts)
def on_pivot_solve(meta: Optional[dict] = None) -> dict:
    """Solve the pivot calibration and persist results to disk."""
    global _last_tip_offset_mm
    result = _needle_pivot_calibrator.solve()
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_path = CALIBRATIONS_DIR / f"needle_pivot_{timestamp}.json"
    meta_out = dict(meta or {})
    meta_out.update(
        {
            "timestamp": timestamp,
            "units": "mm",
            "huber_delta_mm": _needle_pivot_calibrator.huber_delta_mm,
            "ransac_thresh_mm": _needle_pivot_calibrator.ransac_thresh_mm,
            "n_samples": result["n_samples"],
            "n_inliers": result["n_inliers"],
            "n_outliers": result["n_outliers"],
            "rms_mm": result["rms_mm"],
            "max_abs_residual_mm": result["max_abs_residual_mm"],
        }
    )
    _needle_pivot_calibrator.save_json(str(output_path), meta=meta_out)
    _last_tip_offset_mm = result["r_H_tip"]
    result_with_meta = dict(result)
    result_with_meta["json_path"] = str(output_path)
    result_with_meta["meta"] = meta_out
    return result_with_meta
def get_last_tip_offset() -> Optional[np.ndarray]:
    return _last_tip_offset_mm
def compute_tip_position(T_W_H: np.ndarray) -> Optional[np.ndarray]:
    if _last_tip_offset_mm is None:
        return None
    return NeedlePivotCalibrator.apply_tip(T_W_H, _last_tip_offset_mm)

recording_buffer = RecordingBuffer()