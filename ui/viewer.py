# IMPORTANT: Pose producer converts Motive/NatNet (x,y,z) -> Napari (z,y,x) and flips Napari Z (first coord).
# Long arm up in Motive (+z) therefore appears as -Z in Napari; short arm left in Motive (+x) appears as +X in Napari.
# Set long arm to point up (+z in motive, +x in napari), and short arm to point left (+x in motive, +z in napari)

"""
This script implements a robust, real-time ultrasound and optical tracking viewer.
Key design goals:
- Lossless sample preservation (500 Hz streams, 60 Hz GUI)
- Headless and GUI modes share a single core pipeline
- No daemon threads: explicit, observable thread lifecycle; join on shutdown
- All exceptions in threads are logged and immediately propagated to the GUI
- All hardware dependencies imported BEFORE Qt/QtPy (required on Windows)
- Pre-allocated buffers for low-latency updates and no memory leaks
- Full cleanup on close: producer threads joined, napari subwindows closed
"""
USE_FAKE_ULTRASOUND = True  # or False, depending on your test
USE_FAKE_POSE = False        # or False, depending on your test
#!/usr/bin/env python3
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

class RealTimeUSViewer(QWidget):
    def __init__(
        self,
        frame_q: queue.Queue,
        stop_event: threading.Event,
        consumer_thread: Optional[threading.Thread] = None,
        volume: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()
        self.q = frame_q
        self.stop_event = stop_event
        self.consumer_thread = consumer_thread
        # Pre-allocations
        if volume is None:
            # Start with a random grayscale volume (values 0-255)
            self.volume = (np.random.rand(*VOLUME_SHAPE) * 255).astype(np.float32)
        else:
            self.volume = volume
        self.us_correction_matrix = np.eye(4, dtype=np.float32)
        self.needle_correction_matrix = np.eye(4, dtype=np.float32)
        self._needle_tip_matrix: Optional[np.ndarray] = None
        self._needle_tip_offset: Optional[np.ndarray] = None
        self._last_raw_pos_m: Optional[np.ndarray] = None
        self._last_raw_rot: Optional[R] = None
        self._last_raw_needle_pos_m: Optional[np.ndarray] = None
        self._last_raw_needle_rot: Optional[R] = None
        self._current_needle_pose: Optional[Pose] = None
        self._current_needle_top_pose: Optional[Pose] = None
        self._needle_tip_points: List[np.ndarray] = []
        self.us_archive = deque(maxlen=ARCHIVE_MAX_SAMPLES)
        self.cam_archive = deque(maxlen=ARCHIVE_MAX_SAMPLES)
        self._max_markers = 5
        self.arrow_data = np.zeros((1, 2, 3), dtype=np.float32)
        self.needle_arrow_data = np.zeros((1, 2, 3), dtype=np.float32)
        self.ground_outline_mesh: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
        self.ground_outline_layer: Optional[Surface] = None
        self.ground_axes_meshes: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self.ground_axes_layers: List[Surface] = []
        self.marker_data = np.zeros((self._max_markers, 3), dtype=np.float32)
        self._current_pixel_spacing_mm = np.array(ASSUMED_US_PIXEL_SPACING_MM, dtype=np.float64)
        init_slice_thickness = float(np.mean(self._current_pixel_spacing_mm))
        self._voxel_size_mm = np.array(
            [init_slice_thickness, self._current_pixel_spacing_mm[0], self._current_pixel_spacing_mm[1]],
            dtype=np.float32,
        )
        self.log = logger
        self._needle_has_pose = False
        self._has_recon_volume = False
        self.vec_layer = None
        self.needle_vec_layer = None
        self.recon_volume_layer: Optional[Image] = None
        self.slice_viewers = {}
        self.slice_layers = {}
        self.recon_slice_layers: Dict[str, Optional[Image]] = {}
        self.ultrasound_container: Optional[QWidget] = None
        self.ultrasound_model: Optional[ViewerModel] = None
        self.ultrasound_viewer: Optional[QtViewer] = None
        self.ultrasound_image_layer: Optional[Image] = None
        self._ultrasound_frame: Optional[np.ndarray] = None
        self._last_recorded_us_ts: Optional[float] = None
        self._pivot_recording = False
        self._pivot_capture_count = 0
        self._pivot_last_sample_ts: Optional[float] = None
        self._pivot_skipped_close_ts_count = 0
        self._last_rb_status_text: str = ""
        self._us_calibration_recording = False
        self._us_calibration_samples: List[UltrasoundCalibrationSample] = []
        self._us_calibration_last_ts: Optional[float] = None
        self._last_us_calibration_result: Optional[Dict[str, Any]] = None
        # Calibration configuration (table plane + assumed pixel spacing; replace spacing with metadata later).
        self._calibration_pixel_spacing_mm = self._current_pixel_spacing_mm.copy()
        self._plane_normal = TABLE_PLANE_NORMAL.astype(np.float64)
        self._plane_offset_m = float(TABLE_PLANE_OFFSET_M)
        self._min_calibration_samples = 8
        # Build UI and Napari viewers
        print("init UI")
        self._init_ui()
        print("init Layers")
        self._init_viewers_and_layers()
        # Timer for update_frame
        interval = int(1000 / GUI_FPS)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update_frame)
        self._timer.start(interval)
        self._closed = False
        from ui.widgets import ControlPanel #circular import, placed here to delay import 
        self.controls = ControlPanel(self)
        self.controls.record_btn.clicked.connect(self.toggle_recording)
    def quit_application(self) -> None:
        self._perform_shutdown()
        app = QApplication.instance()
        if app is not None:
            app.quit()
    def _init_viewers_and_layers(self) -> None:
        # 3D viewer
        model3d = ViewerModel(ndisplay=3)
        if hasattr(model3d.axes, "in_canvas"):
            model3d.axes.in_canvas = True
        self.volume_layer = model3d.add_image(self.volume, name='Ultrasound Volume')  # <--- save handle!
        model3d.dims.axis_labels = ('Z','Y','X')
        self.model3d = model3d
        self._apply_camera_defaults()
        qtviewer3d = QtViewer(model3d, show_welcome_screen=False)
        qtviewer3d.setParent(self)
        qtviewer3d.show()
        self.viewer3d = qtviewer3d
        self.grid.addWidget(qtviewer3d, 0, 0)
        # Live ultrasound image viewer (2D stream in its own pane)
        self.ultrasound_model = ViewerModel(ndisplay=2)
        if hasattr(self.ultrasound_model.axes, "in_canvas"):
            self.ultrasound_model.axes.in_canvas = True
        ultrasound_qt = QtViewer(self.ultrasound_model, show_welcome_screen=False)
        self.ultrasound_container = QWidget(self)
        ultrasound_layout = QVBoxLayout(self.ultrasound_container)
        ultrasound_layout.setContentsMargins(0, 0, 0, 0)
        ultrasound_layout.setSpacing(2)
        ultrasound_label = QLabel("Live Ultrasound", parent=self.ultrasound_container)
        ultrasound_label.setAlignment(Qt.AlignCenter)
        ultrasound_layout.addWidget(ultrasound_label)
        ultrasound_layout.addWidget(ultrasound_qt)
        ultrasound_qt.show()
        self.grid.addWidget(self.ultrasound_container, 0, 2)
        self._ultrasound_frame = np.zeros(ultrasoundImage.US_2D_SHAPE, dtype=np.float32)
        self.ultrasound_image_layer = self.ultrasound_model.add_image(
            self._ultrasound_frame,
            name='Live Ultrasound',
            colormap='gray',
        )
        self.ultrasound_viewer = ultrasound_qt
        pane_visible = True
        if hasattr(self, "ultrasound_pane_chk"):
            pane_visible = self.ultrasound_pane_chk.isChecked()
        if self.ultrasound_container is not None:
            self.ultrasound_container.setVisible(pane_visible)
        if self.ultrasound_image_layer is not None:
            self.ultrasound_image_layer.visible = pane_visible
        # Live US slice layer (3D transformed image)
        self.live_slice = np.zeros((1, self.volume.shape[1], self.volume.shape[2]), dtype=np.float32)
        self.live_slice_spacing = self._voxel_size_mm.copy() * float(ULTRASOUND_PLANE_VIS_SCALE)
        self.live_slice_anchor_idx = np.array(
            [0.0, 0.0, (self.live_slice.shape[2] - 1.0) / 2.0], dtype=np.float32
        )
        self.live_slice_layer = self.model3d.add_image(
            self.live_slice,
            name='Live US Slice',
            colormap='gray',
            opacity=1.0,
            blending='additive',
            affine=np.eye(4)
        )
        # 2D slice viewers
        self.slice_viewers = {}
        self.slice_layers = {}
        self._max_markers = 5
        self.arrow_data = np.zeros((1, 2, 3), dtype=np.float32)
        self.marker_data = np.zeros((self._max_markers, 3), dtype=np.float32)
        for key in SLICE_AXES:
            pos = SLICE_POSITIONS[key]
            axis = SLICE_AXES[key]
            model2d = ViewerModel(ndisplay=2)
            if hasattr(model2d.axes, "in_canvas"):
                model2d.axes.in_canvas = True
            qt2d = QtViewer(model2d, show_welcome_screen=False)
            container = QWidget(self)
            container_layout = QVBoxLayout(container)
            container_layout.setContentsMargins(0, 0, 0, 0)
            container_layout.setSpacing(2)
            label = QLabel(f"{key.capitalize()} Plane", parent=container)
            label.setAlignment(Qt.AlignCenter)
            container_layout.addWidget(label)
            container_layout.addWidget(qt2d)
            qt2d.show()
            self.grid.addWidget(container, *pos)
            # Store both!
            self.slice_viewers[key] = {"model": model2d, "qt": qt2d}
            img = Image(self.volume, name=key.capitalize())
            pts = Points(np.zeros((self._max_markers, 2), dtype=np.float32), name=f"{key}_pt", size=8)
            vec = Vectors(np.zeros((1, 2, 2), dtype=np.float32), name=f"{key}_vec", vector_style="arrow", edge_width=2.0)
            if hasattr(self, "us_arrow_chk"):
                vec.visible = self.us_arrow_chk.isChecked()
            row_axis, col_axis = self._slice_axes_indices(key)
            model2d.add_layer(img)
            model2d.add_layer(pts)
            model2d.add_layer(vec)
            if key == "axial":
                model2d.dims.order = (0, 1, 2)
                model2d.dims.axis_labels = ("Z", "Y", "X")
            elif key == "coronal":
                model2d.dims.order = (1, 0, 2)
                model2d.dims.axis_labels = ("Y", "Z", "X")
            elif key == "sagittal":
                model2d.dims.order = (2, 0, 1)
                model2d.dims.axis_labels = ("X", "Z", "Y")
            self.slice_layers[key] = {
                "img": img,
                "pts": pts,
                "probe_vec": vec,
                "axes": (row_axis, col_axis),
            }
            self.recon_slice_layers[key] = None
        # 3D overlay layers
        vec_width = 6.0
        self.ground_outline_mesh = self._build_ground_outline_mesh(
            half_extent_mm=GROUND_OUTLINE_HALF_SIZE_MM,
            thickness_mm=GROUND_THICKNESS_MM,
            y_mm=0.0,
        )
        self.ground_outline_layer = self.model3d.add_surface(
            self.ground_outline_mesh,
            name='Ground Outline',
            colormap='gray',
            contrast_limits=(0.0, 1.0),
            shading='flat',
            opacity=0.5,
            blending='translucent',
        )
        self.ground_axes_meshes = self._build_ground_axes_meshes(
            half_extent_mm=GROUND_OUTLINE_HALF_SIZE_MM,
            thickness_mm=GROUND_THICKNESS_MM,
            y_mm=0.0,
        )
        self.ground_axes_layers = []
        axes_colors = ['red', 'blue']
        axes_names = ['Ground X Axis', 'Ground Z Axis']
        for mesh, color, name in zip(self.ground_axes_meshes, axes_colors, axes_names):
            layer = self.model3d.add_surface(
                mesh,
                name=name,
                colormap=color,
                contrast_limits=(0.0, 1.0),
                shading='smooth',
                opacity=0.85,
                blending='translucent',
            )
            self.ground_axes_layers.append(layer)
        self.vec_layer = self.model3d.add_vectors(
            self.arrow_data,
            name='Ultrasound Arrow',
            edge_width=vec_width,
            vector_style='arrow',
            visible=self.us_arrow_chk.isChecked() if hasattr(self, "us_arrow_chk") else True,
        )
        self.needle_vec_layer = self.model3d.add_vectors(
            self.needle_arrow_data,
            name='Needle Line',
            edge_width=vec_width,
            vector_style='line',
            edge_color='yellow',
            visible=self.needle_arrow_chk.isChecked() and self._needle_has_pose if hasattr(self, "needle_arrow_chk") else False,
        )
        self.marker_layer = self.model3d.add_points(self.marker_data[:0], ndim=3, size=5.0, name='Markers')
        self.needle_tip_layer = self.model3d.add_points(
            np.zeros((0, 3), dtype=np.float32),
            ndim=3,
            size=6.0,
            name='Needle Tip Markers',
            face_color='red',
        )
        if hasattr(self.needle_tip_layer, "edge_color"):
            self.needle_tip_layer.edge_color = 'white'
    def _apply_camera_defaults(self) -> None:
        """Configure the 3D napari camera to the preferred baseline view."""
        if self.model3d is None:
            return
        self.model3d.theme = "dark"
        self.model3d.dims.ndisplay = 3
        # Force axes overlay to render in the corner (not in-volume).
        if hasattr(self.model3d.axes, "in_canvas"):
            self.model3d.axes.in_canvas = True
        # Hide in-volume axes; rely on the standard corner orientation widget instead.
        self.model3d.axes.visible = False
        if hasattr(self.model3d, "scale_bar"):
            self.model3d.scale_bar.visible = True
        # Start from (-X, +Y, 0) looking slightly down toward the origin.
        # Napari's camera uses center + angles to derive the eye position.
        self.model3d.camera.center = (0.0, 0.0, 0.0)
        # Angles are (elevation, azimuth, roll) in degrees.
        self.model3d.camera.angles = (180.0, 0.0, -225.0)
        self.model3d.camera.zoom = 1.25
    @staticmethod
    def _build_prism_mesh_for_segment(start: np.ndarray, end: np.ndarray, thickness_mm: float) -> Tuple[np.ndarray, np.ndarray]:
        """Extrude a line segment into a thin rectangular prism mesh."""
        start = np.asarray(start, dtype=np.float32)
        end = np.asarray(end, dtype=np.float32)
        direction = end - start
        length = float(np.linalg.norm(direction))
        if length <= 0.0:
            raise ValueError("Segment length must be positive for prism mesh")
        dir_unit = direction / length
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        if np.isclose(np.abs(np.dot(dir_unit, up)), 1.0, atol=1e-3):
            up = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        side1 = np.cross(dir_unit, up)
        side1 /= np.linalg.norm(side1)
        side2 = np.cross(dir_unit, side1)
        half = float(max(thickness_mm, 0.01)) * 0.5
        offset_a = side1 * half
        offset_b = side2 * half
        corners_start = np.array(
            [
                start + offset_a + offset_b,
                start + offset_a - offset_b,
                start - offset_a - offset_b,
                start - offset_a + offset_b,
            ],
            dtype=np.float32,
        )
        corners_end = corners_start + direction
        vertices = np.vstack([corners_start, corners_end]).astype(np.float32)
        faces = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
                [4, 6, 5],
                [4, 7, 6],
                [0, 4, 5],
                [0, 5, 1],
                [1, 5, 6],
                [1, 6, 2],
                [2, 6, 7],
                [2, 7, 3],
                [3, 7, 4],
                [3, 4, 0],
            ],
            dtype=np.int32,
        )
        return vertices, faces
    @staticmethod
    def _combine_meshes(meshes: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
        vertices_list: List[np.ndarray] = []
        faces_list: List[np.ndarray] = []
        offset = 0
        for verts, faces in meshes:
            vertices_list.append(verts)
            faces_list.append(faces + offset)
            offset += verts.shape[0]
        if not vertices_list:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32)
        vertices = np.concatenate(vertices_list, axis=0)
        faces = np.concatenate(faces_list, axis=0)
        return vertices.astype(np.float32), faces.astype(np.int32)
    @classmethod
    def _build_ground_outline_mesh(cls, half_extent_mm: float, thickness_mm: float, y_mm: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Wrap the ground outline in a set of low-height prisms so it stays visible."""
        half = max(0.0, float(half_extent_mm))
        y = float(y_mm)
        corners = np.array(
            [
                [-half, y, -half],
                [half, y, -half],
                [half, y, half],
                [-half, y, half],
            ],
            dtype=np.float32,
        )
        segments = [
            (corners[0], corners[1]),
            (corners[1], corners[2]),
            (corners[2], corners[3]),
            (corners[3], corners[0]),
        ]
        meshes = [cls._build_prism_mesh_for_segment(start, end, thickness_mm) for start, end in segments]
        vertices, faces = cls._combine_meshes(meshes)
        values = np.ones((vertices.shape[0],), dtype=np.float32)
        return vertices, faces, values
    @classmethod
    def _build_ground_axes_meshes(cls, half_extent_mm: float, thickness_mm: float, y_mm: float = 0.0) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Build thickened axis bars along +X and +Z on the ground plane."""
        half = max(0.0, float(half_extent_mm))
        y = float(y_mm)
        start_x = np.array([-half, y, 0.0], dtype=np.float32)
        end_x = np.array([half, y, 0.0], dtype=np.float32)
        start_z = np.array([0.0, y, -half], dtype=np.float32)
        end_z = np.array([0.0, y, half], dtype=np.float32)
        meshes: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for start, end in ((start_x, end_x), (start_z, end_z)):
            verts, faces = cls._build_prism_mesh_for_segment(start, end, thickness_mm)
            values = np.ones((verts.shape[0],), dtype=np.float32)
            meshes.append((verts, faces, values))
        return meshes
    @staticmethod
    def _pose_to_matrix(pos_m: np.ndarray, rot: R) -> np.ndarray:
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = rot.as_matrix().astype(np.float32)
        T[:3, 3] = pos_m.astype(np.float32)
        return T
    @staticmethod
    def _matrix_to_pose(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rot_block = np.asarray(T[:3, :3], dtype=np.float64)
        det = np.linalg.det(rot_block)
        if not np.isfinite(det) or det <= 0.0:
            try:
                u, _, vh = np.linalg.svd(rot_block)
                rot_block = u @ vh
                if np.linalg.det(rot_block) < 0.0:
                    u[:, -1] *= -1.0
                    rot_block = u @ vh
                logger.warning(
                    "Corrected an invalid rotation matrix (det=%.5f) to the nearest proper rotation.",
                    float(det) if np.isfinite(det) else float("nan"),
                )
            except Exception:
                logger.exception(
                    "Failed to orthonormalize rotation matrix; falling back to identity."
                )
                rot_block = np.eye(3, dtype=np.float64)
        rot = R.from_matrix(rot_block)
        pos = T[:3, 3]
        return pos.astype(np.float32), rot.as_quat().astype(np.float32)
    def _update_pixel_spacing(self, spacing_mm: Optional[Tuple[float, float]]) -> None:
        """Refresh cached pixel spacing (row_mm, col_mm) and derived scales when metadata is available."""
        if spacing_mm is None:
            return
        try:
            spacing_arr = np.asarray(spacing_mm, dtype=np.float64).reshape(2)
        except Exception:
            return
        if not np.all(np.isfinite(spacing_arr)) or np.any(spacing_arr <= 0.0):
            return
        self._current_pixel_spacing_mm = spacing_arr
        self._calibration_pixel_spacing_mm = spacing_arr.copy()
        slice_thickness_mm = float(np.mean(spacing_arr))
        self._voxel_size_mm = np.array(
            [slice_thickness_mm, spacing_arr[0], spacing_arr[1]], dtype=np.float32
        )
        self.live_slice_spacing = self._voxel_size_mm * float(ULTRASOUND_PLANE_VIS_SCALE)
        px_per_mm_x = 1.0 / spacing_arr[1]
        px_per_mm_y = 1.0 / spacing_arr[0]
        self.PX_PER_MM = (px_per_mm_x, px_per_mm_y)
    def update_frame(self) -> None:
        if not self._timer.isActive():
            return
        self._timer.stop()
        try:
            packets: List[RenderPacket] = []
            while True:
                try:
                    pkt = self.q.get_nowait()
                except queue.Empty:
                    break
                packets.append(pkt)
            if not packets:
                return
            packet = packets[-1]
            frame2d = getattr(packet, "frame", None)
            if (
                frame2d is None
                or not isinstance(frame2d, np.ndarray)
                or frame2d.ndim != 2
                or frame2d.dtype != np.float32
            ):
                return
            self._update_pixel_spacing(getattr(packet, "pixel_spacing_mm", None))
            spacing_snapshot: Optional[Tuple[float, float]] = None
            if getattr(self, "_current_pixel_spacing_mm", None) is not None:
                spacing_snapshot = tuple(float(x) for x in self._current_pixel_spacing_mm)
            self.us_archive.append((packet.frame_ts, frame2d, packet.drop_count))
            if packet.pose_ts is not None:
                self.cam_archive.append(
                    (packet.pose_ts, packet.poses, packet.markers, packet.pose_drop_count)
                )
            poses = packet.poses if packet.poses is not None else {}
            markers = packet.markers if packet.markers is not None else []
            us_pose = poses.get(ULTRASOUND_RB_ID) if isinstance(poses, dict) else None
            if (
                us_pose is None
                or us_pose[0] is None
                or us_pose[1] is None
            ):
                return
            pos, rot = us_pose
            pos_arr = np.asarray(pos, dtype=np.float32)
            quat_arr = np.asarray(rot, dtype=np.float64)
            raw_rot = R.from_quat(quat_arr)
            pose_matrix = self._pose_to_matrix(pos_arr, raw_rot)
            if self._us_calibration_recording:
                self._capture_ultrasound_calibration_sample(packet.frame_ts, frame2d, pose_matrix)
            corrected_matrix = pose_matrix @ self.us_correction_matrix
            corrected_pos, corrected_quat = self._matrix_to_pose(corrected_matrix)
            pose = Pose.from_pos_rot(
                corrected_pos.astype(np.float32),
                corrected_quat.astype(np.float32),
            )
            self._last_raw_pos_m = pos_arr
            self._last_raw_rot = raw_rot
            if hasattr(self, 'calibrate_us_btn'):
                self.calibrate_us_btn.setEnabled(True)
            if hasattr(self, "simple_us_origin_btn"):
                self.simple_us_origin_btn.setEnabled(True)
            if hasattr(self, "simple_us_orient_btn"):
                self.simple_us_orient_btn.setEnabled(True)
            if self.recording:
                if self._last_recorded_us_ts is None or not np.isclose(self._last_recorded_us_ts, packet.frame_ts):
                    frame_copy = frame2d.copy()
                    pose_copy = Pose.from_pos_rot(
                        pose.pos_m.copy(),
                        pose.quat.copy(),
                        pose.align_quat.copy() if pose.align_quat is not None else None,
                    )
                    self.recorded_frames.append((packet.frame_ts, frame_copy, pose_copy, spacing_snapshot))
                    self._last_recorded_us_ts = packet.frame_ts
            self._update_rigid_body_status(poses)
            pose.arrow(out=self.arrow_data)
            self.vec_layer.data = self.arrow_data
            if self.vec_layer is not None:
                self.vec_layer.visible = self.us_arrow_chk.isChecked()
            needle_entry = poses.get(NEEDLE_RB_ID)
            if (
                needle_entry is not None
                and needle_entry[0] is not None
                and needle_entry[1] is not None
            ):
                needle_pos, needle_rot = needle_entry
                needle_pos_arr = np.asarray(needle_pos, dtype=np.float32)
                needle_quat_arr = np.asarray(needle_rot, dtype=np.float64)
                raw_needle_rot = R.from_quat(needle_quat_arr)
                self._last_raw_needle_pos_m = needle_pos_arr
                self._last_raw_needle_rot = raw_needle_rot
                if hasattr(self, "needle_pivot_btn"):
                    self.needle_pivot_btn.setEnabled(True)
                    if not self._pivot_recording:
                        self._update_needle_pivot_status("ready")
                needle_pose_matrix = self._pose_to_matrix(needle_pos_arr, raw_needle_rot)
                corrected_top_matrix = self.needle_correction_matrix @ needle_pose_matrix
                tip_matrix = self._needle_tip_matrix
                corrected_tip_matrix = (
                    corrected_top_matrix @ tip_matrix if tip_matrix is not None else corrected_top_matrix
                )
                top_pos, top_quat = self._matrix_to_pose(corrected_top_matrix)
                tip_pos, tip_quat = self._matrix_to_pose(corrected_tip_matrix)
                top_pose = Pose.from_pos_rot(top_pos.astype(np.float32), top_quat.astype(np.float32))
                tip_pose = Pose.from_pos_rot(tip_pos.astype(np.float32), tip_quat.astype(np.float32))
                start_mm = top_pose.pos_mm.astype(np.float32)
                tip_mm = tip_pose.pos_mm.astype(np.float32)
                vec_mm = tip_mm - start_mm
                if np.linalg.norm(vec_mm) < 1e-6:
                    vec_mm = (top_pose.body_rot * top_pose.align_rot).apply(LOCAL_ARROW_DIR)
                    vec_mm = (vec_mm * NEEDLE_ARROW_LENGTH_MM).astype(np.float32)
                self.needle_arrow_data[0, 0, :] = start_mm
                self.needle_arrow_data[0, 1, :] = vec_mm.astype(np.float32)
                if self.needle_vec_layer is not None:
                    self.needle_vec_layer.data = self.needle_arrow_data
                    self.needle_vec_layer.visible = self.needle_arrow_chk.isChecked()
                self._needle_has_pose = True
                self._current_needle_pose = tip_pose
                self._current_needle_top_pose = top_pose
                if hasattr(self, "place_needle_tip_btn"):
                    self.place_needle_tip_btn.setEnabled(True)
                if self._pivot_recording:
                    self._record_pivot_sample(top_pose, packet.frame_ts)
            else:
                self._needle_has_pose = False
                self._current_needle_pose = None
                self._current_needle_top_pose = None
                if hasattr(self, "place_needle_tip_btn"):
                    self.place_needle_tip_btn.setEnabled(False)
                if self.needle_vec_layer is not None:
                    self.needle_vec_layer.visible = False
                if hasattr(self, "needle_pivot_btn"):
                    if self._pivot_recording:
                        self.needle_pivot_btn.setEnabled(True)
                        self._update_needle_pivot_status("no pose")
                    else:
                        self.needle_pivot_btn.setEnabled(False)
                        self._update_needle_pivot_status("no pose")
            self._last_pos_m = pose.pos_m
            self._last_body_rot = pose.body_rot
            markers_list = markers if markers is not None else []
            self._last_marker_m = markers_list[0] if markers_list else None
            # --- 3D ultrasound slice rendering ---
            if self.ultrasound_image_layer is not None:
                if self._ultrasound_frame is None or self._ultrasound_frame.shape != frame2d.shape:
                    self._ultrasound_frame = frame2d.copy()
                else:
                    np.copyto(self._ultrasound_frame, frame2d)
                self.ultrasound_image_layer.data = self._ultrasound_frame
            np.copyto(self.live_slice[0], frame2d)
            T = np.eye(4, dtype=np.float32)
            rotation = (pose.body_rot * pose.align_rot).as_matrix().astype(np.float32)
            scale_matrix = np.diag(self.live_slice_spacing)
            T[:3, :3] = rotation @ scale_matrix
            anchor_offset_mm = scale_matrix @ self.live_slice_anchor_idx
            T[:3, 3] = pose.pos_mm - rotation @ anchor_offset_mm
            self.live_slice_layer.data = self.live_slice
            self.live_slice_layer.affine = T
            # --- Markers ---
            n_markers = len(markers_list)
            if n_markers > self._max_markers:
                self._max_markers = max(n_markers, self._max_markers * 2)
                self.marker_data = np.zeros((self._max_markers, 3), dtype=np.float32)
            if n_markers > 0:
                pose.markers_mm(markers_list, out=self.marker_data[:n_markers])
                if self.marker_layer.data.shape == (n_markers, 3):
                    self.marker_layer.data[:] = self.marker_data[:n_markers]
                    if hasattr(self.marker_layer, "refresh"):
                        self.marker_layer.refresh()
                else:
                    self.marker_layer.data = self.marker_data[:n_markers]
            else:
                self.marker_layer.data = np.zeros((0, 3), dtype=np.float32)
            # --- 2D overlays ---
            for key, viewer in self.slice_viewers.items():
                layers = self.slice_layers[key]
                pts_layer = layers["pts"]
                model2d = viewer["model"]
                axis = SLICE_AXES[key]
                slice_idx = self._clamp_slice_index(model2d, axis)
                slice_mm = slice_idx * float(self._voxel_size_mm[axis])
                half_voxel_mm = float(self._voxel_size_mm[axis]) * 0.5
                pts = []
                for m in self.marker_data[:n_markers]:
                    if abs(float(m[axis]) - slice_mm) <= half_voxel_mm:
                        pts.append(self.project_to_slice(key, m))
                if pts:
                    pts_arr = np.array(pts, dtype=np.float32)
                    if pts_layer.data.shape == pts_arr.shape:
                        pts_layer.data[:] = pts_arr
                    else:
                        pts_layer.data = pts_arr
                else:
                    pts_layer.data = np.zeros((0, 2), dtype=np.float32)
        except Exception as e:
            print("Exception in update_frame:", e)
            import traceback; traceback.print_exc()
        finally:
            self._timer.start(int(1000 / GUI_FPS))
    @staticmethod
    def _slice_axes_indices(key: str) -> Tuple[int, int]:
        if key == "axial":
            return 1, 2
        if key == "coronal":
            return 0, 2
        if key == "sagittal":
            return 0, 1
        raise ValueError(f"Unknown slice key: {key}")
    def _clamp_slice_index(self, model: ViewerModel, axis: int) -> float:
        volume = getattr(self, "volume", None)
        if isinstance(volume, np.ndarray):
            volume_shape = volume.shape
        else:
            volume_shape = VOLUME_SHAPE
        axis_limit = float(max(volume_shape[axis] - 1, 0))
        current = float(model.dims.current_step[axis])
        if not np.isfinite(current):
            current = 0.0
        clamped = min(max(current, 0.0), axis_limit)
        if clamped != current:
            model.dims.set_point(axis, clamped)
        return clamped
    def _update_rigid_body_status(self, poses: Dict[int, Tuple[Optional[np.ndarray], Optional[np.ndarray]]]) -> None:
        if not hasattr(self, "rb_status_label"):
            return
        streaming_ids: List[int] = []
        for rigid_id, entry in poses.items():
            if entry is None:
                continue
            pos, rot = entry
            if pos is not None and rot is not None:
                streaming_ids.append(int(rigid_id))
        streaming_ids.sort()
        if streaming_ids:
            text = "Streaming rigid bodies: " + ", ".join(str(rid) for rid in streaming_ids)
        else:
            text = "Streaming rigid bodies: (none)"
        if text != self._last_rb_status_text:
            self.rb_status_label.setText(text)
            self._last_rb_status_text = text
    def toggle_probe_arrow(self, state: int) -> None:
        visible = bool(state)
        if self.vec_layer is not None:
            self.vec_layer.visible = visible
        for layers in self.slice_layers.values():
            layers["probe_vec"].visible = visible
    def toggle_needle_arrow(self, state: int) -> None:
        visible = bool(state)
        if self.needle_vec_layer is not None:
            self.needle_vec_layer.visible = visible and self._needle_has_pose
    def add_needle_tip_marker(self) -> None:
        pose = self._current_needle_pose
        if pose is None:
            logger.warning("Cannot add needle tip marker: no needle pose available.")
            return
        tip_mm = pose.pos_mm.astype(np.float32).reshape(1, 3)
        self._needle_tip_points.append(tip_mm[0].copy())
        if self.needle_tip_layer is not None:
            data = np.asarray(self._needle_tip_points, dtype=np.float32)
            self.needle_tip_layer.data = data
    def toggle_needle_slice_overlay(self, state: int) -> None:
        _ = state
    def toggle_markers(self, state: int) -> None:
        visible = bool(state)
        self.marker_layer.visible = visible
        for layers in self.slice_layers.values():
            layers["pts"].visible = visible
    def toggle_ultrasound(self, state: int) -> None:
        visible = bool(state)
        if getattr(self, "volume_layer", None) is not None:
            self.volume_layer.visible = visible
        if getattr(self, "live_slice_layer", None) is not None:
            self.live_slice_layer.visible = visible
        for layers in self.slice_layers.values():
            img_layer = layers.get("img")
            if img_layer is not None:
                img_layer.visible = visible
    def toggle_ultrasound_pane(self, state: int) -> None:
        visible = bool(state)
        if self.ultrasound_image_layer is not None:
            self.ultrasound_image_layer.visible = visible
        if self.ultrasound_container is not None:
            self.ultrasound_container.setVisible(visible)
    def toggle_recon_volume(self, state: int) -> None:
        visible = bool(state)
        if self.recon_volume_layer is not None and self._has_recon_volume:
            self.recon_volume_layer.visible = visible
        for layer in self.recon_slice_layers.values():
            if layer is not None:
                layer.visible = visible
    def toggle_axes(self, state: int) -> None:
        visible = bool(state)
        if self.model3d is not None:
            if hasattr(self.model3d.axes, "in_canvas"):
                self.model3d.axes.in_canvas = True
            self.model3d.axes.visible = visible
        # Keep 2D and ultrasound viewers free of axes overlays.
        for viewer in self.slice_viewers.values():
            model2d = viewer.get("model")
            if model2d is not None:
                model2d.axes.visible = False
        if self.ultrasound_model is not None:
            self.ultrasound_model.axes.visible = False
    def toggle_recording(self) -> None:
        self.recording = not self.recording
        if self.recording:
            self.recorded_frames.clear()
            self._last_recorded_us_ts = None
            self.record_btn.setText("Stop Recording")
            logger.info("Recording started.")
        else:
            self.record_btn.setText("Start Recording")
            self._last_recorded_us_ts = None
            logger.info(f"Recording stopped. {len(self.recorded_frames)} frames captured.")
            try:
                saved_path = self._save_recording_to_disk()
                if saved_path is not None:
                    logger.info(f"Recording saved to {saved_path}")
            except Exception:
                logger.exception("Failed to save recording to disk")
    def _save_recording_to_disk(self) -> Optional[Path]:
        if not self.recorded_frames:
            return None
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = RECORDINGS_DIR / f"us_recording_{timestamp}.csv"
        fieldnames = [
            "timestamp",
            "pos_x_mm",
            "pos_y_mm",
            "pos_z_mm",
            "quat_x",
            "quat_y",
            "quat_z",
            "quat_w",
            "align_qx",
            "align_qy",
            "align_qz",
            "align_qw",
            "align_present",
            "frame_height",
            "frame_width",
            "pixel_spacing_row_mm",
            "pixel_spacing_col_mm",
            "slice_thickness_mm",
            "frame_base64",
        ]
        with out_path.open("w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for rec in self.recorded_frames:
                if len(rec) >= 4:
                    ts, frame, pose, spacing_mm = rec
                else:
                    ts, frame, pose = rec
                    spacing_mm = None
                spacing_row_mm, spacing_col_mm = PIXEL_SPACING_MM
                spacing_source = spacing_mm
                if spacing_source is None:
                    spacing_source = getattr(self, "_current_pixel_spacing_mm", None)
                if spacing_source is None:
                    spacing_source = getattr(self, "_calibration_pixel_spacing_mm", None)
                try:
                    spacing_arr = np.asarray(spacing_source, dtype=np.float64).reshape(2)
                    if np.all(np.isfinite(spacing_arr)) and np.all(spacing_arr > 0.0):
                        spacing_row_mm = float(spacing_arr[0])
                        spacing_col_mm = float(spacing_arr[1])
                except Exception:
                    spacing_row_mm, spacing_col_mm = PIXEL_SPACING_MM
                slice_thickness_mm = float(np.mean((spacing_row_mm, spacing_col_mm)))
                pos = pose.pos_mm.astype(np.float32)
                quat = pose.quat.astype(np.float32)
                if pose.align_quat is None:
                    align = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
                    align_present = False
                else:
                    align = pose.align_quat.astype(np.float32)
                    align_present = True
                frame_f32 = frame.astype(np.float32, copy=False)
                frame_bytes = frame_f32.tobytes(order="C")
                frame_b64 = base64.b64encode(frame_bytes).decode("ascii")
                writer.writerow(
                    {
                        "timestamp": f"{ts:.6f}" if ts is not None else "",
                        "pos_x_mm": f"{pos[0]:.6f}",
                        "pos_y_mm": f"{pos[1]:.6f}",
                        "pos_z_mm": f"{pos[2]:.6f}",
                        "quat_x": f"{quat[0]:.6f}",
                        "quat_y": f"{quat[1]:.6f}",
                        "quat_z": f"{quat[2]:.6f}",
                        "quat_w": f"{quat[3]:.6f}",
                        "align_qx": f"{align[0]:.6f}",
                        "align_qy": f"{align[1]:.6f}",
                        "align_qz": f"{align[2]:.6f}",
                        "align_qw": f"{align[3]:.6f}",
                        "align_present": "1" if align_present else "0",
                        "frame_height": frame_f32.shape[0],
                        "frame_width": frame_f32.shape[1],
                        "pixel_spacing_row_mm": f"{spacing_row_mm:.6f}",
                        "pixel_spacing_col_mm": f"{spacing_col_mm:.6f}",
                        "slice_thickness_mm": f"{slice_thickness_mm:.6f}",
                        "frame_base64": frame_b64,
                    }
                )
        return out_path
    def reconstruct_volume_from_recording(self) -> None:
        if not getattr(self, "recorded_frames", None) or len(self.recorded_frames) == 0:
            self.log.info("Reconstruction skipped: no recorded frames.")
            return
        px_per_mm = getattr(self, "PX_PER_MM", None)
        if px_per_mm is None:
            spacing_mm = None
            first_rec = self.recorded_frames[0]
            if len(first_rec) >= 4:
                spacing_mm = first_rec[3]
            if spacing_mm is None:
                spacing_mm = getattr(self, "_current_pixel_spacing_mm", None)
            try:
                spacing_arr = np.asarray(spacing_mm, dtype=np.float64).reshape(2)
                if np.all(np.isfinite(spacing_arr)) and np.all(spacing_arr > 0.0):
                    px_per_mm = (1.0 / float(spacing_arr[1]), 1.0 / float(spacing_arr[0]))
            except Exception:
                px_per_mm = None
        if px_per_mm is None:
            px_per_mm = globals().get("PIXELS_PER_MM", 10.0)
        vol, weights, meta = reconstruct_volume_from_frames(
            self.recorded_frames,
            px_per_mm=px_per_mm,
            voxel_size_mm=0.75,
            pixel_step=1,
            frame_step=1,
            inplane_origin="topcenter",
        )
        self.last_volume_zyx = vol
        self.last_weights_zyx = weights
        self.last_recon_meta = meta
        self.last_recon_affine = self._build_recon_affine(meta)
        self._has_recon_volume = True
        if hasattr(self, "recon_volume_chk"):
            self.recon_volume_chk.setEnabled(True)
        if self.recon_volume_layer is None:
            self.recon_volume_layer = self.model3d.add_image(
                vol,
                name="Reconstructed Volume",
                opacity=0.6,
                blending="translucent",
                visible=self.recon_volume_chk.isChecked(),
                affine=self.last_recon_affine,
            )
        else:
            self.recon_volume_layer.data = vol
            self.recon_volume_layer.affine = self.last_recon_affine
            self.recon_volume_layer.visible = self.recon_volume_chk.isChecked()
        self._update_recon_slice_layers(vol)
        nonzero = int(np.count_nonzero(weights)) if weights is not None else 0
        self.log.info(
            "3D recon complete: frames=%d, shape(Z,Y,X)=%s, voxel=%.3f mm, origin(mm)=%s, nonzero weights=%d",
            len(self.recorded_frames),
            vol.shape,
            meta["voxel_size_mm"],
            np.array2string(meta["origin_mm"], precision=3),
            nonzero,
        )
    def _update_recon_slice_layers(self, volume: np.ndarray) -> None:
        if not hasattr(self, "recon_slice_layers"):
            self.recon_slice_layers = {}
        affine = (
            self.last_recon_affine
            if self.last_recon_affine is not None
            else np.eye(4, dtype=np.float32)
        )
        visible = self.recon_volume_chk.isChecked() if hasattr(self, "recon_volume_chk") else True
        for key, viewer in self.slice_viewers.items():
            model2d = viewer["model"]
            layer = self.recon_slice_layers.get(key)
            if layer is None:
                layer = Image(
                    volume,
                    name=f"{key.capitalize()} Recon",
                    colormap="gray",
                    opacity=0.6,
                    blending="translucent",
                    visible=visible,
                    affine=affine,
                )
                model2d.add_layer(layer)
                self.recon_slice_layers[key] = layer
            else:
                layer.data = volume
                layer.affine = affine
                layer.visible = visible
            if hasattr(layer, "reset_contrast_limits"):
                layer.reset_contrast_limits()
    @staticmethod
    def _build_recon_affine(meta: dict) -> np.ndarray:
        voxel = float(meta.get("voxel_size_mm", 1.0))
        origin = np.asarray(meta.get("origin_mm", (0.0, 0.0, 0.0)), dtype=np.float32)
        translate = np.array([origin[2], origin[1], origin[0]], dtype=np.float32)
        affine = np.eye(4, dtype=np.float32)
        affine[0, 0] = voxel
        affine[1, 1] = voxel
        affine[2, 2] = voxel
        affine[:3, 3] = translate
        return affine
    def reset_view(self) -> None:
        # Reset 3D viewer camera
        if hasattr(self, 'model3d') and self.model3d is not None:
            try:
                self.model3d.reset_view()
            except Exception as e:
                logger.warning(f"Error resetting 3D view: {e}", exc_info=True)
            else:
                self._apply_camera_defaults()
        # Reset 2D slice viewers
        for k, viewers in self.slice_viewers.items():
            model2d = viewers.get("model")
            if model2d is not None:
                try:
                    model2d.reset_view()
                except Exception as e:
                    logger.warning(f"Error resetting 2D view '{k}': {e}", exc_info=True)
        if self.ultrasound_model is not None:
            try:
                self.ultrasound_model.reset_view()
            except Exception as e:
                logger.warning("Error resetting live ultrasound view: %s", e, exc_info=True)
    def calibrate_ultrasound_transform(self) -> None:
        if not self._us_calibration_recording:
            if self._last_raw_pos_m is None or self._last_raw_rot is None:
                logger.warning("Cannot start ultrasound calibration without a recent pose.")
                return
            self._us_calibration_recording = True
            self._us_calibration_last_ts = None
            self._us_calibration_samples.clear()
            self.calibrate_us_btn.setText("Stop Ultrasound Calibration")
            logger.info("Ultrasound calibration recording started. Sweep the planar phantom.")
        else:
            self._us_calibration_recording = False
            self.calibrate_us_btn.setText("Calibrate Ultrasound Transform")
            try:
                result = self._solve_ultrasound_calibration()
            except Exception:
                logger.exception("Ultrasound calibration solve failed.")
                QMessageBox.critical(
                    self,
                    "Ultrasound Calibration Failed",
                    "Unable to compute a calibration transform. Check logs for details.",
                )
                self._clear_ultrasound_calibration_samples()
                return
            if result is None:
                QMessageBox.warning(
                    self,
                    "Ultrasound Calibration",
                    "Not enough valid samples were captured for calibration.",
                )
                self._clear_ultrasound_calibration_samples()
                return
            self._last_us_calibration_result = result
            tilt_text = (
                f"{result['tilt_degrees']:.2f}°" if result.get("tilt_degrees") is not None else "n/a"
            )
            spacing_text = result.get("assumed_pixel_spacing_mm")
            if spacing_text is not None:
                spacing_text = f"{spacing_text[0]:.4f}, {spacing_text[1]:.4f} mm/px"
            else:
                spacing_text = "unknown"
            summary_lines = [
                f"Samples: {result['num_samples']} (points: {result.get('num_points', 0)})",
                f"Residual RMS: {result['rms_mm']:.3f} mm (max {result['max_abs_mm']:.3f} mm)",
                f"Mean image-plane tilt vs table: {tilt_text}",
                f"Pose spread: {result.get('spread_xy_mm', 0.0):.1f} mm, tilt span {result.get('tilt_span_deg', 0.0):.1f}°",
                f"Pixel spacing used: {spacing_text}",
            ]
            QMessageBox.information(
                self,
                "Ultrasound Calibration Complete",
                "\n".join(summary_lines),
            )
            logger.info(
                "Ultrasound calibration complete with %d samples. RMS=%.4f mm, max=%.4f mm.",
                result["num_samples"],
                result["rms_mm"],
                result["max_abs_mm"],
            )
            self._clear_ultrasound_calibration_samples()
    def reset_ultrasound_transform(self) -> None:
        # Ultrasound calibration: T_probe_image (image -> probe); identity until calibrated.
        self.us_correction_matrix = np.eye(4, dtype=np.float32)
        logger.info("Ultrasound transform reset to identity.")
        self._clear_ultrasound_calibration_samples()

    def _apply_simple_us_calibration(self, rot_matrix: np.ndarray, label: str) -> None:
        """
        Quick calibration: force the ultrasound plane to lie in the YZ plane with rows pointing
        down toward -Y, and anchor the top-middle pixel at the world origin.
        """
        if self._last_raw_pos_m is None or self._last_raw_rot is None:
            logger.warning("Simple ultrasound calibration requires an active ultrasound pose.")
            QMessageBox.warning(
                self,
                "Simple Calibration",
                "No ultrasound pose is available. Wait for tracking before applying the simple calibration.",
            )
            return
        T_probe = np.eye(4, dtype=np.float64)
        try:
            T_probe[:3, :3] = self._last_raw_rot.as_matrix().astype(np.float64)
        except Exception:
            logger.exception("Failed to build probe rotation for simple calibration.")
            return
        try:
            T_probe[:3, 3] = np.asarray(self._last_raw_pos_m, dtype=np.float64).reshape(3)
        except Exception:
            logger.exception("Failed to build probe translation for simple calibration.")
            return
        T_target = np.eye(4, dtype=np.float64)
        T_target[:3, :3] = rot_matrix
        try:
            T_probe_inv = np.linalg.inv(T_probe)
        except np.linalg.LinAlgError:
            logger.error("Simple ultrasound calibration failed: probe pose is not invertible.")
            return
        correction = T_probe_inv @ T_target
        self.us_correction_matrix = correction.astype(np.float32)
        self._clear_ultrasound_calibration_samples()
        normal = rot_matrix[:, 2]
        x_axis = rot_matrix[:, 0]
        y_axis = rot_matrix[:, 1]
        logger.info(
            "Applied simple ultrasound calibration [%s]. x->(%.2f, %.2f, %.2f), y->(%.2f, %.2f, %.2f), normal=(%.2f, %.2f, %.2f), origin @ (0,0,0).",
            label,
            float(x_axis[0]),
            float(x_axis[1]),
            float(x_axis[2]),
            float(y_axis[0]),
            float(y_axis[1]),
            float(y_axis[2]),
            float(normal[0]),
            float(normal[1]),
            float(normal[2]),
        )

    def simple_calibrate_ultrasound_origin(self) -> None:
        """Anchor at (0,0,0) with normal +X (faces forward), rows -> -Z, cols -> +Y."""
        self._apply_simple_us_calibration(SIMPLE_US_ROT_MAT_FWD, label="face +X")

    def simple_calibrate_ultrasound_flipped(self) -> None:
        """Anchor at (0,0,0) with normal -X (flipped), rows -> -Z, cols -> +Y."""
        self._apply_simple_us_calibration(SIMPLE_US_ROT_MAT_REV, label="face -X")

    def _detect_plane_points(self, frame: np.ndarray, max_points: int = 200) -> np.ndarray:
        """
        Lightweight plane-line detector: pick the brightest pixel in each column,
        keep those above a relative threshold, and optionally downsample.
        Returns Nx2 array of (row, col) pixel coordinates (float32).
        """
        empty = np.zeros((0, 2), dtype=np.float32)
        if frame is None or frame.ndim != 2:
            return empty
        try:
            col_indices = np.arange(frame.shape[1])
            row_indices = np.argmax(frame, axis=0)
            max_vals = frame[row_indices, col_indices]
            global_max = float(np.max(max_vals))
            if not np.isfinite(global_max) or global_max <= 0.0:
                return empty
            # Keep only columns with strong response to reject noise.
            keep_mask = max_vals >= (0.35 * global_max)
            cols = col_indices[keep_mask]
            rows = row_indices[keep_mask]
            if cols.size == 0:
                return empty
            # Refine row estimate with an intensity-weighted centroid in a small vertical window
            window_radius = 3
            pts_list = []
            for c, r_peak in zip(cols.tolist(), rows.tolist()):
                r0 = max(0, int(r_peak) - window_radius)
                r1 = min(frame.shape[0], int(r_peak) + window_radius + 1)
                slice_vals = frame[r0:r1, int(c)].astype(np.float64)
                if slice_vals.size == 0:
                    continue
                # Shift to avoid all-zero weights if the slice is flat
                shifted = slice_vals - np.min(slice_vals)
                weights = shifted if np.any(shifted > 0.0) else slice_vals
                if np.max(weights) <= 0.0:
                    continue
                rows_window = np.arange(r0, r1, dtype=np.float64)
                row_refined = float(np.average(rows_window, weights=weights))
                pts_list.append((row_refined, float(c)))
            if not pts_list:
                return empty
            pts = np.array(pts_list, dtype=np.float32)
            if pts.shape[0] > max_points:
                idx = np.linspace(0, pts.shape[0] - 1, max_points).astype(np.int32)
                pts = pts[idx]
            return pts
        except Exception:
            logger.exception("Plane point detection failed; returning no points.")
            return empty

    def _image_points_to_metric(self, points_px: np.ndarray) -> np.ndarray:
        """
        Convert (row, col) pixel coordinates into metric coordinates (meters) in the
        ultrasound image frame. Pixel spacing is assumed fixed for now (see TODO).
        """
        if points_px.size == 0:
            return np.zeros((0, 3), dtype=np.float64)
        spacing_y_mm = float(self._calibration_pixel_spacing_mm[0])
        spacing_x_mm = float(self._calibration_pixel_spacing_mm[1])
        # Guard against invalid spacing by falling back to the compile-time constant.
        if spacing_y_mm <= 0.0 or spacing_x_mm <= 0.0:
            logger.warning("Invalid pixel spacing encountered; falling back to ASSUMED_US_PIXEL_SPACING_MM.")
            spacing_y_mm = float(ASSUMED_US_PIXEL_SPACING_MM[0])
            spacing_x_mm = float(ASSUMED_US_PIXEL_SPACING_MM[1])
        pts = np.zeros((points_px.shape[0], 3), dtype=np.float64)
        pts[:, 0] = points_px[:, 1] * spacing_x_mm / SCALE  # x/right
        pts[:, 1] = points_px[:, 0] * spacing_y_mm / SCALE  # y/down
        # z remains 0 (image plane)
        return pts

    def _pose_spread_metrics(self, samples: List[UltrasoundCalibrationSample]) -> Tuple[float, float]:
        """Return (xy_spread_mm, tilt_span_deg) across the captured probe poses."""
        if not samples:
            return 0.0, 0.0
        positions = np.array([s.pose_matrix[:3, 3] for s in samples], dtype=np.float64)
        spread_xy_mm = float(np.linalg.norm(np.ptp(positions[:, :2], axis=0)) * SCALE)
        tilts_deg: List[float] = []
        for s in samples:
            try:
                z_axis = s.pose_matrix[:3, :3][:, 2]
                z_axis /= np.linalg.norm(z_axis)
                tilt = float(np.degrees(np.arccos(np.clip(np.dot(z_axis, self._plane_normal), -1.0, 1.0))))
                tilts_deg.append(tilt)
            except Exception:
                continue
        tilt_span_deg = float(max(tilts_deg) - min(tilts_deg)) if tilts_deg else 0.0
        return spread_xy_mm, tilt_span_deg

    def _capture_ultrasound_calibration_sample(
        self, timestamp: float, frame: np.ndarray, pose_matrix: np.ndarray
    ) -> None:
        if not np.isfinite(timestamp):
            timestamp = float(time.perf_counter())
        if (
            self._us_calibration_last_ts is not None
            and np.isclose(self._us_calibration_last_ts, timestamp)
        ):
            return
        self._us_calibration_last_ts = float(timestamp)
        try:
            frame_copy = np.asarray(frame, dtype=np.float32).copy()
        except Exception:
            logger.warning("Skipping ultrasound calibration sample with invalid frame data.")
            return
        image_points = self._detect_plane_points(frame_copy)
        if image_points.size == 0:
            logger.warning("Skipping ultrasound calibration sample with no detected plane points.")
            return
        pose_copy = np.asarray(pose_matrix, dtype=np.float32).copy()
        self._us_calibration_samples.append(
            UltrasoundCalibrationSample(
                self._us_calibration_last_ts, frame_copy, pose_copy, image_points
            )
        )
        if len(self._us_calibration_samples) > 600:
            self._us_calibration_samples.pop(0)
        if len(self._us_calibration_samples) % 25 == 0:
            logger.info(
                "Captured %d ultrasound calibration samples...",
                len(self._us_calibration_samples),
            )

    def _clear_ultrasound_calibration_samples(self) -> None:
        self._us_calibration_samples.clear()
        self._us_calibration_last_ts = None
        self._us_calibration_recording = False
        if hasattr(self, "calibrate_us_btn"):
            self.calibrate_us_btn.setText("Calibrate Ultrasound Transform")

    def _solve_ultrasound_calibration(self) -> Optional[Dict[str, Any]]:
        samples = list(self._us_calibration_samples)
        if len(samples) < self._min_calibration_samples:
            logger.warning(
                "Ultrasound calibration requires at least %d samples; only %d collected.",
                self._min_calibration_samples,
                len(samples),
            )
            return None
        initial_points = int(
            sum(s.image_points_px.shape[0] for s in samples if s.image_points_px is not None)
        )
        if initial_points < 20:
            logger.warning(
                "Ultrasound calibration needs at least 20 plane points; collected %d.",
                initial_points,
            )
            return None
        spread_xy_mm, tilt_span_deg = self._pose_spread_metrics(samples)
        if spread_xy_mm < 5.0 or tilt_span_deg < 8.0:
            logger.warning(
                "Ultrasound calibration poses lack diversity (spread %.2f mm, tilt span %.2f°).",
                spread_xy_mm,
                tilt_span_deg,
            )
        plane_normal = self._plane_normal
        plane_offset_m = self._plane_offset_m

        def _params_to_matrix(params: np.ndarray) -> np.ndarray:
            rot = R.from_rotvec(params[:3]).as_matrix()
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = rot
            T[:3, 3] = params[3:]
            return T

        def _residuals(params: np.ndarray, subset: List[UltrasoundCalibrationSample]) -> np.ndarray:
            T_probe_image = _params_to_matrix(params)
            res_list: List[np.ndarray] = []
            for s in subset:
                pts_m = self._image_points_to_metric(s.image_points_px)
                if pts_m.shape[0] == 0:
                    continue
                pts_h = np.concatenate([pts_m, np.ones((pts_m.shape[0], 1), dtype=np.float64)], axis=1)
                world_pts = (s.pose_matrix.astype(np.float64) @ T_probe_image @ pts_h.T).T
                dist = world_pts[:, :3] @ plane_normal + plane_offset_m
                res_list.append(dist)
            if not res_list:
                return np.zeros((1,), dtype=np.float64)
            return np.concatenate(res_list)

        def _solve(subset: List[UltrasoundCalibrationSample]) -> Optional[np.ndarray]:
            init = np.zeros(6, dtype=np.float64)
            try:
                res = least_squares(
                    lambda p: _residuals(p, subset),
                    init,
                    loss="huber",
                    f_scale=0.002,  # ~2 mm robust threshold
                    max_nfev=200,
                )
            except Exception:
                logger.exception("Least-squares solve failed for ultrasound calibration.")
                return None
            if not res.success:
                logger.warning("Ultrasound calibration solver did not converge: %s", res.message)
                return None
            return res.x

        params = _solve(samples)
        if params is None:
            return None

        def _per_frame_stats(params_vec: np.ndarray, subset: List[UltrasoundCalibrationSample]) -> List[Dict]:
            stats: List[Dict[str, Any]] = []
            for idx, s in enumerate(subset):
                res_m = _residuals(params_vec, [s])
                if res_m.size == 0:
                    continue
                res_mm = res_m * SCALE
                stats.append(
                    {
                        "index": idx,
                        "timestamp": float(s.timestamp),
                        "points": int(s.image_points_px.shape[0]),
                        "rms_mm": float(np.sqrt(np.mean(res_mm**2))),
                        "max_mm": float(np.max(np.abs(res_mm))),
                    }
                )
            return stats

        frame_stats = _per_frame_stats(params, samples)
        outlier_thresh_mm = 3.0
        keep_mask = [st["rms_mm"] <= outlier_thresh_mm for st in frame_stats]
        if any(not k for k in keep_mask):
            filtered = [s for i, s in enumerate(samples) if i < len(keep_mask) and keep_mask[i]]
            if len(filtered) >= self._min_calibration_samples:
                logger.info(
                    "Dropping %d outlier frames (rms > %.1f mm) and re-solving.",
                    keep_mask.count(False),
                    outlier_thresh_mm,
                )
                params_filtered = _solve(filtered)
                if params_filtered is not None:
                    samples = filtered
                    params = params_filtered
                    frame_stats = _per_frame_stats(params, samples)
            else:
                logger.warning(
                    "Outlier rejection would leave too few samples (%d); keeping all frames.",
                    len(filtered),
                )

        T_probe_image = _params_to_matrix(params)
        overall_residuals_mm = _residuals(params, samples) * SCALE
        rms_mm = float(np.sqrt(np.mean(overall_residuals_mm**2)))
        max_abs_mm = float(np.max(np.abs(overall_residuals_mm)))
        total_points = int(
            sum(s.image_points_px.shape[0] for s in samples if s.image_points_px is not None)
        )

        normals = []
        for s in samples:
            n_img = s.pose_matrix[:3, :3].astype(np.float64) @ T_probe_image[:3, :3] @ np.array(
                [0.0, 0.0, 1.0], dtype=np.float64
            )
            n_norm = np.linalg.norm(n_img)
            if n_norm > 0:
                normals.append(n_img / n_norm)
        tilt_degrees = None
        if normals:
            mean_normal = np.mean(normals, axis=0)
            norm = np.linalg.norm(mean_normal)
            if norm > 0:
                mean_normal /= norm
                tilt_degrees = float(
                    np.degrees(np.arccos(np.clip(np.dot(mean_normal, plane_normal), -1.0, 1.0)))
                )

        self.us_correction_matrix = T_probe_image.astype(np.float32)
        return {
            "num_samples": len(samples),
            "num_points": total_points,
            "rms_mm": rms_mm,
            "max_abs_mm": max_abs_mm,
            "tilt_degrees": tilt_degrees,
            "spread_xy_mm": spread_xy_mm,
            "tilt_span_deg": tilt_span_deg,
            "assumed_pixel_spacing_mm": tuple(float(x) for x in self._calibration_pixel_spacing_mm),
            "correction_matrix": self.us_correction_matrix.copy(),
            "frame_stats": frame_stats,
        }
    def toggle_needle_pivot_calibration(self, checked: bool) -> None:
        if checked:
            if (
                not self._needle_has_pose
                or self._last_raw_needle_pos_m is None
                or self._last_raw_needle_rot is None
            ):
                logger.warning("Cannot start pivot calibration without an active needle pose.")
                self.needle_pivot_btn.blockSignals(True)
                self.needle_pivot_btn.setChecked(False)
                self.needle_pivot_btn.blockSignals(False)
                self._update_needle_pivot_status("no pose")
                return
            on_pivot_start()
            self._pivot_recording = True
            self._pivot_capture_count = 0
            self._pivot_last_sample_ts = None
            self._pivot_skipped_close_ts_count = 0
            self.needle_pivot_btn.setText("Stop Needle Pivot Calibration")
            self._update_needle_pivot_status("recording")
            logger.info("Needle pivot calibration recording started.")
        else:
            if not self._pivot_recording:
                self.needle_pivot_btn.setText("Start Needle Pivot Calibration")
                self._update_needle_pivot_status()
                return
            self._pivot_recording = False
            self.needle_pivot_btn.setText("Start Needle Pivot Calibration")
            self._pivot_last_sample_ts = None
            try:
                result = on_pivot_solve(
                    meta={
                        "source": "visualization",
                        "sample_count": _needle_pivot_calibrator.num_samples(),
                    }
                )
            except Exception as exc:
                logger.error("Needle pivot calibration failed: %s", exc, exc_info=True)
                self._update_needle_pivot_status("error")
                return
            self._pivot_capture_count = _needle_pivot_calibrator.num_samples()
            summary = result.get("summary")
            if summary:
                logger.info(summary)
            rms = result.get("rms_mm")
            if isinstance(rms, (int, float)):
                self._update_needle_pivot_status(f"RMS {rms:.2f} mm")
            else:
                self._update_needle_pivot_status("completed")
            tip = result.get("r_H_tip")
            if tip is not None:
                logger.info(
                    "Calibrated needle tip offset (mm): %s",
                    np.array2string(np.asarray(tip), precision=3),
                )
                self._apply_tip_transform_to_viewer(np.asarray(tip, dtype=np.float32))
            else:
                self._apply_tip_transform_to_viewer(None)
    def _record_pivot_sample(self, needle_pose: Pose, ts: Optional[float]) -> None:
        if not self._pivot_recording:
            return
        timestamp = None
        if ts is not None:
            try:
                timestamp = float(ts)
            except (TypeError, ValueError):
                timestamp = None
        if (
            timestamp is not None
            and self._pivot_last_sample_ts is not None
            and np.isclose(self._pivot_last_sample_ts, timestamp)
        ):
            self._pivot_skipped_close_ts_count += 1
            logger.info(
                "Pivot sample skipped #%d (ts close to last; dt=%.6f s)",
                self._pivot_skipped_close_ts_count,
                float(timestamp - self._pivot_last_sample_ts),
            )
            return
        try:
            on_pivot_capture(needle_pose, ts=timestamp)
        except Exception as exc:
            logger.error("Error capturing pivot sample: %s", exc, exc_info=True)
            self.needle_pivot_btn.blockSignals(True)
            self.needle_pivot_btn.setChecked(False)
            self.needle_pivot_btn.blockSignals(False)
            self.needle_pivot_btn.setText("Start Needle Pivot Calibration")
            self._pivot_recording = False
            self._update_needle_pivot_status("error")
            return
        self._pivot_capture_count += 1
        self._pivot_last_sample_ts = timestamp
        self._update_needle_pivot_status("recording")
        if self._pivot_capture_count % 10 == 0:
            logger.info("Recorded %d pivot samples.", self._pivot_capture_count)
    def _update_needle_pivot_status(self, status: Optional[str] = None) -> None:
        if not hasattr(self, "needle_pivot_status"):
            return
        if status is None:
            status = "recording" if self._pivot_recording else "idle"
            if not self._needle_has_pose and not self._pivot_recording:
                status = "no pose"
        self.needle_pivot_status.setText(f"Pivot samples: {self._pivot_capture_count} ({status})")
    def _apply_tip_transform_to_viewer(self, tip_offset: Optional[np.ndarray]) -> None:
        """Update viewer-side transform so the 3D needle aligns with the calibrated tip."""
        if tip_offset is None:
            if self._needle_tip_matrix is not None:
                logger.info("Cleared needle tip transform from 3D viewer.")
            self._needle_tip_offset = None
            self._needle_tip_matrix = None
            return
        tip_vec_mm = np.asarray(tip_offset, dtype=np.float32).reshape(3)
        if (
            self._needle_tip_offset is not None
            and np.allclose(self._needle_tip_offset, tip_vec_mm, atol=1e-6)
        ):
            return
        tip_matrix = np.eye(4, dtype=np.float32)
        tip_matrix[:3, 3] = tip_vec_mm / SCALE
        self._needle_tip_offset = tip_vec_mm
        self._needle_tip_matrix = tip_matrix
        logger.info(
            "Applied needle tip transform to 3D viewer (offset: %s).",
            np.array2string(tip_vec_mm, precision=4),
        )
    def clear_needle_tip_calibration(self) -> None:
        global _last_tip_offset_mm
        _last_tip_offset_mm = None
        _needle_pivot_calibrator.reset()
        self._apply_tip_transform_to_viewer(None)
        self._pivot_capture_count = 0
        self._pivot_last_sample_ts = None
        self._pivot_skipped_close_ts_count = 0
        if self._pivot_recording and hasattr(self, "needle_pivot_btn"):
            self.needle_pivot_btn.blockSignals(True)
            self.needle_pivot_btn.setChecked(False)
            self.needle_pivot_btn.blockSignals(False)
            self._pivot_recording = False
        if hasattr(self, "needle_pivot_btn"):
            self.needle_pivot_btn.setText("Start Needle Pivot Calibration")
        self._update_needle_pivot_status("cleared")
        logger.info("Needle tip calibration cleared.")
    def _calibration_payload(self) -> Dict[str, Any]:
        tip_offset = _last_tip_offset_mm.tolist() if _last_tip_offset_mm is not None else None
        pivot_result = _needle_pivot_calibrator._last_result
        serialized_pivot: Optional[Dict[str, Any]] = None
        if pivot_result is not None:
            serialized_pivot = {}
            for key, value in pivot_result.items():
                if isinstance(value, np.ndarray):
                    serialized_pivot[key] = value.tolist()
                elif isinstance(value, (np.floating, np.integer)):
                    serialized_pivot[key] = float(value)
                else:
                    serialized_pivot[key] = value
        payload: Dict[str, Any] = {
            "version": 1,
            "saved_at": datetime.utcnow().isoformat() + "Z",
            "us_correction_matrix": self.us_correction_matrix.astype(float).tolist(),
            "assumed_pixel_spacing_mm": tuple(float(x) for x in self._calibration_pixel_spacing_mm),
            "plane_normal": self._plane_normal.tolist(),
            "plane_offset_m": float(self._plane_offset_m),
            "needle_correction_matrix": self.needle_correction_matrix.astype(float).tolist(),
            "needle_tip_offset_mm": tip_offset,
            "pivot_sample_count": _needle_pivot_calibrator.num_samples(),
        }
        if serialized_pivot is not None:
            payload["pivot_result"] = serialized_pivot
        return payload
    def save_calibrations(self) -> None:
        default_name = CALIBRATIONS_DIR / f"calibration_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Calibration",
            str(default_name),
            "Calibration Files (*.json);;All Files (*)",
        )
        if not path:
            return
        try:
            payload = self._calibration_payload()
            target_path = Path(path)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with open(target_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
        except Exception:
            logger.exception("Failed to save calibration file to %s", path)
            QMessageBox.critical(self, "Save Calibration Failed", f"Unable to save calibration to:\n{path}")
            return
        QMessageBox.information(self, "Calibration Saved", f"Calibration saved to:\n{path}")
    def load_calibrations(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Calibration",
            str(CALIBRATIONS_DIR),
            "Calibration Files (*.json);;All Files (*)",
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception:
            logger.exception("Failed to read calibration file from %s", path)
            QMessageBox.critical(self, "Load Calibration Failed", f"Unable to open calibration:\n{path}")
            return
        updated = []
        try:
            us_matrix = np.asarray(payload["us_correction_matrix"], dtype=np.float32)
            if us_matrix.shape == (4, 4):
                self.us_correction_matrix = us_matrix
                updated.append("Ultrasound transform")
            else:
                raise ValueError("us_correction_matrix must be 4x4")
        except Exception as exc:
            logger.warning("Invalid ultrasound calibration in %s: %s", path, exc)
        spacing = payload.get("assumed_pixel_spacing_mm")
        if spacing is not None:
            try:
                spacing_arr = np.asarray(spacing, dtype=np.float64).reshape(2)
            except Exception as exc:
                logger.warning("Invalid pixel spacing in %s: %s", path, exc)
            else:
                self._update_pixel_spacing(tuple(float(x) for x in spacing_arr))
                updated.append("Pixel spacing (assumed)")
        plane_normal = payload.get("plane_normal")
        if plane_normal is not None:
            try:
                norm_arr = np.asarray(plane_normal, dtype=np.float64).reshape(3)
                self._plane_normal = norm_arr
                updated.append("Plane normal")
            except Exception as exc:
                logger.warning("Invalid plane normal in %s: %s", path, exc)
        plane_offset = payload.get("plane_offset_m")
        if plane_offset is not None:
            try:
                self._plane_offset_m = float(plane_offset)
                updated.append("Plane offset")
            except Exception as exc:
                logger.warning("Invalid plane offset in %s: %s", path, exc)
        try:
            needle_matrix = np.asarray(payload["needle_correction_matrix"], dtype=np.float32)
            if needle_matrix.shape == (4, 4):
                self.needle_correction_matrix = needle_matrix
                updated.append("Needle transform")
            else:
                raise ValueError("needle_correction_matrix must be 4x4")
        except Exception as exc:
            logger.warning("Invalid needle calibration in %s: %s", path, exc)
        global _last_tip_offset_mm
        tip_offset = payload.get("needle_tip_offset_mm")
        if tip_offset is not None:
            try:
                tip_arr = np.asarray(tip_offset, dtype=np.float64).reshape(3)
            except Exception as exc:
                logger.warning("Invalid needle tip offset in %s: %s", path, exc)
                tip_arr = None
            if tip_arr is not None:
                _last_tip_offset_mm = tip_arr
                updated.append("Needle tip offset")
                self._apply_tip_transform_to_viewer(tip_arr)
            else:
                self._apply_tip_transform_to_viewer(None)
        else:
            _last_tip_offset_mm = None
            self._apply_tip_transform_to_viewer(None)
        pivot_result = payload.get("pivot_result")
        if pivot_result is not None:
            converted: Dict[str, Any] = {}
            for key, value in pivot_result.items():
                if isinstance(value, list):
                    converted[key] = np.asarray(value, dtype=np.float64)
                else:
                    converted[key] = value
            _needle_pivot_calibrator._last_result = converted
        self._pivot_capture_count = int(payload.get("pivot_sample_count", 0))
        self._pivot_recording = False
        if hasattr(self, "needle_pivot_btn"):
            self.needle_pivot_btn.blockSignals(True)
            self.needle_pivot_btn.setChecked(False)
            self.needle_pivot_btn.blockSignals(False)
            self.needle_pivot_btn.setText("Start Needle Pivot Calibration")
        self._update_needle_pivot_status("loaded")
        if updated:
            details = "\n".join(updated)
            QMessageBox.information(self, "Calibration Loaded", f"Loaded items:\n{details}")
        else:
            QMessageBox.warning(self, "Calibration Loaded", "File loaded but no calibration entries were applied.")
    def _close_napari_viewer(self, viewer, label=""):
        if viewer is None:
            return
        try:
            if hasattr(viewer, "isVisible") and not viewer.isVisible():
                logger.info(f"Viewer {label} already closed/skipped.")
                return
            if hasattr(viewer, "close"):
                viewer.close()
            else:
                logger.debug(f"Viewer {label} has no close method; skipping.")
                return
        except RuntimeError as e:
            logger.info(f"Viewer {label} already closed: {e}")
        except Exception as e:
            logger.warning(f"Exception closing napari viewer {label}: {e}", exc_info=True)
        try:
            qt_window = getattr(viewer, "_qt_window", None)
            if qt_window is None:
                outer_window = getattr(viewer, "window", None)
                qt_window = getattr(outer_window, "_qt_window", None) if outer_window else None
            if qt_window and hasattr(qt_window, "isVisible") and qt_window.isVisible():
                qt_window.close()
        except Exception as e:
            logger.warning(f"Exception closing napari Qt window for {label}: {e}", exc_info=True)
    def closeEvent(self, event) -> None:
        self._perform_shutdown()
        super().closeEvent(event)
    def project_to_slice(self, key: str, marker: np.ndarray, clip: bool = True) -> np.ndarray:
        """
        Project a 3D marker (z, y, x) onto the 2D plane for the given slice key.
        Returns (row, col) pixel coordinates as used by napari's points layer (y, x).
        Set `clip=False` to leave bounds handling to the caller.
        """
        marker_mm = np.asarray(marker, dtype=np.float32)
        row_axis, col_axis = self._slice_axes_indices(key)
        voxel_sizes = getattr(self, "_voxel_size_mm", None)
        if voxel_sizes is None or len(voxel_sizes) != 3:
            spacing_src = getattr(self, "_current_pixel_spacing_mm", None)
            try:
                spacing_arr = np.asarray(spacing_src, dtype=np.float64).reshape(2)
                slice_thickness = float(np.mean(spacing_arr))
                voxel_sizes = np.array([slice_thickness, spacing_arr[0], spacing_arr[1]], dtype=np.float32)
            except Exception:
                voxel_sizes = np.array(
                    [SLICE_THICKNESS_MM, PIXEL_SPACING_MM[0], PIXEL_SPACING_MM[1]],
                    dtype=np.float32,
                )
        volume = getattr(self, "volume", None)
        if isinstance(volume, np.ndarray):
            volume_shape = volume.shape
        else:
            volume_shape = VOLUME_SHAPE
        row_spacing = float(voxel_sizes[row_axis]) if row_axis < len(voxel_sizes) else 1.0
        col_spacing = float(voxel_sizes[col_axis]) if col_axis < len(voxel_sizes) else 1.0
        if row_spacing <= 0.0:
            row_spacing = 1.0
        if col_spacing <= 0.0:
            col_spacing = 1.0
        row_mm = float(marker_mm[row_axis])
        col_mm = float(marker_mm[col_axis])
        row = row_mm / row_spacing
        col = col_mm / col_spacing
        if clip:
            max_row = float(max(volume_shape[row_axis] - 1, 0))
            max_col = float(max(volume_shape[col_axis] - 1, 0))
            row = float(np.clip(row, 0.0, max_row))
            col = float(np.clip(col, 0.0, max_col))
        return np.array([row, col], dtype=np.float32)
    def _perform_shutdown(self) -> None:
        if getattr(self, '_closed', False):
            return
        self._closed = True
        if self._timer is not None:
            self._timer.stop()
            self._timer = None  # Defensive against double-stop
        self.stop_event.set()
        if self.consumer_thread and self.consumer_thread.is_alive():
            self.consumer_thread.join(timeout=2.0)
            if self.consumer_thread.is_alive():
                logger.warning("Consumer thread did not terminate in time!")
            self.consumer_thread = None  # Prevent double-join
        if getattr(self, "viewer3d", None) is not None:
            self._close_napari_viewer(self.viewer3d, "3D")
        for label, viewer_entry in list(self.slice_viewers.items()):
            qt_viewer = viewer_entry.get("qt") if isinstance(viewer_entry, dict) else viewer_entry
            self._close_napari_viewer(qt_viewer, label)
        if getattr(self, "ultrasound_viewer", None) is not None:
            self._close_napari_viewer(self.ultrasound_viewer, "ultrasound")
        if self.ultrasound_container is not None:
            try:
                self.ultrasound_container.deleteLater()
            except Exception:
                pass
            self.ultrasound_container = None
        self.ultrasound_model = None
        self.ultrasound_viewer = None
        self.ultrasound_image_layer = None
        self._ultrasound_frame = None
        # Null viewer handles
        self.model3d = None
        if hasattr(self, "viewer3d"):
            self.viewer3d = None
        for k in list(self.slice_viewers.keys()):
            self.slice_viewers[k] = None
        for k in list(self.recon_slice_layers.keys()):
            self.recon_slice_layers[k] = None
        # Delete arrays
        del self.volume, self.arrow_data, self.marker_data
        self.us_archive.clear()
        self.cam_archive.clear()
        gc.collect()
    def close(self):
        self._perform_shutdown()
        super().close()
