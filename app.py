import logging
import os
import gc
import queue
import sys

os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
os.environ["QT_API"] = "pyqt5"

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
import drivers.imaging as drimg
from drivers.imaging import args, UltrasoundImage
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
from core.spatial import Pose
from ui.viewer import UltrasoundCalibrationSample, on_pivot_start, on_pivot_capture, on_pivot_solve

from ui.constants import ACQ_QUEUE_MAXSIZE, ARCHIVE_MAX_SAMPLES, RECORDER_QUEUE_MAXSIZE, ULTRASOUND_PLANE_VIS_SCALE, POSE_STALE_THRESHOLD_MS, DROP_LOG_INTERVAL
from ui.constants import VOLUME_SHAPE, ULTRASOUND_RB_ID,ASSUMED_US_PIXEL_SPACING_MM, TABLE_PLANE_NORMAL, TABLE_PLANE_OFFSET_M, GUI_FPS, SLICE_AXES,SLICE_POSITIONS, GROUND_OUTLINE_HALF_SIZE_MM, GROUND_THICKNESS_MM
from ui.constants import NEEDLE_RB_ID, LOCAL_ARROW_DIR, NEEDLE_ARROW_LENGTH_MM, RECORDINGS_DIR, PIXEL_SPACING_MM, SIMPLE_US_ROT_MAT_FWD, SIMPLE_US_ROT_MAT_REV
from ui.constants import SCALE, _needle_pivot_calibrator, CALIBRATIONS_DIR, SLICE_THICKNESS_MM, QUEUE_MAXSIZE, TRACKER_CLIENT_ADDRESS, TRACKER_SERVER_ADDRESS, TRACKER_USE_MULTICAST, TRACKER_PRINT_LEVEL

# IMPORTANT: On Windows, importing Qt *before* hardware DLLs can cause crashes or device enumeration failures.
# Always import all hardware dependencies (e.g., ultrasound, NatNet) FIRST!

from qtpy.QtCore import QTimer, Qt
from qtpy.QtWidgets import (
    QApplication,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QCheckBox,
    QFileDialog,
    QMessageBox,
)

USE_FAKE_ULTRASOUND = True   
USE_FAKE_POSE = True        

recording_buffer = RecordingBuffer()

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

def consumer_main(
    out_q: queue.Queue,
    stop_event: threading.Event,
    tracker_factory: Callable[[], NatNetClientWrapper],
) -> None:
    logger.info("consumer_main: starting (stop_event=%s)", stop_event.is_set())
    acq_q: queue.Queue = queue.Queue(maxsize=ACQ_QUEUE_MAXSIZE)
    recorder_q: queue.Queue = (
        queue.Queue(maxsize=RECORDER_QUEUE_MAXSIZE)
        if RECORDER_QUEUE_MAXSIZE > 0
        else queue.Queue()
    )
    drop_counters: defaultdict[str, int] = defaultdict(int)
    seq_counters: defaultdict[str, int] = defaultdict(int)
    drop_lock = threading.Lock()
    render_drop_count = 0
    render_lock = threading.Lock()

    def _enqueue_event(source: str, timestamp: float, payload: Any) -> None:
        with drop_lock:
            seq_counters[source] += 1
            event = Event(source=source, timestamp=timestamp, payload=payload, seq=seq_counters[source])
            event.drop_count = drop_counters[source]
            try:
                acq_q.put_nowait(event)
                return
            except queue.Full:
                try:
                    acq_q.get_nowait()
                except queue.Empty:
                    pass
                drop_counters[source] += 1
                event.drop_count = drop_counters[source]
                try:
                    acq_q.put_nowait(event)
                except queue.Full:
                    logger.warning("Failed to enqueue %s event; acquisition queue saturated", source)

    def _push_render(packet: RenderPacket) -> None:
        nonlocal render_drop_count
        with render_lock:
            try:
                out_q.put_nowait(packet)
                return
            except queue.Full:
                try:
                    out_q.get_nowait()
                except queue.Empty:
                    pass
                render_drop_count += 1
                try:
                    out_q.put_nowait(packet)
                except queue.Full:
                    logger.warning("Render queue saturated; dropping packet seq=%d", packet.seq)

    try:
        try:
            p = psutil.Process(os.getpid())
            p.cpu_affinity([1, 2, 3])
            p.nice(psutil.REALTIME_PRIORITY_CLASS)
        except Exception:
            logger.warning("Failed to set real-time priority/affinity", exc_info=True)
        if tracker_factory is None:
            raise ValueError("tracker_factory must be provided")
        with tracker_factory() as tracker:
            def us_thread():
                try:
                    while not stop_event.is_set():
                        ts, frame, microns_per_px = drimg.get_new_us_frame()
                        spacing_mm = None
                        if microns_per_px is not None:
                            try:
                                mm_per_px = float(microns_per_px) / 1000.0
                                if np.isfinite(mm_per_px) and mm_per_px > 0.0:
                                    spacing_mm = (mm_per_px, mm_per_px)
                            except Exception:
                                spacing_mm = None
                        _enqueue_event("us", ts, (frame, spacing_mm))
                        time.sleep(0.001)
                except Exception:
                    logger.exception("Fatal error in US thread")
                    stop_event.set()
                finally:
                    logger.info("us_thread exiting (stop_event=%s)", stop_event.is_set())
            def cam_thread():
                try:
                    while not stop_event.is_set():
                        poses = tracker.get_rigid_body_poses()
                        markers = tracker.get_unlabeled_markers()
                        us_pose = poses.get(ULTRASOUND_RB_ID)
                        if (
                            us_pose is not None
                            and us_pose[0] is not None
                            and us_pose[1] is not None
                        ):
                            ts = time.perf_counter()
                            _enqueue_event("cam", ts, (poses, markers))
                        time.sleep(1.0 / 120.0)
                except Exception:
                    logger.exception("Fatal error in Cam thread")
                    stop_event.set()
                finally:
                    logger.info("cam_thread exiting (stop_event=%s)", stop_event.is_set())

            placeholder_frame = np.zeros((480, 640), dtype=np.float32)

            def coordinator_thread():
                # Initialize state for synchronization
                latest_cam_ts: Optional[float] = None
                latest_poses: Dict[int, Tuple[Optional[np.ndarray], Optional[np.ndarray]]] = {}
                latest_markers: List[np.ndarray] = []
                last_cam_drop = 0
                last_log = time.perf_counter()

                while not stop_event.is_set():
                    try:
                        event = acq_q.get(timeout=0.05)
                    except queue.Empty:
                        continue

                    try:
                        recorder_q.put_nowait(event)
                    except queue.Full:
                        logger.warning("Recorder queue full; dropping %s event seq=%d", event.source, event.seq)
                    
                    # 1. Update Tracking State
                    if event.source == "cam":
                        latest_cam_ts = event.timestamp
                        latest_poses, latest_markers = event.payload
                        last_cam_drop = event.drop_count
                        # We do NOT push a render packet here; we wait for an ultrasound frame 
                        # so the image and pose are coupled in time.
                        continue
                    
                    # 2. Process Ultrasound and Render
                    if event.source == "us":
                        frame_payload = event.payload
                        pixel_spacing_mm = None
                        frame = frame_payload
                        
                        # Handle tuple payload (frame, spacing)
                        if isinstance(frame_payload, tuple) and len(frame_payload) == 2:
                            frame, pixel_spacing_mm = frame_payload
                        
                        # Calculate sync metrics
                        pose_age_ms: Optional[float] = None
                        synced = False
                        if latest_cam_ts is not None:
                            pose_age_ms = (event.timestamp - latest_cam_ts) * 1000.0
                            # A "synced" frame is one where the pose is very close in time to the image
                            synced = 0.0 <= pose_age_ms <= POSE_STALE_THRESHOLD_MS
                        
                        # Create the packet using the LATEST known tracking data
                        packet = RenderPacket(
                            frame_ts=event.timestamp,
                            frame=frame,
                            pixel_spacing_mm=pixel_spacing_mm,
                            poses=latest_poses,
                            markers=latest_markers,
                            pose_ts=latest_cam_ts,
                            pose_age_ms=pose_age_ms,
                            synced=synced,
                            drop_count=event.drop_count,
                            pose_drop_count=last_cam_drop,
                            seq=event.seq,
                        )
                        
                        _push_render(packet)
                        
                        # Logging
                        now = time.perf_counter()
                        if now - last_log > DROP_LOG_INTERVAL:
                            logger.debug(
                                "Drops -> US: %d, Cam: %d, Render: %d",
                                drop_counters["us"],
                                drop_counters["cam"],
                                render_drop_count,
                            )
                            last_log = now

                logger.info("coordinator_thread exiting (stop_event=%s)", stop_event.is_set())


            def recorder_thread():
                while not stop_event.is_set():
                    try:
                        evt: Event = recorder_q.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    if evt.source == "us":
                        recording_buffer.us.append(evt)
                    elif evt.source == "cam":
                        recording_buffer.cam.append(evt)
                logger.info("recorder_thread exiting (stop_event=%s)", stop_event.is_set())
            threads = []
            for fn in (us_thread, cam_thread, coordinator_thread, recorder_thread):
                t = threading.Thread(target=fn, daemon=False)
                t.start()
                threads.append(t)
            logger.info("consumer_main: threads started %s", [t.name for t in threads])
            # Stay alive until we are told to stop; exit early if any worker dies unexpectedly.
            while not stop_event.wait(timeout=0.5):
                dead_threads = [t.name for t in threads if not t.is_alive()]
                if dead_threads:
                    logger.error("Worker thread(s) terminated unexpectedly: %s", dead_threads)
                    stop_event.set()
                    break
            for t in threads:
                t.join(timeout=1.5)
                logger.info("consumer_main: join %s alive=%s", t.name, t.is_alive())
    except Exception:
        logger.exception("Fatal error in consumer_main")
    finally:
        logger.info("consumer_main: finally stop_event.set() (was %s)", stop_event.is_set())
        stop_event.set()

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
        self.us_correction_matrix[:3, :3] = SIMPLE_US_ROT_MAT_REV.astype(np.float32)
        rot = R.from_euler('z', 22, degrees=True).as_matrix()
        self.us_correction_matrix[:3, :3] = self.us_correction_matrix[:3, :3] @ rot
        self.world_origin_offset = np.eye(4, dtype=np.float32)
        self.needle_correction_matrix = np.eye(4, dtype=np.float32)
        self.needle_correction_matrix[:3, :3] = R.from_euler('x', 180, degrees=True).as_matrix() #ensure dir is correct 
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
        
    def _init_ui(self) -> None:
        self.setWindowTitle(f"Kaneko Lab - US Reconstruction ({GUI_FPS} Hz)")
        self.resize(1600, 850)
        main_layout = QHBoxLayout(self)
        self.setLayout(main_layout)
        controls_container = QWidget(self)
        controls_layout = QVBoxLayout(controls_container)
        controls_layout.setContentsMargins(4, 4, 4, 4)
        controls_layout.setSpacing(6)
        controls_container.setMaximumWidth(320)
        
        self.title = QLabel("Kaneko Lab - 3D US Reconstruction")
        self.title.setStyleSheet("font-weight: bold; font-size: 16px;")
        controls_layout.addWidget(self.title)


        #-----------------------

        self.us_arrow_chk = QCheckBox("Show Probe Arrow")
        self.us_arrow_chk.setChecked(True)
        self.us_arrow_chk.stateChanged.connect(self.toggle_probe_arrow)
        controls_layout.addWidget(self.us_arrow_chk)
        self.needle_arrow_chk = QCheckBox("Show Needle Line")
        self.needle_arrow_chk.setChecked(True)
        self.needle_arrow_chk.stateChanged.connect(self.toggle_needle_arrow)
        controls_layout.addWidget(self.needle_arrow_chk)
        self.needle_slice_chk = QCheckBox("Show Needle on Slices")
        self.needle_slice_chk.setChecked(True)
        self.needle_slice_chk.stateChanged.connect(self.toggle_needle_slice_overlay)
        controls_layout.addWidget(self.needle_slice_chk)
        self.marker_chk = QCheckBox("Show Markers")
        self.marker_chk.setChecked(True)
        self.marker_chk.stateChanged.connect(self.toggle_markers)
        controls_layout.addWidget(self.marker_chk)
        self.ultrasound_chk = QCheckBox("Show Ultrasound Overlays")
        self.ultrasound_chk.setChecked(True)
        self.ultrasound_chk.stateChanged.connect(self.toggle_ultrasound)
        controls_layout.addWidget(self.ultrasound_chk)
        self.ultrasound_pane_chk = QCheckBox("Show Ultrasound Pane")
        self.ultrasound_pane_chk.setChecked(False)
        self.ultrasound_pane_chk.stateChanged.connect(self.toggle_ultrasound_pane)
        controls_layout.addWidget(self.ultrasound_pane_chk)

        #--------------------------

        self.calib_stats_group = QGroupBox("Calibration Status")
        stats_layout = QVBoxLayout(self.calib_stats_group)
        self.us_label = QLabel("Ultrasound")
        self.us_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.rms_label = QLabel("RMS Error: N/A")
        self.rms_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        stats_layout.addWidget(self.rms_label)
        self.sample_count_label = QLabel("Samples: 0")
        stats_layout.addWidget(self.sample_count_label)
        self.calib_health_label = QLabel("Status: Not Calibrated")
        stats_layout.addWidget(self.calib_health_label)

        self.needle_label = QLabel("Needle")
        self.needle_pivot_status = QLabel("Pivot samples: 0 (idle)")
        self.needle_pivot_status.setWordWrap(True)
        stats_layout.addWidget(self.needle_pivot_status)

        self.needle_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.place_needle_tip_btn = QPushButton("Add Needle Tip Marker")
        self.place_needle_tip_btn.setStyleSheet("font-weight: normal;")
        self.place_needle_tip_btn.clicked.connect(self.add_needle_tip_marker)
        self.place_needle_tip_btn.setEnabled(False)
        stats_layout.addWidget(self.place_needle_tip_btn)

        controls_layout.addWidget(self.calib_stats_group)


        #-------------

        self.us_calib_group = QGroupBox("Calibrate Ultrasound")
        us_layout = QVBoxLayout(self.us_calib_group)
        us_layout.setSpacing(10)
        self.simple_us_origin_btn = QPushButton("Set Ultrasound Origin")
        self.simple_us_origin_btn.clicked.connect(self.simple_set_ultrasound_origin)
        self.simple_us_origin_btn.setEnabled(True)
        self.simple_us_origin_btn.setStyleSheet("font-weight: normal;")
        us_layout.addWidget(self.simple_us_origin_btn)

        self.calibrate_us_btn = QPushButton("Calibrate Ultrasound Transform")
        self.calibrate_us_btn.setStyleSheet("font-weight: normal;")
        self.calibrate_us_btn.clicked.connect(self.calibrate_ultrasound_transform)
        self.calibrate_us_btn.setEnabled(False)
        us_layout.addWidget(self.calibrate_us_btn)

        self.reset_us_transform_btn = QPushButton("Reset Ultrasound Transform")
        self.reset_us_transform_btn.setStyleSheet("font-weight: normal;")
        self.reset_us_transform_btn.clicked.connect(self.reset_ultrasound_transform)
        us_layout.addWidget(self.reset_us_transform_btn)

        controls_layout.addWidget(self.us_calib_group)

        #---------------

        self.needle_calib_group = QGroupBox("Calibrate Needle")
        needle_layout = QVBoxLayout(self.needle_calib_group)
        needle_layout.setSpacing(10)

        self.simple_needle_origin_btn = QPushButton("Set Needle Origin")
        self.simple_needle_origin_btn.setStyleSheet("font-weight: normal;")
        self.simple_needle_origin_btn.clicked.connect(self.simple_set_needle_origin)
        self.simple_needle_origin_btn.setEnabled(True)
        needle_layout.addWidget(self.simple_needle_origin_btn)

        self.needle_pivot_btn = QPushButton("Start Needle Pivot Calibration")
        self.needle_pivot_btn.setStyleSheet("font-weight: normal;")
        self.needle_pivot_btn.setCheckable(True)
        self.needle_pivot_btn.toggled.connect(self.toggle_needle_pivot_calibration)
        self.needle_pivot_btn.setEnabled(False)
        needle_layout.addWidget(self.needle_pivot_btn)

        self.clear_needle_tip_btn = QPushButton("Clear Needle Tip Calibration")
        self.clear_needle_tip_btn.setStyleSheet("font-weight: normal;")
        self.clear_needle_tip_btn.clicked.connect(self.clear_needle_tip_calibration)
        needle_layout.addWidget(self.clear_needle_tip_btn)

        controls_layout.addWidget(self.needle_calib_group)

        # --------------------------

        self.recon = QGroupBox("Reconstruct Volume")

        recon_layout = QVBoxLayout(self.recon)
        recon_layout.setSpacing(10)

        self.recording = False
        self.recorded_frames: List[Tuple[float, np.ndarray, Pose, Optional[Tuple[float, float]]]] = []
        self.last_volume_zyx: Optional[np.ndarray] = None
        self.last_weights_zyx: Optional[np.ndarray] = None
        self.last_recon_meta: Optional[dict] = None
        self.last_recon_affine: Optional[np.ndarray] = None
        self.record_btn = QPushButton("Start Recording")
        self.record_btn.setStyleSheet("font-weight: normal;")
        self.record_btn.setCheckable(True)
        self.record_btn.clicked.connect(self.toggle_recording)
        recon_layout.addWidget(self.record_btn)
        self.reconstruct_btn = QPushButton("Reconstruct Volume")
        self.reconstruct_btn.setStyleSheet("font-weight: normal;")
        self.reconstruct_btn.clicked.connect(self.reconstruct_volume_from_recording)
        recon_layout.addWidget(self.reconstruct_btn)

        self.recon_volume_chk = QCheckBox("Show Reconstructed Volume")
        self.recon_volume_chk.setStyleSheet("font-weight: normal;")
        self.recon_volume_chk.setChecked(False)
        self.recon_volume_chk.setEnabled(False)
        self.recon_volume_chk.stateChanged.connect(self.toggle_recon_volume)
        recon_layout.addWidget(self.recon_volume_chk)

        controls_layout.addWidget(self.recon)

        #---------------

        
        self.features = QGroupBox("Other Features")
        features_layout = QVBoxLayout(self.features)
        features_layout.setSpacing(10)

        self.rb_status_label = QLabel(f"Streaming rigid bodies:____")
        self.rb_status_label.setWordWrap(True)
        features_layout.addWidget(self.rb_status_label)

        self.reset_btn = QPushButton("Reset View")
        self.reset_btn.setStyleSheet("font-weight: normal;")
        self.reset_btn.clicked.connect(self.reset_view)
        features_layout.addWidget(self.reset_btn)

        self.axes_chk = QCheckBox("Show Axes")
        self.axes_chk.setStyleSheet("font-weight: normal;")
        self.axes_chk.setChecked(False)
        self.axes_chk.stateChanged.connect(self.toggle_axes)
        features_layout.addWidget(self.axes_chk)

        self.load_calibration_btn = QPushButton("Load Calibration...")
        self.load_calibration_btn.setStyleSheet("font-weight: normal;")
        self.load_calibration_btn.clicked.connect(self.load_calibrations)
        features_layout.addWidget(self.load_calibration_btn)

        self.save_calibration_btn = QPushButton("Save Calibration...")
        self.save_calibration_btn.setStyleSheet("font-weight: normal;")
        self.save_calibration_btn.clicked.connect(self.save_calibrations)
        features_layout.addWidget(self.save_calibration_btn)

        controls_layout.addWidget(self.features)

        # --------------------------
    
        self.quit = QGroupBox()
        quit_layout = QVBoxLayout(self.quit)
        self.quit_btn = QPushButton("Quit")
        self.quit_btn.clicked.connect(self.quit_application)
        quit_layout.addWidget(self.quit_btn)
        controls_layout.addWidget(self.quit)

        #-------------

        controls_layout.addStretch(1)
        main_layout.addWidget(controls_container, 0)
        main_container = QWidget(self)
        main_container_layout = QVBoxLayout(main_container)
        main_container_layout.setContentsMargins(0, 0, 0, 0)
        main_container_layout.setSpacing(0)
        self.grid = QGridLayout()
        main_container_layout.addLayout(self.grid)
        main_layout.addWidget(main_container, 1)

    def quit_application(self) -> None:
        self._perform_shutdown()
        app = QApplication.instance()
        if app is not None:
            app.quit()

    #create an stl probe to overlay the ultrasound arrow (napari mesh layer)

    import meshio 

    def load_probe_mesh(self, path):
        mesh = self.meshio.read(path)
        #for a napari surface: (vertices, faces, values)
        return (mesh.points, mesh.cells_dict['triangle'], np.ones(len(mesh.points)))
    
    def _init_viewers_and_layers(self) -> None:
        # 3D viewer
        model3d = ViewerModel(ndisplay=3)
        if hasattr(model3d.axes, "in_canvas"):
            model3d.axes.in_canvas = True
        self.volume_layer = model3d.add_image(self.volume, name='Ultrasound Volume')  
        model3d.dims.axis_labels = ('Z','Y','X')
        self.model3d = model3d

        self._apply_camera_defaults()
        qtviewer3d = QtViewer(model3d, show_welcome_screen=False)
        qtviewer3d.setParent(self)
        qtviewer3d.show()
        self.viewer3d = qtviewer3d
        self.grid.addWidget(qtviewer3d, 0, 0, 1, 3) #starts at (0,0), takes 1 row, spans 3 columns 
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
      
        self._ultrasound_frame = np.zeros(UltrasoundImage.US_2D_SHAPE, dtype=np.float32)
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
        plane_keys = ['axial', 'coronal', 'sagittal']
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
            edge_color='yellow',
            visible=self.us_arrow_chk.isChecked() if hasattr(self, "us_arrow_chk") else True,
        )

        self.probe_mesh_data = self.load_probe_mesh(r"C:\Users\mingd\Desktop\ultrasound_project\ui\probe.stl")

        self.probe_layer = self.model3d.add_surface(
            self.probe_mesh_data,
            name='Probe STL',
            blending='translucent',
            colormap="yellow",
            shading='smooth',
            opacity=1
        )

        # self.needle_layer = self.model3d.add_surface(
        #     self.needle_mesh_data,
        #     name='Needle STL',
        #     colormap="cyan",
        #     blending='translucent',
        #     shading='smooth',
        #     opacity=1
        # )

        self.needle_vec_layer = self.model3d.add_vectors(
            self.needle_arrow_data,
            name='Needle',
            edge_width=vec_width,
            vector_style='line',
            edge_color='cyan',
            visible=self.needle_arrow_chk.isChecked() and self._needle_has_pose if hasattr(self, "needle_arrow_chk") else False,
        )
        self.marker_layer = self.model3d.add_points(self.marker_data[:0], ndim=3, size=5.0, name='Markers')
        self.needle_tip_layer = self.model3d.add_points(
            np.zeros((0, 3), dtype=np.float32),
            ndim=3,
            size=5.0,
            name='Needle Tip Markers',
            face_color='magenta',
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
        
            calibrated_pose = pose_matrix @ self.us_correction_matrix
            corrected_matrix = self.world_origin_offset @ calibrated_pose
            probe_matrix = corrected_matrix
            probe_translation = probe_matrix.copy()

            orientation_correction = R.from_euler('y', 90, degrees=True).as_matrix()
            probe_orientation = np.eye(4) 
            probe_orientation[:3,:3] = orientation_correction
            probe_translation[:3,3] = probe_translation[:3,3] * 1000
            # print(f"DEBUG: probe translation is: {probe_translation[:3,3]}")
            self.probe_layer.affine = probe_translation @ probe_orientation
            if self.probe_layer is not None:
                self.probe_layer.visible = self.us_arrow_chk.isChecked()

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
                
                affine_matrix = np.eye(4)
                needle_matrix = np.eye(4)
                orientation_correction = R.from_euler('x', 90, degrees=True).as_matrix()
                needle_matrix[:3,:3] = orientation_correction
                affine_matrix[:3, :3] = corrected_top_matrix[:3, :3] @ orientation_correction # Rotation 
                affine_matrix[:3, 3] = corrected_top_matrix[:3, 3]  * 1000 # Translation (Position)

                if np.linalg.norm(vec_mm) < 1e-6:
                    vec_mm = (top_pose.body_rot * top_pose.align_rot).apply(LOCAL_ARROW_DIR)
                    vec_mm = (vec_mm * NEEDLE_ARROW_LENGTH_MM).astype(np.float32)
                self.needle_arrow_data[0, 0, :] = start_mm
                self.needle_arrow_data[0, 1, :] = vec_mm.astype(np.float32)
                if self.needle_vec_layer is not None:
                    self.needle_vec_layer.data = self.needle_arrow_data
                    # self.needle_layer.affine = affine_matrix
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
        
        if visible:
            self.grid.removeWidget(self.viewer3d)
            self.grid.removeWidget(self.ultrasound_container)

            self.grid.addWidget(self.viewer3d, 0, 0, 1, 2)
            self.grid.addWidget(self.ultrasound_container, 0, 2, 1, 1)

            self.grid.setColumnStretch(0, 1)
            self.grid.setColumnStretch(1, 1)
            self.grid.setColumnStretch(2, 1)
        else: 
            self.grid.addWidget(self.viewer3d, 0, 0, 1, 3)
            self.grid.setColumnStretch(0, 1)
            self.grid.setColumnStretch(1, 1)
            self.grid.setColumnStretch(2, 1)

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

                rms = result.get('rms_mm', "n/a")
                count = result.get('num_samples', "n/a")
                        
                self.rms_label.setText(f"RMS Error: {rms:.4f} mm")
                self.sample_count_label.setText(f"Samples: {count}")
                
                if rms < 1.0:
                    self.rms_label.setStyleSheet("color: #2ecc71; font-weight: bold; font-size: 14px;") # Green
                    self.calib_health_label.setText("Status: Good Alignment")
                elif rms < 2.5:
                    self.rms_label.setStyleSheet("color: #f1c40f; font-weight: bold; font-size: 14px;") # Yellow
                    self.calib_health_label.setText("Status: Acceptable")
                else:
                    self.rms_label.setStyleSheet("color: #e74c3c; font-weight: bold; font-size: 14px;") # Red
                    self.calib_health_label.setText("Status: Poor Alignment")

                self._last_us_calibration_result = result

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

    def simple_set_ultrasound_origin(self) -> None:
        if self._last_raw_pos_m is not None and self._last_raw_rot is not None:
            current_calibrated_pose = self._pose_to_matrix(self._last_raw_pos_m, self._last_raw_rot) @ self.us_correction_matrix
            self.world_origin_offset = np.eye(4, dtype=np.float32)
            self.world_origin_offset[:3, 3] = -current_calibrated_pose[:3, 3]

    def simple_set_needle_origin(self):
        if self._last_raw_needle_pos_m is not None and self._last_raw_needle_rot is not None:

            current_inv_rot = self._last_raw_needle_rot.inv().as_matrix()
            current_inv_trans = -current_inv_rot @ self._last_raw_needle_pos_m
            
            self.needle_correction_matrix = np.eye(4)
            self.needle_correction_matrix[:3, :3] = current_inv_rot
            self.needle_correction_matrix[:3, 3] = current_inv_trans
    
        logger.info("Needle origin calibrated to current pose.")


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

                # print("DEBUG: First world pt:", world_pts[0, :3])
                # print("DEBUG: Distance (mm):", dist[0] * 1000)

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

        print("DEBUG: Pose translation:", s.pose_matrix[:3, 3])
        # print("DEBUG: Plane normal:", plane_normal)

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

def main() -> None:

    # Always create QApplication FIRST!
    app = QApplication(sys.argv)
    print("QApplication created:", app)
    if qdarkstyle is not None:
        # Apply a consistent dark theme across all Qt widgets
        app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5'))
    else:
        logger.warning(
            "qdarkstyle is not installed; falling back to the default Qt theme."
        )


    # Safe to create queues and widgets now
    frame_q = queue.Queue(maxsize=QUEUE_MAXSIZE)
    stop_event = threading.Event()
    rc = 0
    try:
        blank_volume = np.zeros(VOLUME_SHAPE, dtype=np.float32)
        # blank_volume = (np.random.rand(*VOLUME_SHAPE) * 255).astype(np.float32)

        # Start the ultrasound stream (fake or real, based on config)
        drimg.start_ultrasound_stream(ip=args.ip, port=args.port,fake_data=USE_FAKE_ULTRASOUND)

        tracker = NatNetClientWrapper(
            server_address=TRACKER_SERVER_ADDRESS,
            client_address=TRACKER_CLIENT_ADDRESS,
            use_multicast=TRACKER_USE_MULTICAST,
            print_level=TRACKER_PRINT_LEVEL,
            use_fake=USE_FAKE_POSE,
        )

        tracker.start()        

        print("creating viewer")
        viewer = RealTimeUSViewer(frame_q, stop_event, consumer_thread=None, volume=blank_volume)
        print("created viewer")
        viewer.show()
        print("showed viewer")

        # Now start the consumer thread (AFTER GUI setup)
        consumer = threading.Thread(
            target=consumer_main,
            args=(frame_q, stop_event, lambda: tracker),
            daemon=False,
        )
        consumer.start()
        viewer.consumer_thread = consumer  # If viewer manages/join the thread
        rc = app.exec_()
        print("executing app")
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Shutting down gracefully...")
        stop_event.set()
        rc = 130
    except Exception:
        logger.exception("Unhandled exception in GUI")
        rc = 1
    finally:
        # Shutdown signal for everything
        stop_event.set()
        # Join background threads
        try:
            if 'consumer' in locals() and consumer.is_alive():
                consumer.join(timeout=2.0)
        except Exception:
            logger.warning("Error joining consumer thread", exc_info=True)
        # Stop hardware streams
        try:
            drimg.stop_ultrasound_stream()
            tracker.stop()
        except Exception:
            logger.warning("Error shutting down ultrasound stream", exc_info=True)
    sys.exit(rc)

if __name__ == '__main__':
    main()