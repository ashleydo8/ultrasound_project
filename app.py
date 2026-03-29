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
import drivers.imaging as drimg
from drivers.imaging import args
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
from ui.viewer import UltrasoundCalibrationSample, RealTimeUSViewer

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
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QCheckBox,
    QFileDialog,
    QMessageBox,
)

USE_FAKE_ULTRASOUND = False  # or False, depending on your test
USE_FAKE_POSE = True        # or False, depending on your test

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

        tracker_factory = partial(
            NatNetClientWrapper,
            server_address=TRACKER_SERVER_ADDRESS,
            client_address=TRACKER_CLIENT_ADDRESS,
            use_multicast=TRACKER_USE_MULTICAST,
            print_level=TRACKER_PRINT_LEVEL,
            use_fake=USE_FAKE_POSE,
        )
        print("creating viewer")
        viewer = viewer.RealTimeUSViewer(frame_q, stop_event, consumer_thread=None, volume=blank_volume)
        print("created viewer")
        viewer.show()
        print("showed viewer")
        # Now start the consumer thread (AFTER GUI setup)
        consumer = threading.Thread(
            target=consumer_main,
            args=(frame_q, stop_event, tracker_factory),
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
        except Exception:
            logger.warning("Error shutting down ultrasound stream", exc_info=True)
    sys.exit(rc)

if __name__ == '__main__':
    main()
