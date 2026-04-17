import numpy as np
from typing import Tuple, Optional
from pathlib import Path
from core.calibration import NeedlePivotCalibrator

THRESHOLD_VAL = 180

SCALE: float = 1000.0              # meters -> mm
ARROW_LENGTH_MM: float = 100.0
NEEDLE_ARROW_LENGTH_MM: float = 120.0 #small so we mainly see the STL needle surface 
# Napari vectors/layers operate in (z, y, x) order; +y runs from image top toward bottom.
LOCAL_ARROW_DIR = np.array([0.0, 1.0, 0.0])
VOLUME_SHAPE: Tuple[int, int, int] = (64, 64, 64)
QUEUE_MAXSIZE: int = 5  # GUI/render queue
ACQ_QUEUE_MAXSIZE: int = 256
RECORDER_QUEUE_MAXSIZE: int = 0  # 0 => unbounded for recorder  
GUI_FPS: int = 60
ULTRASOUND_RB_ID: int = 0 #IMPORTANT! make sure that these streaming IDs match what is in motive 
NEEDLE_RB_ID: int = 1
POSE_STALE_THRESHOLD_MS: float = 50.0
DROP_LOG_INTERVAL: float = 5.0
TRACKER_SERVER_ADDRESS: str = "127.0.0.1"
TRACKER_CLIENT_ADDRESS: str = "127.0.0.1"
TRACKER_USE_MULTICAST: bool = False
TRACKER_PRINT_LEVEL: int = 0
ARCHIVE_MAX_SAMPLES = 10_000  # or whatever is reasonable for your use-case
SLICE_AXES = {'axial': 0, 'coronal': 1, 'sagittal': 2}
SLICE_POSITIONS = {'axial': (1,0), 'coronal': (1, 1), 'sagittal': (1, 2)}
# napari expects: 2D (row, col) = (y, x); 3D (z, y, x)
VOLUME_SHAPE = (64, 480, 640)  # (num_slices, height, width)
PIXELS_PER_MM: float = 10.0
PIXEL_SPACING_MM: Tuple[float, float] = (1.0 / PIXELS_PER_MM, 1.0 / PIXELS_PER_MM)
# TODO: Replace this assumed spacing with values parsed from ultrasound metadata/DICOM export.
ASSUMED_US_PIXEL_SPACING_MM: Tuple[float, float] = PIXEL_SPACING_MM
SLICE_THICKNESS_MM: float = 1.0 / PIXELS_PER_MM
PROBE_TO_IMAGE_XFORM = np.eye(4, dtype=np.float32)
RECORDINGS_DIR = Path("recordings")
# Scale factor to restore ultrasound plane visual size (1 mm per pixel instead of physical spacing).
ULTRASOUND_PLANE_VIS_SCALE = 1.0  # keep ultrasound plane in physical mm units
TABLE_PLANE_NORMAL = np.array([0.0, 1.0, 0.0], dtype=np.float64) #y is normal (up)
TABLE_PLANE_OFFSET_M = 0  # meters; set to 0 for flush-with-origin tables
# Initial alignment quaternion applied to incoming poses (x, y, z, w order).
CALIBRATIONS_DIR = Path("calibrations")
_needle_pivot_calibrator = NeedlePivotCalibrator()
_last_tip_offset_mm: Optional[np.ndarray] = None
GROUND_OUTLINE_HALF_SIZE_MM: float = 200.0  # extends 20 cm from origin in +/-X and +/-Y
GROUND_THICKNESS_MM: float = 2.5  # ground outline/axis prism thickness in mm
# Desired simple-calibration orientations (image y points down):
# - Forward faces +X (plane parallel to YZ), rows -> -Y (down), cols -> +Z (right), normal -> +X.
SIMPLE_US_ROT_MAT_FWD = np.array(
    [
        [0.0, 0.0, 1.0],   # image x (cols) -> +Z
        [0.0, -1.0, 0.0],  # image y (rows) -> -Y (arrow down)
        [1.0, 0.0, 0.0],   # image z (normal) -> +X
    ],
    dtype=np.float64,
)
# - Flipped faces -X (still YZ-parallel), rows -> -Y, cols -> -Z, normal -> -X.
SIMPLE_US_ROT_MAT_REV = np.array(
    [
        [0.0, 0.0, -1.0],  # image x (cols) -> -Z
        [0.0, -1.0, 0.0],  # image y (rows) -> -Y (arrow down)
        [-1.0, 0.0, 0.0],  # image z (normal) -> -X
    ],
    dtype=np.float64,
)

#for density mapping, with opacities, want green to be the most visible to indicate sufficient scanning
COLOR_RED    = np.array([[1.0, 0.0, 0.0, 0.2]], dtype=np.float32)
COLOR_YELLOW = np.array([[1.0, 1.0, 0.0, 0.5]], dtype=np.float32)
COLOR_GREEN  = np.array([[0.0, 1.0, 0.0, 1.0]], dtype=np.float32)