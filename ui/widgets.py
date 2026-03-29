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
from ui.viewer import Pose

GUI_FPS: int = 60
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple
import numpy as np 



class ControlPanel(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle(f"Real-time US Viewer ({GUI_FPS} Hz)")
        self.resize(1200, 800)
        main_layout = QHBoxLayout(self)
        self.setLayout(main_layout)
        controls_container = QWidget(self)
        controls_layout = QVBoxLayout(controls_container)
        controls_layout.setContentsMargins(4, 4, 4, 4)
        controls_layout.setSpacing(6)
        controls_container.setMaximumWidth(320)
        self.reset_btn = QPushButton("Reset View")
        self.reset_btn.clicked.connect(self.reset_view)
        controls_layout.addWidget(self.reset_btn)
        self.calibrate_us_btn = QPushButton("Calibrate Ultrasound Transform")
        self.calibrate_us_btn.clicked.connect(self.calibrate_ultrasound_transform)
        self.calibrate_us_btn.setEnabled(False)
        controls_layout.addWidget(self.calibrate_us_btn)
        self.simple_us_origin_btn = QPushButton("Simple: Origin @0 (face +X)")
        self.simple_us_origin_btn.clicked.connect(self.simple_calibrate_ultrasound_origin)
        self.simple_us_origin_btn.setEnabled(False)
        controls_layout.addWidget(self.simple_us_origin_btn)
        self.simple_us_orient_btn = QPushButton("Simple: Origin @0 (face -X)")
        self.simple_us_orient_btn.clicked.connect(self.simple_calibrate_ultrasound_flipped)
        self.simple_us_orient_btn.setEnabled(False)
        controls_layout.addWidget(self.simple_us_orient_btn)
        self.reset_us_transform_btn = QPushButton("Reset Ultrasound Transform")
        self.reset_us_transform_btn.clicked.connect(self.reset_ultrasound_transform)
        controls_layout.addWidget(self.reset_us_transform_btn)
        self.needle_pivot_btn = QPushButton("Start Needle Pivot Calibration")
        self.needle_pivot_btn.setCheckable(True)
        self.needle_pivot_btn.toggled.connect(self.toggle_needle_pivot_calibration)
        self.needle_pivot_btn.setEnabled(False)
        controls_layout.addWidget(self.needle_pivot_btn)
        self.clear_needle_tip_btn = QPushButton("Clear Needle Tip Calibration")
        self.clear_needle_tip_btn.clicked.connect(self.clear_needle_tip_calibration)
        controls_layout.addWidget(self.clear_needle_tip_btn)
        self.place_needle_tip_btn = QPushButton("Add Needle Tip Marker")
        self.place_needle_tip_btn.clicked.connect(self.add_needle_tip_marker)
        self.place_needle_tip_btn.setEnabled(False)
        controls_layout.addWidget(self.place_needle_tip_btn)
        self.load_calibration_btn = QPushButton("Load Calibration...")
        self.load_calibration_btn.clicked.connect(self.load_calibrations)
        controls_layout.addWidget(self.load_calibration_btn)
        self.save_calibration_btn = QPushButton("Save Calibration...")
        self.save_calibration_btn.clicked.connect(self.save_calibrations)
        controls_layout.addWidget(self.save_calibration_btn)
        self.needle_pivot_status = QLabel("Pivot samples: 0 (idle)")
        self.needle_pivot_status.setWordWrap(True)
        controls_layout.addWidget(self.needle_pivot_status)
        self.quit_btn = QPushButton("Quit")
        self.quit_btn.clicked.connect(self.quit_application)
        controls_layout.addWidget(self.quit_btn)
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
        self.rb_status_label = QLabel("Streaming rigid bodies: (none)")
        self.rb_status_label.setWordWrap(True)
        controls_layout.addWidget(self.rb_status_label)
        self.marker_chk = QCheckBox("Show Markers")
        self.marker_chk.setChecked(True)
        self.marker_chk.stateChanged.connect(self.toggle_markers)
        controls_layout.addWidget(self.marker_chk)
        self.ultrasound_chk = QCheckBox("Show Ultrasound Overlays")
        self.ultrasound_chk.setChecked(True)
        self.ultrasound_chk.stateChanged.connect(self.toggle_ultrasound)
        controls_layout.addWidget(self.ultrasound_chk)
        self.ultrasound_pane_chk = QCheckBox("Show Ultrasound Pane")
        self.ultrasound_pane_chk.setChecked(True)
        self.ultrasound_pane_chk.stateChanged.connect(self.toggle_ultrasound_pane)
        controls_layout.addWidget(self.ultrasound_pane_chk)
        self.recon_volume_chk = QCheckBox("Show Reconstructed Volume")
        self.recon_volume_chk.setChecked(False)
        self.recon_volume_chk.setEnabled(False)
        self.recon_volume_chk.stateChanged.connect(self.toggle_recon_volume)
        controls_layout.addWidget(self.recon_volume_chk)
        self.axes_chk = QCheckBox("Show Axes")
        self.axes_chk.setChecked(False)
        self.axes_chk.stateChanged.connect(self.toggle_axes)
        controls_layout.addWidget(self.axes_chk)
        self.recording = False
        self.recorded_frames: List[Tuple[float, np.ndarray, Pose, Optional[Tuple[float, float]]]] = []
        self.last_volume_zyx: Optional[np.ndarray] = None
        self.last_weights_zyx: Optional[np.ndarray] = None
        self.last_recon_meta: Optional[dict] = None
        self.last_recon_affine: Optional[np.ndarray] = None
        self.record_btn = QPushButton("Start Recording")
        self.record_btn.setCheckable(True)
        self.record_btn.clicked.connect(self.toggle_recording)
        controls_layout.addWidget(self.record_btn)
        self.reconstruct_btn = QPushButton("Reconstruct Volume")
        self.reconstruct_btn.clicked.connect(self.reconstruct_volume_from_recording)
        controls_layout.addWidget(self.reconstruct_btn)
        controls_layout.addStretch(1)
        main_layout.addWidget(controls_container, 0)
        main_container = QWidget(self)
        main_container_layout = QVBoxLayout(main_container)
        main_container_layout.setContentsMargins(0, 0, 0, 0)
        main_container_layout.setSpacing(0)
        self.grid = QGridLayout()
        main_container_layout.addLayout(self.grid)
        main_layout.addWidget(main_container, 1)

        # Add a stretch to keep buttons at the top

        self.layout.addStretch(1)















