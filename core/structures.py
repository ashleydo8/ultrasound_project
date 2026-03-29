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
import numpy as np
import psutil
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
from qtpy.QtCore import QTimer, Qt

SCALE: float = 1000.0              # meters -> mm
ARROW_LENGTH_MM: float = 100.0
NEEDLE_ARROW_LENGTH_MM: float = 120.0
# Napari vectors/layers operate in (z, y, x) order; +y runs from image top toward bottom.
LOCAL_ARROW_DIR = np.array([0.0, 1.0, 0.0])
ARCHIVE_MAX_SAMPLES = 10_000  # or whatever is reasonable for use-case

@dataclass
class Event:
    source: str
    timestamp: float
    payload: Any
    drop_count: int = 0
    seq: int = 0


@dataclass
class RenderPacket:
    frame_ts: float
    frame: Any
    pixel_spacing_mm: Optional[Tuple[float, float]]
    poses: Dict[int, Tuple[Optional[np.ndarray], Optional[np.ndarray]]]
    markers: Any
    pose_ts: Optional[float]
    pose_age_ms: Optional[float]
    synced: bool
    drop_count: int
    pose_drop_count: int
    seq: int

@dataclass
class RecordingBuffer:
    us: Deque[Event] = field(default_factory=lambda: deque(maxlen=ARCHIVE_MAX_SAMPLES))
    cam: Deque[Event] = field(default_factory=lambda: deque(maxlen=ARCHIVE_MAX_SAMPLES))

@dataclass(frozen=True)
class Pose:
    pos_m: np.ndarray
    quat: np.ndarray
    align_quat: Optional[np.ndarray] = None
    napari_basis: bool = field(init=False, default=True, repr=False)
    body_rot: R = field(init=False)
    align_rot: R = field(init=False)
    def __post_init__(self) -> None:
        object.__setattr__(self, 'body_rot', R.from_quat(self.quat))
        if self.align_quat is not None:
            object.__setattr__(self, 'align_rot', R.from_quat(self.align_quat))
        else:
            object.__setattr__(self, 'align_rot', R.identity())
        object.__setattr__(self, 'napari_basis', True)


    def arrow(self, out: Optional[np.ndarray] = None, length_mm: Optional[float] = None) -> np.ndarray:
        """
        Returns a (1,2,3) float32 vector array for napari vectors layer.
        If `out` is provided, fills in-place. Otherwise allocates.
        """
        start = self.pos_mm.astype(np.float32)
        length = ARROW_LENGTH_MM if length_mm is None else float(length_mm)
        vec = (self.body_rot * self.align_rot).apply(LOCAL_ARROW_DIR) * length
        vec = vec.astype(np.float32)
        if out is not None and out.shape == (1, 2, 3):
            out[0, 0, :] = start
            out[0, 1, :] = vec
            return out
        else:
            arr = np.empty((1, 2, 3), dtype=np.float32)
            arr[0, 0, :] = start
            arr[0, 1, :] = vec
            return arr
        
    @property
    def pos_mm(self) -> np.ndarray:
        return self.pos_m * SCALE
    
    def markers_mm(self, markers: List[np.ndarray], out: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Converts marker list to napari (z, y, x) mm coordinates.
        Fills in-place if `out` is provided and has correct shape.
        """
        n = len(markers)
        if n == 0:
            if out is not None and out.shape[0] == 0:
                out[:] = []
                return out
            return np.zeros((0, 3), dtype=np.float32)
        arr = np.empty((n, 3), dtype=np.float32) if out is None or out.shape[0] != n else out
        arr[:, :] = np.asarray(markers, dtype=np.float32) * SCALE
        return arr
    

    @classmethod
    def from_pos_rot(cls, pos: np.ndarray, quat: np.ndarray, align_quat: Optional[np.ndarray] = None
    ) -> "Pose":
        new_pose = Pose(pos_m=pos, quat=quat)
        return new_pose 
    
  

