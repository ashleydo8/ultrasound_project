# '''
# opitrack natnet --> napari 

# 1. receive tracking data 
# 2. convert coordinate systems 

# Notes: 
# Natnet: (x,y,z)
# Napari: (-z,y,x)
# '''

import time
import numpy as np
import threading
import logging
from typing import Optional, Dict, List, Tuple, Any

from scipy.spatial.transform import Rotation as R

try:
    from dependencies.NatNetClient import NatNetClient
except ImportError:
    NatNetClient = None  # So this file can still be imported when faking

# NatNet reports positions as (x, y, z); Visualization/Napari use (z, y, x).
# Swap x<->z at the boundary so everything downstream stays in Napari order.
# Additionally, flip Napari Z (first coord) so +Z in NatNet becomes -Z in Napari.
_NATNET_TO_NAPARI_AXIS = np.array([2, 1, 0], dtype=np.int32)
_NATNET_TO_NAPARI_MATRIX = np.eye(3, dtype=np.float64)[_NATNET_TO_NAPARI_AXIS]
_NAPARI_Z_FLIP_MATRIX = np.diag([-1.0, 1.0, 1.0]).astype(np.float64)
_NATNET_TO_NAPARI_WORLD = _NAPARI_Z_FLIP_MATRIX @ _NATNET_TO_NAPARI_MATRIX


def _permute_position_to_napari(pos: Optional[Any]) -> Optional[np.ndarray]:
    if pos is None:
        return None
    arr = np.asarray(pos, dtype=np.float32)
    if arr.shape != (3,):
        return arr
    nap = arr[_NATNET_TO_NAPARI_AXIS]
    nap[0] *= -1.0
    return nap


def _permute_quat_to_napari(quat: Optional[Any]) -> Optional[np.ndarray]:
    if quat is None:
        return None
    arr = np.asarray(quat, dtype=np.float64)
    if arr.shape != (4,):
        return arr.astype(np.float32, copy=False)
    rot_nat = R.from_quat(arr)
    rot_nap = R.from_matrix(
        _NATNET_TO_NAPARI_WORLD @ rot_nat.as_matrix() @ _NATNET_TO_NAPARI_WORLD.T
    )
    return rot_nap.as_quat().astype(np.float32)


def _permute_markers_to_napari(markers: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
    if not markers:
        return []
    arr = np.asarray(markers, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        return [tuple(m.tolist()) for m in arr]
    arr = arr[:, _NATNET_TO_NAPARI_AXIS]
    arr[:, 0] *= -1.0
    return [tuple(m) for m in arr]

def _fake_quat():
    ang = (time.perf_counter() % 6.28)
    # no rotation
    return np.array([0, 1, 0, 1], dtype=np.float32)
    # rotation around arrow
    # return np.array([0, np.sin(ang/2), 0, np.cos(ang/2)], dtype=np.float32)

def _fake_pos():
    t = time.perf_counter()
    return np.array([0.3 + 0.1*np.cos(t), 0.5 + 0.1*np.sin(t), 0.1 + 0.05*np.sin(0.8*t)], dtype=np.float32)


def _fake_needle_pos():
    t = time.perf_counter()
    center = np.array([0.45, 0.45, 0.12], dtype=np.float32)
    radius = 0.08
    x = center[0] + radius * np.cos(t)
    y = center[1] + radius * np.sin(t)
    z = center[2]
    return np.array([x, y, z], dtype=np.float32)


def _fake_needle_quat():
    ang = (time.perf_counter() % 6.28)
    return np.array([0.0, 0.0, np.cos(ang/2)/4, 1.0], dtype=np.float32)

def _fake_markers(n=3):
    base = _fake_pos()
    offset = np.array([
        [0.01, 0, 0],
        [0, 0.01, 0],
        [0, 0, 0.01]
    ])
    jitter = np.random.uniform(-0.001, 0.001, (n, 3))
    return [tuple(base + offset[i] + jitter[i]) for i in range(n)]

class NatNetClientWrapper:
    def __init__(
        self,
        server_address: str = "127.0.0.1",
        client_address: str = "127.0.0.1",
        use_multicast: bool = False,
        print_level: int = 0,
        use_fake: bool = True,
        thread_option: str = "d",
    ) -> None:
        self._use_fake = use_fake
        self._rigid_body_data: Dict[int, Tuple[Any, Any]] = {}
        self._unlabeled_markers: List[Tuple[float, float, float]] = []
        self._marker_sets: Dict[str, List[Tuple[float, float, float]]] = {}
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None
        if thread_option not in ("c", "d"):
            raise ValueError(f"Unsupported NatNet thread option: {thread_option!r}")
        self._thread_option = thread_option

        if not self._use_fake:
            assert NatNetClient is not None, "NatNetClient import failed"
            self._client = NatNetClient()
            self._client.set_client_address(client_address)
            self._client.set_server_address(server_address)
            self._client.set_use_multicast(use_multicast)
            self._client.set_print_level(print_level)
            self._client.rigid_body_listener = self._on_rigid_body
            self._client.unlabeled_marker_listener = self._on_unlabeled_markers
            self._client.marker_set_listener = self._on_marker_sets

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # --- Real/Callback methods ---
    
    def _on_rigid_body(self, rigid_id, position, rotation):
        self._rigid_body_data[rigid_id] = (position, rotation)

    def _on_unlabeled_markers(self, unlabeled_markers_list):
        self._unlabeled_markers = unlabeled_markers_list

    def _on_marker_sets(self, marker_sets_list):
        self._marker_sets = {name: markers for name, markers in marker_sets_list}

    def start(self, wait_for_connection: float = 1.0) -> None:
        log = logging.getLogger(__name__)
        if self._use_fake:
            self._running = True
            log.info("NatNetClientWrapper.start(use_fake=True)")
            return
        if self._running:
            return
        self._running = True
        log.info("NatNetClientWrapper.start(use_fake=False)")
        def runner():
            ok = self._client.run(self._thread_option)
            if not ok:
                raise RuntimeError("NatNetClient failed to start")
        self._thread = threading.Thread(target=runner, daemon=False)
        self._thread.start()
        time.sleep(wait_for_connection)
        connected = False
        try:
            connected = bool(self._client.connected())
        except Exception:
            connected = False
        if not connected:
            self._running = False
            log.error("NatNetClient did not connect; check streaming on Motive")
            raise RuntimeError("NatNetClient did not connect; check streaming on Motive")
        log.info("NatNetClientWrapper.start connected=%s", connected)

    def stop(self) -> None:
        log = logging.getLogger(__name__)
        if not self._running:
            return
        log.info("NatNetClientWrapper.stop(use_fake=%s)", self._use_fake)
        if self._use_fake:
            self._running = False
            return
        self._client.shutdown()
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)
            self._thread = None
        self._rigid_body_data.clear()
        self._unlabeled_markers.clear()
        self._marker_sets.clear()

    # --- Main API, returns fake or real data ---
    def get_rigid_body_pose(self, rigid_id: int):
        poses = self.get_rigid_body_poses()
        pose = poses.get(rigid_id)
        if pose is None:
            return None, None
        return pose

    def get_rigid_body_poses(self) -> Dict[int, Tuple[Optional[np.ndarray], Optional[np.ndarray]]]:
        if self._use_fake:
            return {
                0: (
                    _permute_position_to_napari(_fake_pos()),
                    _permute_quat_to_napari(_fake_quat()),
                ),
                1: (
                    _permute_position_to_napari(_fake_needle_pos()),
                    _permute_quat_to_napari(_fake_needle_quat()),
                ),
            }
        poses: Dict[int, Tuple[Optional[np.ndarray], Optional[np.ndarray]]] = {}
        for rigid_id, (pos, quat) in self._rigid_body_data.items():
            pos_p = _permute_position_to_napari(pos)
            quat_p = _permute_quat_to_napari(quat)
            if pos_p is not None and quat_p is not None:
                poses[rigid_id] = (pos_p, quat_p)
        return poses

    def get_unlabeled_markers(self):
        if self._use_fake:
            markers = _fake_markers()
        else:
            markers = self._unlabeled_markers
        return _permute_markers_to_napari(markers)

    def get_marker_sets(self):
        if self._use_fake:
            marker_sets = {"fake_set": _fake_markers(3)}
        else:
            marker_sets = self._marker_sets
        return {name: _permute_markers_to_napari(markers) for name, markers in marker_sets.items()}
