import time
import numpy as np
import logging

def _fake_probe_motion(t):
    return np.array([
        0.3 + 0.1 * np.cos(t),
        0.5 + 0.1 * np.sin(t),
        0.1 + 0.05 * np.sin(0.5 * t)
    ], dtype=np.float32)


def _fake_probe_quat(t):
    # simple stable orientation
    return np.array([0, 0, 0, 1], dtype=np.float32)


def _fake_needle_motion(t):
    center = np.array([0.45, 0.45, 0.12], dtype=np.float32)
    r = 0.08
    return np.array([
        center[0] + r * np.cos(t),
        center[1] + r * np.sin(t),
        center[2]
    ], dtype=np.float32)


def _fake_needle_quat(t):
    return np.array([0, 0, np.sin(t/2)*0.2, 1], dtype=np.float32)


class TrackingClient:

    def __init__(self):
        self._log = logging.getLogger(__name__)
        self._running = False

    def start(self):
        self._running = True
        self._log.info("Fake Tracking started")

    def stop(self):
        self._running = False
        self._log.info("Fake Tracking stopped")

    def get_poses(self):
        if not self._running:
            return {}

        t = time.perf_counter()

        return {
            0: (_fake_probe_motion(t), _fake_probe_quat(t), t),
            1: (_fake_needle_motion(t), _fake_needle_quat(t), t),
        }

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()