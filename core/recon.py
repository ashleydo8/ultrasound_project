from typing import List, Tuple, Sequence, Union, Optional
import numpy as np

_NAPARI_TO_WORLD_PERM = np.array([2, 1, 0], dtype=np.int32)
_NAPARI_TO_WORLD_MATRIX = np.eye(3, dtype=np.float64)[_NAPARI_TO_WORLD_PERM]
_WORLD_TO_NAPARI_MATRIX = _NAPARI_TO_WORLD_MATRIX.T


def _quat_to_R_wxyz(q: Sequence[float]) -> np.ndarray:
    w, x, y, z = map(float, q)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n == 0:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / n, x / n, y / n, z / n
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _quat_to_R_xyzw(q: Sequence[float]) -> np.ndarray:
    x, y, z, w = map(float, q)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n == 0:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / n, x / n, y / n, z / n
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _pose_to_matrix(pose) -> np.ndarray:
    """Map a Pose-like object to a 4x4 world_T_image matrix in mm."""
    pos_mm = None
    napari_basis = getattr(pose, "napari_basis", False)
    if hasattr(pose, "pos_mm"):
        pos_mm = np.asarray(pose.pos_mm, dtype=np.float64)
    elif hasattr(pose, "position"):
        pos_mm = np.asarray(getattr(pose, "position"), dtype=np.float64)
    elif hasattr(pose, "xyz"):
        pos_mm = np.asarray(getattr(pose, "xyz"), dtype=np.float64)
    elif hasattr(pose, "pos_m"):
        pos_mm = np.asarray(getattr(pose, "pos_m"), dtype=np.float64) * 1000.0
    if pos_mm is None:
        raise AttributeError("Pose translation must provide pos_mm, position, xyz, or pos_m")
    pos_mm = pos_mm.reshape(3)
    if napari_basis:
        pos_mm = pos_mm[_NAPARI_TO_WORLD_PERM]

    if hasattr(pose, "body_rot") and hasattr(pose, "align_rot"):
        rot_mat = (pose.body_rot * pose.align_rot).as_matrix()
        napari_basis = getattr(pose, "napari_basis", napari_basis)
    else:
        quat = getattr(pose, "quat", None)
        if quat is None:
            quat = getattr(pose, "quaternion", None)
        if quat is None:
            raise AttributeError("Pose must expose quaternion as 'quat' or 'quaternion'")
        quat = np.asarray(quat, dtype=np.float64).reshape(4)
        # Heuristic: default to xyzw (SciPy) ordering but support wxyz.
        if abs(quat[0]) > abs(quat[-1]):
            rot_mat = _quat_to_R_wxyz(quat)
        else:
            rot_mat = _quat_to_R_xyzw(quat)
        napari_basis = False
    if napari_basis:
        rot_mat = _WORLD_TO_NAPARI_MATRIX @ rot_mat @ _NAPARI_TO_WORLD_MATRIX

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rot_mat
    T[:3, 3] = pos_mm
    return T


def reconstruct_volume_from_frames(
    recorded_frames: List[Tuple[float, np.ndarray, object]],
    px_per_mm: Union[float, Tuple[float, float]] = 10.0,
    voxel_size_mm: float = 0.75,
    pixel_step: int = 1,
    frame_step: int = 1,
    inplane_origin: str = "center",
    intensity_scale: Optional[float] = None,
    return_weights: bool = True,
    margin_mm: float = 3.0,
):
    assert len(recorded_frames) > 0, "No frames to reconstruct."

    def _unpack(rec):
        """
        Accept legacy 3-tuples (ts, frame, pose) or newer 4-tuples
        (ts, frame, pose, spacing_mm). Extra fields are ignored here.
        """
        if len(rec) < 3:
            raise ValueError("Recorded frame entries must have at least (ts, frame, pose).")
        ts, frame, pose = rec[0], rec[1], rec[2]
        return ts, frame, pose

    if isinstance(px_per_mm, (tuple, list, np.ndarray)):
        px_per_mm_x, px_per_mm_y = float(px_per_mm[0]), float(px_per_mm[1])
    else:
        px_per_mm_x = px_per_mm_y = float(px_per_mm)
    mm_per_px_x = 1.0 / px_per_mm_x
    mm_per_px_y = 1.0 / px_per_mm_y

    if intensity_scale is None:
        _, samp, _ = _unpack(recorded_frames[0])
        if np.issubdtype(samp.dtype, np.integer):
            intensity_scale = 1.0 / float(np.iinfo(samp.dtype).max)
        else:
            intensity_scale = 1.0

    _, first_frame, _ = _unpack(recorded_frames[0])
    H0, W0 = first_frame.shape[:2]
    if inplane_origin == "center":
        cx, cy = (W0 - 1) / 2.0, (H0 - 1) / 2.0
        corners_px = np.array(
            [
                [0 - cx, 0 - cy, 0.0],
                [W0 - 1 - cx, 0 - cy, 0.0],
                [0 - cx, H0 - 1 - cy, 0.0],
                [W0 - 1 - cx, H0 - 1 - cy, 0.0],
            ],
            dtype=np.float64,
        )
    elif inplane_origin == "topcenter":
        cx = (W0 - 1) / 2.0
        corners_px = np.array(
            [
                [0 - cx, 0.0, 0.0],
                [W0 - 1 - cx, 0.0, 0.0],
                [0 - cx, H0 - 1.0, 0.0],
                [W0 - 1 - cx, H0 - 1.0, 0.0],
            ],
            dtype=np.float64,
        )
    elif inplane_origin == "topleft":
        corners_px = np.array(
            [
                [0.0, 0.0, 0.0],
                [W0 - 1.0, 0.0, 0.0],
                [0.0, H0 - 1.0, 0.0],
                [W0 - 1.0, H0 - 1.0, 0.0],
            ],
            dtype=np.float64,
        )
    else:
        raise ValueError("inplane_origin must be 'center', 'topcenter', or 'topleft'.")

    corners_mm = corners_px * np.array([mm_per_px_x, mm_per_px_y, 1.0], dtype=np.float64)

    mins, maxs = [], []
    ch = np.c_[corners_mm, np.ones((4, 1))]
    for i in range(0, len(recorded_frames), frame_step):
        _, _, pose = _unpack(recorded_frames[i])
        T = _pose_to_matrix(pose)
        world = (T @ ch.T).T[:, :3]
        mins.append(world.min(axis=0))
        maxs.append(world.max(axis=0))
    mins = np.min(np.stack(mins), axis=0) - margin_mm
    maxs = np.max(np.stack(maxs), axis=0) + margin_mm

    size_mm = maxs - mins
    nx = int(np.ceil(size_mm[0] / voxel_size_mm)) + 1
    ny = int(np.ceil(size_mm[1] / voxel_size_mm)) + 1
    nz = int(np.ceil(size_mm[2] / voxel_size_mm)) + 1
    vol_sum = np.zeros((nz, ny, nx), dtype=np.float32)
    w_sum = np.zeros((nz, ny, nx), dtype=np.float32)

    cols = np.arange(0, W0, pixel_step, dtype=np.float64)
    rows = np.arange(0, H0, pixel_step, dtype=np.float64)
    if inplane_origin == "center":
        cols = (cols - (W0 - 1) / 2.0) * mm_per_px_x
        rows = (rows - (H0 - 1) / 2.0) * mm_per_px_y
    elif inplane_origin == "topcenter":
        cols = (cols - (W0 - 1) / 2.0) * mm_per_px_x
        rows = rows * mm_per_px_y
    else:
        cols = cols * mm_per_px_x
        rows = rows * mm_per_px_y
    Xmm, Ymm = np.meshgrid(cols, rows, indexing="xy")
    plane_pts = np.stack(
        [Xmm.ravel(), Ymm.ravel(), np.zeros_like(Xmm).ravel()], axis=1
    )

    def _splat(pts_vox: np.ndarray, vals: np.ndarray) -> None:
        xv, yv, zv = pts_vox[:, 0], pts_vox[:, 1], pts_vox[:, 2]
        x0 = np.floor(xv).astype(np.int64)
        y0 = np.floor(yv).astype(np.int64)
        z0 = np.floor(zv).astype(np.int64)
        dx, dy, dz = xv - x0, yv - y0, zv - z0
        for ox in (0, 1):
            wx = (1.0 - dx) if ox == 0 else dx
            ix = x0 + ox
            for oy in (0, 1):
                wy = (1.0 - dy) if oy == 0 else dy
                iy = y0 + oy
                for oz in (0, 1):
                    wz = (1.0 - dz) if oz == 0 else dz
                    iz = z0 + oz
                    w = wx * wy * wz
                    m = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz)
                    if not np.any(m):
                        continue
                    ii, jj, kk = iz[m], iy[m], ix[m]
                    ww = (w[m]).astype(np.float32)
                    vv = (vals[m] * ww).astype(np.float32)
                    np.add.at(vol_sum, (ii, jj, kk), vv)
                    np.add.at(w_sum, (ii, jj, kk), ww)

    for i in range(0, len(recorded_frames), frame_step):
        _, frame, pose = _unpack(recorded_frames[i])
        fr = frame[::pixel_step, ::pixel_step].astype(np.float32, copy=False)
        vals = fr.ravel() * float(intensity_scale)
        T = _pose_to_matrix(pose)
        pts_h = np.c_[plane_pts, np.ones((plane_pts.shape[0], 1))]
        world = (T @ pts_h.T).T[:, :3]
        rel = (world - mins.reshape(1, 3)) / voxel_size_mm
        _splat(rel, vals)

    with np.errstate(invalid="ignore", divide="ignore"):
        vol = vol_sum / w_sum
        vol[~np.isfinite(vol)] = 0.0

    meta = {
        "voxel_size_mm": float(voxel_size_mm),
        "origin_mm": mins.astype(np.float64),
        "bbox_min_mm": mins.astype(np.float64),
        "bbox_max_mm": maxs.astype(np.float64),
        "shape_zyx": vol.shape,
        "px_per_mm": (px_per_mm_x, px_per_mm_y),
        "pixel_step": int(pixel_step),
        "frame_step": int(frame_step),
        "inplane_origin": inplane_origin,
        "intensity_scale": float(intensity_scale),
    }
    return vol.astype(np.float32), (w_sum if return_weights else None), meta
