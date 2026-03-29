"""Needle pivot calibration utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import json

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import least_squares


def _project_to_so3(R: np.ndarray) -> np.ndarray:
    """Return the closest rotation matrix in SO(3) using SVD."""

    u, _, vh = np.linalg.svd(R)
    R_proj = u @ vh
    if np.linalg.det(R_proj) < 0:
        u[:, -1] *= -1.0
        R_proj = u @ vh
    return R_proj


def _rotation_angle(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """Return rotation angle in radians between two rotation matrices."""

    delta = R_a.T @ R_b
    trace_val = np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(trace_val))


@dataclass
class _PoseSample:
    R: np.ndarray
    t: np.ndarray
    quality: float
    ts: Optional[float]


class NeedlePivotCalibrator:
    """Robust estimator for fixed needle tip offset using pivot calibration."""

    def __init__(
        self,
        huber_delta_mm: float = 3.0,
        ransac_thresh_mm: float = 2.0,
        ransac_min_samples: int = 2,
        ransac_max_trials: int = 500,
        quality_weighting: bool = True,
    ) -> None:
        self.huber_delta_mm = float(huber_delta_mm)
        self.ransac_thresh_mm = float(ransac_thresh_mm)
        self.ransac_min_samples = int (ransac_min_samples)
        self.ransac_max_trials = int(ransac_max_trials)
        self.quality_weighting = bool(quality_weighting)

        self._samples: List[_PoseSample] = []
        self._last_result: Optional[Dict[str, object]] = None
        self._rng = np.random.default_rng(0)

    def reset(self) -> None:
        self._samples.clear()
        self._last_result = None

    def add_pose(
        self,
        R_i: ArrayLike,
        t_i: ArrayLike,
        quality: float | None = None,
        ts: float | None = None,
    ) -> None:
        R_i = np.asarray(R_i, dtype=np.float64)
        t_i = np.asarray(t_i, dtype=np.float64)
        if R_i.shape != (3, 3):
            raise ValueError("R_i must be shape (3,3)")
        if t_i.shape not in {(3,), (3, 1)}:
            raise ValueError("t_i must be a length-3 vector")
        t_i = t_i.reshape(3)
        R_i = _project_to_so3(R_i)
        q = 1.0 if quality is None else float(quality)
        q = float(np.clip(q, 0.0, 1.0))
        self._samples.append(_PoseSample(R_i, t_i, q, ts))
        self._last_result = None

    def num_samples(self) -> int:
        return len(self._samples)

    def has_sufficient_diversity(self) -> bool:
        n = len(self._samples)
        if n < 8:
            return False
        Rs = [s.R for s in self._samples]
        angles: List[float] = []
        for i in range(n):
            for j in range(i + 1, n):
                angles.append(_rotation_angle(Rs[i], Rs[j]))
        if not angles:
            return False
        avg_angle_deg = np.degrees(np.mean(angles))
        return avg_angle_deg >= 10.0

    def _stack_system(self, mask: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
        samples = self._samples
        if mask is None:
            mask = np.ones(len(samples), dtype=bool)
        Rs = []
        ts = []
        for active, s in zip(mask, samples):
            if not active:
                continue
            Rs.append(s.R)
            ts.append(s.t)
        if not Rs:
            raise ValueError("No samples selected")
        A = np.zeros((3 * len(Rs), 6), dtype=np.float64)
        b = np.zeros((3 * len(Rs),), dtype=np.float64)
        for idx, (R, t) in enumerate(zip(Rs, ts)):
            row = slice(3 * idx, 3 * idx + 3)
            A[row, :3] = R
            A[row, 3:] = -np.eye(3)
            b[row] = -t
        return A, b

    def _least_squares(self, mask: Optional[np.ndarray] = None) -> np.ndarray:
        A, b = self._stack_system(mask)
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        return x

    def has_data(self) -> bool:
        return bool(self._samples)

    def solve(self) -> Dict[str, object]:
        if not self.has_sufficient_diversity():
            n = len(self._samples)
            if n < 8:
                raise ValueError("Need at least 8 poses for calibration")
            Rs = [s.R for s in self._samples]
            angles: List[float] = []
            for i in range(len(Rs)):
                for j in range(i + 1, len(Rs)):
                    angles.append(_rotation_angle(Rs[i], Rs[j]))
            avg_angle_deg = np.degrees(np.mean(angles)) if angles else 0.0
            raise ValueError(
                f"Add more tilt/roll; current average rotation spread = {avg_angle_deg:.2f}°"
            )

        n_samples = len(self._samples)
        inlier_mask = np.ones(n_samples, dtype=bool)
        best_rms = np.inf
        best_mask = np.zeros(n_samples, dtype=bool)
        best_model = None

        if n_samples >= max(self.ransac_min_samples, 2) and self.ransac_max_trials > 0:
            thresh = self.ransac_thresh_mm
            for _ in range(self.ransac_max_trials):
                indices = self._rng.choice(n_samples, size=self.ransac_min_samples, replace=False)
                mask = np.zeros(n_samples, dtype=bool)
                mask[indices] = True
                try:
                    x = self._least_squares(mask)
                except np.linalg.LinAlgError:
                    continue
                residuals = self._raw_residuals(x)
                norms = np.linalg.norm(residuals.reshape(-1, 3), axis=1)
                mask = norms <= thresh
                if not np.any(mask):
                    continue
                x_full = self._least_squares(mask)
                residuals_full = self._raw_residuals(x_full)
                norms_full = np.linalg.norm(residuals_full.reshape(-1, 3), axis=1)
                rms = float(np.sqrt(np.mean(norms_full[mask] ** 2)))
                inlier_count = int(np.sum(mask))
                best_inliers = int(np.sum(best_mask))
                if inlier_count > best_inliers or (inlier_count == best_inliers and rms < best_rms):
                    best_rms = rms
                    best_mask = mask
                    best_model = x_full

            if best_model is not None:
                inlier_mask = best_mask

        if best_model is None:
            x0 = self._least_squares(inlier_mask)
        else:
            x0 = best_model

        def residual_func(theta: np.ndarray) -> np.ndarray:
            r = theta[:3]
            P = theta[3:]
            res = []
            for idx, s in enumerate(self._samples):
                if not inlier_mask[idx]:
                    continue
                e = s.R @ r + s.t - P
                w = 1.0
                if self.quality_weighting:
                    w = max(float(s.quality), 1e-6)
                res.append(np.sqrt(w) * e)
            return np.concatenate(res)

        result = least_squares(
            residual_func,
            x0,
            loss="huber",
            f_scale=self.huber_delta_mm,
        )
        theta = result.x
        r_est = theta[:3]
        P_est = theta[3:]
        raw_residuals = self._raw_residuals(theta)
        norms = np.linalg.norm(raw_residuals.reshape(-1, 3), axis=1)
        inlier_mask = norms <= (self.ransac_thresh_mm if best_model is not None else np.inf)
        if not np.any(inlier_mask):
            inlier_mask = np.ones_like(norms, dtype=bool)

        rms = float(np.sqrt(np.mean(norms[inlier_mask] ** 2)))
        max_abs = float(np.max(norms))
        n_inliers = int(np.sum(inlier_mask))
        n_outliers = n_samples - n_inliers

        cov_est = None
        if result.jac is not None and result.jac.size:
            m, n = result.jac.shape
            if m > n:
                JTJ = result.jac.T @ result.jac
                try:
                    JTJ_inv = np.linalg.inv(JTJ)
                    sigma2 = 2.0 * result.cost / max(m - n, 1)
                    cov_est = JTJ_inv * sigma2
                except np.linalg.LinAlgError:
                    cov_est = None

        summary = (
            f"Pivot RMS={rms:.2f} mm ({n_inliers}/{n_samples} inliers). "
            f"r_H_tip=[{r_est[0]:.2f}, {r_est[1]:.2f}, {r_est[2]:.2f}] mm"
        )

        solution = {
            "r_H_tip": r_est,
            "P_W_pivot": P_est,
            "rms_mm": rms,
            "max_abs_residual_mm": max_abs,
            "n_samples": n_samples,
            "n_inliers": n_inliers,
            "n_outliers": n_outliers,
            "inlier_mask": inlier_mask,
            "cov_est": cov_est,
            "summary": summary,
        }
        self._last_result = solution
        return solution

    def _raw_residuals(self, theta: np.ndarray) -> np.ndarray:
        r = theta[:3]
        P = theta[3:]
        residuals = []
        for s in self._samples:
            residuals.append(s.R @ r + s.t - P)
        return np.concatenate(residuals)

    def save_json(self, path: str, meta: dict | None = None) -> None:
        if self._last_result is None:
            raise ValueError("No calibration result available to save")
        data = {
            "calibrator_config": {
                "huber_delta_mm": self.huber_delta_mm,
                "ransac_thresh_mm": self.ransac_thresh_mm,
                "ransac_min_samples": self.ransac_min_samples,
                "ransac_max_trials": self.ransac_max_trials,
                "quality_weighting": self.quality_weighting,
            },
            "result": {
                key: (value.tolist() if isinstance(value, np.ndarray) else value)
                for key, value in self._last_result.items()
            },
            "meta": meta or {},
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load_json(path: str) -> "NeedlePivotCalibrator":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        cfg = payload.get("calibrator_config", {})
        cal = NeedlePivotCalibrator(**cfg)
        result = payload.get("result")
        if result is not None:
            converted = {}
            for key, value in result.items():
                if key in {"r_H_tip", "P_W_pivot", "inlier_mask"}:
                    converted[key] = np.asarray(value, dtype=np.float64)
                elif key == "cov_est" and value is not None:
                    converted[key] = np.asarray(value, dtype=np.float64)
                else:
                    converted[key] = value
            cal._last_result = converted
        return cal

    @staticmethod
    def apply_tip(T_W_H: np.ndarray, r_H_tip: np.ndarray) -> np.ndarray:
        R = T_W_H[:3, :3]
        t = T_W_H[:3, 3]
        return R @ r_H_tip + t