"""Frame geometry: 90° rotation correction + camera intrinsics.

Camera params (ТЗ «Параметры съёмки»):
  - native resolution 3840 x 2160 (16:9)
  - sensor diagonal = 16/2.8 mm  ≈ 5.714 mm  (the literal "16/2.8 мм")
  - focal length f = 2.8 mm
  - camera physically rotated 90° COUNTER-clockwise

Derivation of pixel focal length (transparent, not hardcoded):
  diag_mm = 16/2.8 ≈ 5.714
  for 16:9:  w_mm = diag * 16/√(16²+9²),  h_mm = diag * 9/√(16²+9²)
  fx = f_mm * W_px / w_mm ≈ 2159 px ; fy = f_mm * H_px / h_mm ≈ 2159 px
  (cx, cy) = image centre

NOTE (validate in Phase 0 against reference CSV): output bbox coordinates
are produced in the ROTATED (upright, human-viewable) frame. If the
reference CSV uses native orientation, flip via config `output.coord_space`.
At ~95° FOV there is noticeable barrel distortion, but barcodes/QR tolerate
it; undistort is OFF by default and only enabled if it empirically improves
decode rate (see config `preprocess.undistort`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

NATIVE_W, NATIVE_H = 3840, 2160
SENSOR_DIAG_MM = 16.0 / 2.8
FOCAL_MM = 2.8


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]],
            dtype=np.float64,
        )


def compute_intrinsics(width: int = NATIVE_W, height: int = NATIVE_H) -> Intrinsics:
    diag_px_ratio = math.hypot(16.0, 9.0)
    w_mm = SENSOR_DIAG_MM * 16.0 / diag_px_ratio
    h_mm = SENSOR_DIAG_MM * 9.0 / diag_px_ratio
    fx = FOCAL_MM * width / w_mm
    fy = FOCAL_MM * height / h_mm
    return Intrinsics(fx=fx, fy=fy, cx=width / 2.0, cy=height / 2.0)


def rotate_upright(frame: np.ndarray) -> np.ndarray:
    """Undo the physical 90° CCW mounting -> rotate frame 90° CW."""
    return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)


def undistort(frame: np.ndarray, k1: float, k2: float) -> np.ndarray:
    """Approximate radial undistortion (pinhole + 2-term radial).

    k1, k2 are not provided by ТЗ — estimate via straight shelf-line
    minimisation in Phase 0 or leave undistort disabled.
    """
    h, w = frame.shape[:2]
    intr = compute_intrinsics(w, h)
    dist = np.array([k1, k2, 0.0, 0.0, 0.0], dtype=np.float64)
    new_K, _ = cv2.getOptimalNewCameraMatrix(intr.K, dist, (w, h), 0)
    return cv2.undistort(frame, intr.K, dist, None, new_K)
