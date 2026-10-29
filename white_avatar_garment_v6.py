# -*- coding: utf-8 -*-
"""
White Avatar Garment Motion Mirror v6.0
===================================================
This desktop app loads a garment image whose background is a connected green
screen, removes only that background, preserves the uploaded pixels exactly,
and deforms the garment with MediaPipe upper-body pose landmarks.

Recommended environment: Python 3.10 / Windows
Run: python white_avatar_garment_v6.py
"""

from __future__ import annotations

import math
import os
import queue
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    import cv2
    import numpy as np
    from PIL import Image, ImageTk
except ImportError as exc:
    raise SystemExit(
        "Missing dependencies. Run:\n"
        "  python -m pip install -r requirements.txt\n"
        f"Original error: {exc}"
    ) from exc


DEFAULT_WINDOW_W, DEFAULT_WINDOW_H = 450, 650
GREEN_BGR = (0, 255, 0)
GRID_COLS, GRID_ROWS = 9, 13

UPPER_BODY_CONNECTIONS = frozenset({
    (0, 11), (0, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
})

CALIBRATION_POINTS = [
    ("shoulder_l", "L shoulder (viewer left)", "#ff3b30"),
    ("shoulder_r", "R shoulder (viewer right)", "#007aff"),
    ("elbow_l", "L elbow", "#ff9500"),
    ("elbow_r", "R elbow", "#34c759"),
    ("wrist_l", "L wrist", "#af52de"),
    ("wrist_r", "R wrist", "#00c7be"),
]
CALIBRATION_KEYS = [item[0] for item in CALIBRATION_POINTS]


def load_image_unicode(path: str) -> Optional[np.ndarray]:
    """Load an image from a path that may contain non-ASCII characters."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    except Exception:
        return None


def save_image_unicode(path: str, image_bgr: np.ndarray) -> bool:
    """Save an OpenCV image to a path that may contain non-ASCII characters."""
    try:
        ext = os.path.splitext(path)[1] or ".png"
        success, encoded = cv2.imencode(ext, image_bgr)
        if not success:
            return False
        encoded.tofile(path)
        return True
    except Exception:
        return False


def point_segment_distances(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the distance from every point in points to segment a-b."""
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-8:
        return np.linalg.norm(points - a[None, :], axis=1)
    t = ((points - a[None, :]) @ ab) / denom
    t = np.clip(t, 0.0, 1.0)
    closest = a[None, :] + t[:, None] * ab[None, :]
    return np.linalg.norm(points - closest, axis=1)


@dataclass
class Bone:
    """A segment, similarity, or affine transform used by the garment mesh."""

    source: np.ndarray
    target: np.ndarray
    mode: str = "segment"

    def map_points(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32)

        if self.mode == "similarity":
            src0, src1 = self.source[0], self.source[1]
            dst0, dst1 = self.target[0], self.target[1]
            src_vec = src1 - src0
            dst_vec = dst1 - dst0
            src_len = float(np.linalg.norm(src_vec))
            dst_len = float(np.linalg.norm(dst_vec))
            if src_len < 1e-8 or dst_len < 1e-8:
                return points + (dst0 - src0)[None, :]
            src_along = src_vec / src_len
            src_perp = np.array([-src_along[1], src_along[0]], dtype=np.float32)
            dst_along = dst_vec / dst_len
            dst_perp = np.array([-dst_along[1], dst_along[0]], dtype=np.float32)
            rel = points - src0[None, :]
            local_x = rel @ src_along
            local_y = rel @ src_perp
            uniform_scale = dst_len / src_len
            mapped = (
                dst0[None, :]
                + (local_x * uniform_scale)[:, None] * dst_along[None, :]
                + (local_y * uniform_scale)[:, None] * dst_perp[None, :]
            )
            return mapped.astype(np.float32)

        if self.mode == "affine" and self.source.shape[0] >= 3:
            src0, src1, src2 = self.source[:3]
            dst0, dst1, dst2 = self.target[:3]
            src_basis = np.column_stack((src1 - src0, src2 - src0)).astype(np.float32)
            dst_basis = np.column_stack((dst1 - dst0, dst2 - dst0)).astype(np.float32)
            try:
                coeff = np.linalg.solve(src_basis, (points - src0[None, :]).T).T
                mapped = dst0[None, :] + coeff @ dst_basis.T
                return mapped.astype(np.float32)
            except np.linalg.LinAlgError:
                pass

        src0, src1 = self.source[0], self.source[1]
        dst0, dst1 = self.target[0], self.target[1]
        src_vec = src1 - src0
        dst_vec = dst1 - dst0
        src_len = float(np.linalg.norm(src_vec))
        dst_len = float(np.linalg.norm(dst_vec))

        if src_len < 1e-8 or dst_len < 1e-8:
            return points + (dst0 - src0)[None, :]

        src_along = src_vec / src_len
        src_perp = np.array([-src_along[1], src_along[0]], dtype=np.float32)
        dst_along = dst_vec / dst_len
        dst_perp = np.array([-dst_along[1], dst_along[0]], dtype=np.float32)

        rel = points - src0[None, :]
        along = (rel @ src_along) / src_len
        perp = rel @ src_perp
        scale = max(dst_len / src_len, 1e-4)
        thickness_scale = float(np.clip(scale ** -0.5, 0.62, 1.55))
        mapped = (
            dst0[None, :]
            + along[:, None] * dst_vec[None, :]
            + (perp * thickness_scale)[:, None] * dst_perp[None, :]
        )
        return mapped.astype(np.float32)


@dataclass
class BoneSpec:
    name: str
    source: np.ndarray
    target_keys: List[str]
    weight_segments: List[Tuple[np.ndarray, np.ndarray]]
    bias: float = 1.0
    mode: str = "segment"


class GarmentRenderer:
    """Render untouched garment pixels through an upper-body skinning mesh."""

    JOINT_INDEX = {
        "shoulder_l": 12,
        "shoulder_r": 11,
        "elbow_l": 14,
        "elbow_r": 13,
        "wrist_l": 16,
        "wrist_r": 15,
        "hip_l": 24,
        "hip_r": 23,
        "nose": 0,
    }

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.empty = True
        self.rgba: Optional[np.ndarray] = None
        self.source_w = 0
        self.source_h = 0
        self.scale_factor = 1.0
        self.motion_strength = 0.85
        self.manual_calibration = False
        self.background_bgr: Optional[np.ndarray] = None
        self.background_cache_size: Optional[Tuple[int, int]] = None
        self.background_cache_image: Optional[np.ndarray] = None
        self.source_joints: Dict[str, np.ndarray] = {}
        self.bone_specs: List[BoneSpec] = []
        self.bone_names: List[str] = []
        self.weights: Optional[np.ndarray] = None
        self.arm_weights: Optional[np.ndarray] = None
        self.arm_influence: Optional[np.ndarray] = None
        self.normalized_grid, self.triangles = self._make_normalized_grid()
        self.smooth_joints: Dict[str, np.ndarray] = {}
        self.smooth_visibility: Dict[str, float] = {}

    @staticmethod
    def _make_normalized_grid() -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
        xs = np.linspace(0.0, 1.0, GRID_COLS, dtype=np.float32)
        ys = np.linspace(0.0, 1.0, GRID_ROWS, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(xs, ys)
        grid = np.stack((grid_x, grid_y), axis=-1).astype(np.float32)

        triangles: List[Tuple[int, int, int]] = []
        for row in range(GRID_ROWS - 1):
            for col in range(GRID_COLS - 1):
                a = row * GRID_COLS + col
                b = a + 1
                c = a + GRID_COLS
                d = c + 1
                triangles.append((a, b, d))
                triangles.append((a, d, c))
        return grid, triangles

    @staticmethod
    def _remove_green_background(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Remove only the green region connected to the image border."""
        if image.ndim == 2:
            bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            source_alpha = np.full(image.shape, 255, dtype=np.uint8)
        elif image.shape[2] == 4:
            bgr = image[:, :, :3].copy()
            source_alpha = image[:, :, 3].copy()
        else:
            bgr = image[:, :, :3].copy()
            source_alpha = np.full(image.shape[:2], 255, dtype=np.uint8)

        if source_alpha.dtype != np.uint8:
            source_alpha = cv2.convertScaleAbs(source_alpha)

        b, g, r = cv2.split(bgr)
        r16 = r.astype(np.int16)
        g16 = g.astype(np.int16)
        b16 = b.astype(np.int16)

        exact_green = (r16 == 0) & (g16 == 255) & (b16 == 0)
        near_green = (
            (g16 > 165)
            & (r16 < 125)
            & (b16 < 125)
            & (g16 > r16 + 55)
            & (g16 > b16 + 55)
        )
        green_candidate = (exact_green | near_green).astype(np.uint8)

        _, labels = cv2.connectedComponents(green_candidate, connectivity=8)
        border_labels = set()
        for edge in (labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]):
            border_labels.update(int(value) for value in np.unique(edge) if int(value) != 0)

        background = np.zeros_like(green_candidate, dtype=bool)
        for label in border_labels:
            background |= labels == label

        mask = (~background).astype(np.uint8) * 255
        mask = cv2.bitwise_and(mask, source_alpha)
        return bgr, mask

    def set_garment(self, image: np.ndarray) -> bool:
        """Keep the cropped source pixels unchanged and create garment skinning."""
        if image is None or image.size == 0:
            return False

        bgr, mask = self._remove_green_background(image)
        ys, xs = np.where(mask > 20)
        if xs.size < 50:
            return False

        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        crop_bgr = bgr[y0:y1, x0:x1]
        crop_mask = mask[y0:y1, x0:x1]

        # No resizing, color quantization, outline, or pixel-art conversion is done.
        rgba = np.zeros((crop_bgr.shape[0], crop_bgr.shape[1], 4), dtype=np.uint8)
        rgba[:, :, :3] = crop_bgr
        rgba[:, :, 3] = crop_mask

        with self.lock:
            self.empty = False
            self.rgba = rgba
            self.source_h, self.source_w = rgba.shape[:2]
            self.source_joints = self._template_joints(self.source_w, self.source_h)
            self.manual_calibration = False
            self._rebuild_skin()
            self.reset_pose()
        return True

    def set_background(self, image: np.ndarray) -> bool:
        """Store an uploaded background and clear the fitted-image cache."""
        if image is None or image.size == 0:
            return False

        if image.ndim == 2:
            background = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            bgr = image[:, :, :3]
            alpha = image[:, :, 3:4].astype(np.float32) / 255.0
            base = np.full(bgr.shape, GREEN_BGR, dtype=np.uint8)
            background = np.clip(
                bgr.astype(np.float32) * alpha
                + base.astype(np.float32) * (1.0 - alpha),
                0,
                255,
            ).astype(np.uint8)
        else:
            background = image[:, :, :3].copy()

        if background.dtype != np.uint8:
            background = cv2.convertScaleAbs(background)

        with self.lock:
            self.background_bgr = background.copy()
            self.background_cache_size = None
            self.background_cache_image = None
        return True

    def use_green_background(self) -> None:
        with self.lock:
            self.background_bgr = None
            self.background_cache_size = None
            self.background_cache_image = None

    def _background_for_size(self, out_w: int, out_h: int) -> np.ndarray:
        key = (int(out_w), int(out_h))
        with self.lock:
            if self.background_bgr is None:
                return np.full((out_h, out_w, 3), GREEN_BGR, dtype=np.uint8)
            if (
                self.background_cache_size == key
                and self.background_cache_image is not None
            ):
                return self.background_cache_image

        source = self.background_bgr
        source_h, source_w = source.shape[:2]
        if source_w <= 0 or source_h <= 0:
            return np.full((out_h, out_w, 3), GREEN_BGR, dtype=np.uint8)

        # Cover the complete output window while preserving the background aspect ratio.
        scale = max(out_w / source_w, out_h / source_h)
        fitted_w = max(out_w, int(math.ceil(source_w * scale)))
        fitted_h = max(out_h, int(math.ceil(source_h * scale)))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        fitted = cv2.resize(source, (fitted_w, fitted_h), interpolation=interpolation)
        x0 = max(0, (fitted_w - out_w) // 2)
        y0 = max(0, (fitted_h - out_h) // 2)
        result = fitted[y0:y0 + out_h, x0:x0 + out_w].copy()

        with self.lock:
            self.background_cache_size = key
            self.background_cache_image = result
        return result

    def get_source_size(self) -> Tuple[int, int]:
        with self.lock:
            return self.source_w, self.source_h

    def get_source_rgba(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.rgba is None:
                return None
            return self.rgba.copy()

    def get_source_joints(self) -> Dict[str, np.ndarray]:
        with self.lock:
            return {
                key: self.source_joints[key].copy()
                for key in CALIBRATION_KEYS
                if key in self.source_joints
            }

    def get_default_joints(self) -> Dict[str, np.ndarray]:
        with self.lock:
            template = self._template_joints(self.source_w, self.source_h)
            return {key: template[key].copy() for key in CALIBRATION_KEYS}

    def set_source_joints(self, points: Dict[str, np.ndarray]) -> bool:
        """Apply manually selected garment vertices and rebuild the rig."""
        if self.empty or self.rgba is None:
            return False
        if any(key not in points for key in CALIBRATION_KEYS):
            return False

        cleaned: Dict[str, np.ndarray] = {}
        for key in CALIBRATION_KEYS:
            point = np.asarray(points[key], dtype=np.float32)
            if point.shape != (2,) or not np.isfinite(point).all():
                return False
            cleaned[key] = point.copy()

        shoulder_width = float(np.linalg.norm(cleaned["shoulder_r"] - cleaned["shoulder_l"]))
        if shoulder_width < max(4.0, 0.02 * self.source_w):
            return False
        if cleaned["shoulder_r"][0] <= cleaned["shoulder_l"][0]:
            return False

        for key in ("hip_l", "hip_r"):
            if key in self.source_joints:
                cleaned[key] = self.source_joints[key].copy()
        cleaned["shoulder_center"] = self._mid(cleaned["shoulder_l"], cleaned["shoulder_r"])
        cleaned["hip_center"] = self._mid(cleaned["hip_l"], cleaned["hip_r"])
        with self.lock:
            self.source_joints = cleaned
            self.manual_calibration = True
            self._rebuild_skin()
            self.reset_pose()
        return True

    def set_scale(self, scale: float) -> None:
        with self.lock:
            self.scale_factor = max(0.25, min(4.0, float(scale)))

    def set_motion_strength(self, strength: float) -> None:
        with self.lock:
            self.motion_strength = max(0.0, min(1.0, float(strength)))

    @staticmethod
    def _mid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return ((a + b) * 0.5).astype(np.float32)

    def _template_joints(self, width: int, height: int) -> Dict[str, np.ndarray]:
        w = max(2.0, float(width - 1))
        h = max(2.0, float(height - 1))
        joints = {
            "shoulder_l": np.array([0.30 * w, 0.13 * h], np.float32),
            "shoulder_r": np.array([0.70 * w, 0.13 * h], np.float32),
            "elbow_l": np.array([0.13 * w, 0.47 * h], np.float32),
            "elbow_r": np.array([0.87 * w, 0.47 * h], np.float32),
            "wrist_l": np.array([0.03 * w, 0.76 * h], np.float32),
            "wrist_r": np.array([0.97 * w, 0.76 * h], np.float32),
            "hip_l": np.array([0.35 * w, 0.88 * h], np.float32),
            "hip_r": np.array([0.65 * w, 0.88 * h], np.float32),
        }
        joints["shoulder_center"] = self._mid(joints["shoulder_l"], joints["shoulder_r"])
        joints["hip_center"] = self._mid(joints["hip_l"], joints["hip_r"])
        return joints

    def _rebuild_skin(self) -> None:
        j = self.source_joints
        specs = [
            BoneSpec(
                "torso",
                np.stack((j["shoulder_center"], j["shoulder_r"], j["hip_center"])),
                ["shoulder_center", "shoulder_r", "hip_center"],
                [
                    (j["shoulder_l"], j["shoulder_r"]),
                    (j["shoulder_center"], j["hip_center"]),
                    (j["hip_l"], j["hip_r"]),
                ],
                1.45,
                "affine",
            ),
            BoneSpec(
                "upper_arm_l",
                np.stack((j["shoulder_l"], j["elbow_l"])),
                ["shoulder_l", "elbow_l"],
                [(j["shoulder_l"], j["elbow_l"])],
                1.15,
            ),
            BoneSpec(
                "forearm_l",
                np.stack((j["elbow_l"], j["wrist_l"])),
                ["elbow_l", "wrist_l"],
                [(j["elbow_l"], j["wrist_l"])],
                1.05,
            ),
            BoneSpec(
                "upper_arm_r",
                np.stack((j["shoulder_r"], j["elbow_r"])),
                ["shoulder_r", "elbow_r"],
                [(j["shoulder_r"], j["elbow_r"])],
                1.15,
            ),
            BoneSpec(
                "forearm_r",
                np.stack((j["elbow_r"], j["wrist_r"])),
                ["elbow_r", "wrist_r"],
                [(j["elbow_r"], j["wrist_r"])],
                1.05,
            ),
            BoneSpec(
                "pelvis",
                np.stack((j["hip_center"], j["hip_r"], j["shoulder_center"])),
                ["hip_center", "hip_r", "shoulder_center"],
                [
                    (j["hip_l"], j["hip_r"]),
                    (j["hip_center"], j["shoulder_center"]),
                ],
                0.82,
                "affine",
            ),
        ]

        self.bone_specs = specs
        source_grid = self._source_grid_pixels()
        points = source_grid.reshape(-1, 2)
        all_weights = []
        for spec in specs:
            distance = None
            for a, b in spec.weight_segments:
                d = point_segment_distances(points, a.astype(np.float32), b.astype(np.float32))
                distance = d if distance is None else np.minimum(distance, d)
            if distance is None:
                distance = np.zeros(points.shape[0], dtype=np.float32)
            all_weights.append(spec.bias / np.power(distance + 5.5, 2.35))

        weights = np.stack(all_weights, axis=1)
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-8)
        self.weights = weights.reshape(GRID_ROWS, GRID_COLS, -1)
        self.bone_names = [spec.name for spec in specs]

        arm_indices = [
            index for index, spec in enumerate(specs)
            if spec.name.startswith("upper_arm_") or spec.name.startswith("forearm_")
        ]
        arm_weights = np.stack([all_weights[index] for index in arm_indices], axis=1)
        arm_weights /= np.maximum(arm_weights.sum(axis=1, keepdims=True), 1e-8)
        self.arm_weights = arm_weights.reshape(GRID_ROWS, GRID_COLS, -1)

        arm_distance = None
        for index in arm_indices:
            for a, b in specs[index].weight_segments:
                d = point_segment_distances(points, a.astype(np.float32), b.astype(np.float32))
                arm_distance = d if arm_distance is None else np.minimum(arm_distance, d)
        if arm_distance is None:
            arm_distance = np.full(points.shape[0], 1e9, dtype=np.float32)

        influence_radius = max(12.0, 0.30 * min(self.source_w, self.source_h))
        influence = np.clip(1.0 - arm_distance / influence_radius, 0.0, 1.0)
        influence = influence * influence * (3.0 - 2.0 * influence)
        self.arm_influence = influence.reshape(GRID_ROWS, GRID_COLS, 1)

    def _source_grid_pixels(self) -> np.ndarray:
        grid = self.normalized_grid.copy()
        grid[..., 0] *= max(1.0, self.source_w - 1.0)
        grid[..., 1] *= max(1.0, self.source_h - 1.0)
        return grid

    def reset_pose(self) -> None:
        with self.lock:
            self.smooth_joints = {}
            self.smooth_visibility = {}

    def _pose_to_output(self, pose_landmarks, frame_shape: Sequence[int],
                        out_w: int, out_h: int):
        frame_h, frame_w = int(frame_shape[0]), int(frame_shape[1])
        scale = max(out_w / max(1, frame_w), out_h / max(1, frame_h))
        offset_x = (frame_w * scale - out_w) * 0.5
        offset_y = (frame_h * scale - out_h) * 0.5

        joints: Dict[str, np.ndarray] = {}
        visibility: Dict[str, float] = {}
        for name, index in self.JOINT_INDEX.items():
            landmark = pose_landmarks.landmark[index]
            x = float(landmark.x) * frame_w * scale - offset_x
            y = float(landmark.y) * frame_h * scale - offset_y
            joints[name] = np.array([x, y], dtype=np.float32)
            visibility[name] = float(getattr(landmark, "visibility", 1.0))

        shoulder_l = joints["shoulder_l"]
        shoulder_r = joints["shoulder_r"]
        shoulder_width = max(8.0, float(np.linalg.norm(shoulder_r - shoulder_l)))

        # Infer a torso anchor when only the upper body is visible.
        if visibility.get("hip_l", 0.0) < 0.30 or visibility.get("hip_r", 0.0) < 0.30:
            joints["hip_l"] = shoulder_l + np.array([0.0, shoulder_width * 1.20], np.float32)
            joints["hip_r"] = shoulder_r + np.array([0.0, shoulder_width * 1.20], np.float32)
            visibility["hip_l"] = 0.85
            visibility["hip_r"] = 0.85

        # Use stable downward arm positions when elbows or wrists are uncertain.
        for side, sign in (("l", -1.0), ("r", 1.0)):
            shoulder = joints[f"shoulder_{side}"]
            elbow_key = f"elbow_{side}"
            wrist_key = f"wrist_{side}"
            if visibility.get(elbow_key, 0.0) < 0.30:
                joints[elbow_key] = shoulder + np.array(
                    [sign * shoulder_width * 0.52, shoulder_width * 0.74], np.float32
                )
                visibility[elbow_key] = 0.70
            elbow = joints[elbow_key]
            if visibility.get(wrist_key, 0.0) < 0.30:
                fallback = np.array([sign * shoulder_width * 0.14, shoulder_width * 0.72], np.float32)
                joints[wrist_key] = elbow + fallback
                visibility[wrist_key] = 0.65

        joints["shoulder_center"] = self._mid(shoulder_l, shoulder_r)
        joints["hip_center"] = self._mid(joints["hip_l"], joints["hip_r"])
        visibility["shoulder_center"] = min(
            visibility.get("shoulder_l", 0.0), visibility.get("shoulder_r", 0.0)
        )
        visibility["hip_center"] = min(
            visibility.get("hip_l", 0.0), visibility.get("hip_r", 0.0)
        )

        center = self._mid(joints["shoulder_center"], joints["hip_center"])
        if self.scale_factor != 1.0:
            for key in joints:
                joints[key] = center + (joints[key] - center) * self.scale_factor
        return joints, visibility

    def _smooth(self, joints: Dict[str, np.ndarray], visibility: Dict[str, float]):
        smoothed: Dict[str, np.ndarray] = {}
        smoothed_visibility: Dict[str, float] = {}
        for name, point in joints.items():
            vis = visibility.get(name, 1.0)
            previous = self.smooth_joints.get(name)
            previous_vis = self.smooth_visibility.get(name, 0.0)
            if previous is None:
                smoothed[name] = point.copy()
                smoothed_visibility[name] = vis
            elif vis < 0.30:
                smoothed[name] = previous.copy()
                smoothed_visibility[name] = previous_vis * 0.85
            else:
                alpha = 0.34
                smoothed[name] = previous * (1.0 - alpha) + point * alpha
                smoothed_visibility[name] = previous_vis * (1.0 - alpha) + vis * alpha
        self.smooth_joints = {key: value.copy() for key, value in smoothed.items()}
        self.smooth_visibility = dict(smoothed_visibility)
        return smoothed, self.smooth_visibility

    @staticmethod
    def _pose_ready(visibility: Dict[str, float]) -> bool:
        return (
            visibility.get("shoulder_l", 0.0) >= 0.25
            and visibility.get("shoulder_r", 0.0) >= 0.25
        )

    def _build_bones(self, joints: Dict[str, np.ndarray]) -> List[Bone]:
        bones: List[Bone] = []
        for spec in self.bone_specs:
            target = np.stack([joints[key] for key in spec.target_keys]).astype(np.float32)
            if np.isfinite(target).all():
                bones.append(Bone(spec.source.astype(np.float32), target, spec.mode))
        return bones

    def _compute_destination_grid(self, bones: List[Bone]) -> np.ndarray:
        """Keep the full garment similar to the source and clamp arm motion.

        The old renderer blended several strong bone transforms over the entire
        garment. That could pull both sides outward into a fan shape. The new
        method uses one rigid shoulder transform for the whole garment, then
        adds only a small, clamped local offset near the sleeves.
        """
        source_grid = self._source_grid_pixels()
        points = source_grid.reshape(-1, 2)

        try:
            torso_index = self.bone_names.index("torso")
        except ValueError:
            return source_grid.copy()

        torso_bone = bones[torso_index]
        target_center = torso_bone.target[0]
        target_shoulder_r = torso_bone.target[1]
        target_shoulder_l = target_center * 2.0 - target_shoulder_r
        source_shoulder_l = self.source_joints["shoulder_l"]
        source_shoulder_r = self.source_joints["shoulder_r"]

        global_bone = Bone(
            np.stack((source_shoulder_l, source_shoulder_r)),
            np.stack((target_shoulder_l, target_shoulder_r)),
            mode="similarity",
        )
        base = global_bone.map_points(points)

        if self.arm_weights is None or self.arm_influence is None:
            return base.reshape(source_grid.shape)

        arm_indices = [
            index for index, name in enumerate(self.bone_names)
            if name.startswith("upper_arm_") or name.startswith("forearm_")
        ]
        arm_weights = self.arm_weights.reshape(-1, len(arm_indices))
        arm_mapped = np.zeros_like(points, dtype=np.float32)
        for arm_index, bone_index in enumerate(arm_indices):
            arm_mapped += (
                arm_weights[:, arm_index:arm_index + 1]
                * bones[bone_index].map_points(points)
            )

        delta = arm_mapped - base
        max_offset = max(3.0, 0.28 * min(self.source_w, self.source_h))
        delta_length = np.linalg.norm(delta, axis=1)
        clamp_scale = np.minimum(1.0, max_offset / np.maximum(delta_length, 1e-6))
        delta *= clamp_scale[:, None]

        influence = self.arm_influence.reshape(-1, 1)
        mapped = base + delta * influence * self.motion_strength
        mapped = mapped.reshape(source_grid.shape)
        if not np.isfinite(mapped).all():
            return source_grid.copy()
        return mapped

    def _build_inverse_maps(self, source_grid: np.ndarray, destination_grid: np.ndarray,
                            out_w: int, out_h: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        map_x = np.zeros((out_h, out_w), dtype=np.float32)
        map_y = np.zeros((out_h, out_w), dtype=np.float32)
        valid = np.zeros((out_h, out_w), dtype=np.uint8)

        for row in range(GRID_ROWS - 1):
            for col in range(GRID_COLS - 1):
                a = row * GRID_COLS + col
                b = a + 1
                c = a + GRID_COLS
                d = c + 1
                for triangle in ((a, b, d), (a, d, c)):
                    source_tri = np.asarray(
                        [source_grid[i // GRID_COLS, i % GRID_COLS] for i in triangle],
                        dtype=np.float32,
                    )
                    destination_tri = np.asarray(
                        [destination_grid[i // GRID_COLS, i % GRID_COLS] for i in triangle],
                        dtype=np.float32,
                    )

                    x_min = max(0, int(math.floor(float(destination_tri[:, 0].min()))) - 1)
                    x_max = min(out_w - 1, int(math.ceil(float(destination_tri[:, 0].max()))) + 1)
                    y_min = max(0, int(math.floor(float(destination_tri[:, 1].min()))) - 1)
                    y_max = min(out_h - 1, int(math.ceil(float(destination_tri[:, 1].max()))) + 1)
                    if x_max < x_min or y_max < y_min:
                        continue

                    local_w = x_max - x_min + 1
                    local_h = y_max - y_min + 1
                    if local_w <= 0 or local_h <= 0 or local_w * local_h > out_w * out_h:
                        continue

                    shifted = np.round(
                        destination_tri - np.array([x_min, y_min], dtype=np.float32)
                    ).astype(np.int32)
                    local_mask = np.zeros((local_h, local_w), dtype=np.uint8)
                    cv2.fillConvexPoly(local_mask, shifted, 1, lineType=cv2.LINE_8)
                    ys, xs = np.nonzero(local_mask)
                    if xs.size == 0:
                        continue

                    target_points = np.column_stack((xs + x_min, ys + y_min)).astype(np.float32)
                    try:
                        transform = cv2.getAffineTransform(destination_tri, source_tri)
                    except cv2.error:
                        continue
                    source_points = cv2.transform(
                        target_points.reshape(-1, 1, 2), transform
                    ).reshape(-1, 2)

                    map_x[ys + y_min, xs + x_min] = source_points[:, 0]
                    map_y[ys + y_min, xs + x_min] = source_points[:, 1]
                    valid[ys + y_min, xs + x_min] = 255
        return map_x, map_y, valid

    def _warp_garment(self, destination_grid: np.ndarray,
                      out_w: int, out_h: int) -> np.ndarray:
        if self.rgba is None:
            raise ValueError("Garment is empty.")
        source_grid = self._source_grid_pixels()
        map_x, map_y, valid = self._build_inverse_maps(
            source_grid, destination_grid, out_w, out_h
        )

        # Nearest-neighbor sampling preserves the uploaded pixel-art palette.
        warped = cv2.remap(
            self.rgba,
            map_x,
            map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        warped[:, :, 3] = cv2.bitwise_and(warped[:, :, 3], valid)
        return warped

    @staticmethod
    def _composite_rgba(background: np.ndarray, overlay: np.ndarray) -> np.ndarray:
        alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0
        foreground = overlay[:, :, :3].astype(np.float32)
        base = background.astype(np.float32)
        output = foreground * alpha + base * (1.0 - alpha)
        return np.clip(output, 0, 255).astype(np.uint8)

    def _neutral_grid(self, out_w: int, out_h: int) -> np.ndarray:
        j = self.source_joints
        sw = float(np.linalg.norm(j["shoulder_r"] - j["shoulder_l"]))
        top = min(0.0, float(j["shoulder_center"][1]) - sw * 0.9)
        bottom = max(float(self.source_h), float(j["hip_center"][1]) + sw * 1.85)
        pad = 24.0
        scale = min((out_w - 2*pad) / max(self.source_w, 1),
                    (out_h - 2*pad) / max(bottom-top, 1)) * self.scale_factor
        offset = np.array([(out_w-self.source_w*scale)/2,
                           (out_h-(bottom-top)*scale)/2-top*scale], np.float32)
        return self._source_grid_pixels() * scale + offset

    def _mesh_point(self, key: str, grid: np.ndarray) -> np.ndarray:
        """Sample the SAME mesh as the garment so white limbs meet the sleeves."""
        point = self.source_joints[key]
        x = np.clip(point[0] / max(1, self.source_w-1) * (GRID_COLS-1), 0, GRID_COLS-1)
        y = np.clip(point[1] / max(1, self.source_h-1) * (GRID_ROWS-1), 0, GRID_ROWS-1)
        ix, iy = min(int(x), GRID_COLS-2), min(int(y), GRID_ROWS-2)
        fx, fy = x-ix, y-iy
        return ((1-fx)*(1-fy)*grid[iy, ix] + fx*(1-fy)*grid[iy, ix+1]
                + (1-fx)*fy*grid[iy+1, ix] + fx*fy*grid[iy+1, ix+1])

    def _fit_grid_in_view(self, grid, out_w, out_h):
        sl, sr = self._mesh_point("shoulder_l", grid), self._mesh_point("shoulder_r", grid)
        width = max(8.0, float(np.linalg.norm(sr-sl)))
        right = (sr-sl)/width
        down = np.array([-right[1], right[0]], np.float32)
        if down[1] < 0: down = -down
        sc = (sl+sr)*.5
        hip = (self._mesh_point("hip_l",grid)+self._mesh_point("hip_r",grid))*.5
        extra = [sc-down*width*.85-right*width*.3,
                 sc-down*width*.85+right*width*.3,
                 hip+down*width*1.8-right*width*.5,
                 hip+down*width*1.8+right*width*.5]
        pts = np.vstack([grid.reshape(-1,2), extra])
        low, high = pts.min(axis=0), pts.max(axis=0)
        factor = min(1.0, (out_w-32)/max(1,high[0]-low[0]),
                     (out_h-32)/max(1,high[1]-low[1]))
        center = (low+high)*.5
        result = (grid-center)*factor+center
        low = (low-center)*factor+center
        high = (high-center)*factor+center
        shift = np.maximum(16-low,0)-np.maximum(high-np.array([out_w-16,out_h-16]),0)
        return (result+shift).astype(np.float32)

    def _draw_avatar(self, background: np.ndarray, grid: np.ndarray) -> np.ndarray:
        """Draw a featureless, gender-neutral white figure behind the garment."""
        canvas = background.copy()
        j = {key: self._mesh_point(key, grid) for key in
             ("shoulder_l", "shoulder_r", "elbow_l", "elbow_r",
              "wrist_l", "wrist_r", "hip_l", "hip_r")}
        sl, sr = j["shoulder_l"], j["shoulder_r"]
        center = (sl+sr)*0.5
        width = max(8.0, float(np.linalg.norm(sr-sl)))
        right = (sr-sl)/width
        down = np.array([-right[1], right[0]], np.float32)
        if down[1] < 0:
            down = -down
        white, edge = (250, 250, 250), (145, 150, 155)
        def pt(p):
            return tuple(np.clip(np.round(p), -10000, 10000).astype(int))
        def limb(points, thickness):
            line = np.array([pt(p) for p in points], np.int32)
            t = max(3, int(thickness))
            cv2.polylines(canvas, [line], False, edge, t+2, cv2.LINE_AA)
            cv2.polylines(canvas, [line], False, white, t, cv2.LINE_AA)
            for p in points:
                cv2.circle(canvas, pt(p), t//2+1, edge, -1, cv2.LINE_AA)
                cv2.circle(canvas, pt(p), max(1,t//2), white, -1, cv2.LINE_AA)
        # Torso, with no gender-specific anatomy.
        hip = (j["hip_l"]+j["hip_r"])*0.5
        torso = np.array([pt(sl-down*width*.04), pt(sr-down*width*.04),
                          pt(j["hip_r"]+right*width*.08),
                          pt(j["hip_l"]-right*width*.08)], np.int32)
        cv2.fillConvexPoly(canvas, torso, white, cv2.LINE_AA)
        cv2.polylines(canvas, [torso], True, edge, 2, cv2.LINE_AA)
        for side in ("l", "r"):
            limb([j[f"shoulder_{side}"], j[f"elbow_{side}"], j[f"wrist_{side}"]], width*.18)
            hand = j[f"wrist_{side}"]
            radius = max(3, int(width*.105))
            cv2.circle(canvas, pt(hand), radius+1, edge, -1, cv2.LINE_AA)
            cv2.circle(canvas, pt(hand), radius, white, -1, cv2.LINE_AA)
        # Legs are a neutral stance, not tracked gait.
        for sign in (-1, 1):
            upper = hip + right*width*.19*sign
            knee = upper + down*width*.72
            ankle = upper + down*width*1.48
            limb([upper, knee, ankle], width*.24)
            limb([ankle, ankle+right*width*.13*sign], width*.22)
        limb([center, center-down*width*.22], width*.22)
        head_center = center-down*width*.49
        axes = (max(4,int(width*.23)), max(5,int(width*.29)))
        angle = math.degrees(math.atan2(right[1],right[0]))
        cv2.ellipse(canvas, pt(head_center), axes, angle, 0, 360, white, -1, cv2.LINE_AA)
        cv2.ellipse(canvas, pt(head_center), axes, angle, 0, 360, edge, 2, cv2.LINE_AA)
        return canvas

    def _fit_live_avatar(self, joints, out_w, out_h):
        """Frame the estimated whole avatar; legs remain a neutral stance."""
        sc = joints["shoulder_center"]
        hc = joints["hip_center"]
        width = max(8.0, float(np.linalg.norm(joints["shoulder_r"]-joints["shoulder_l"])))
        points = list(joints.values()) + [sc+np.array([0,-width*.85]),
                                         hc+np.array([0,width*1.85])]
        points = np.asarray(points)
        low, high = points.min(axis=0), points.max(axis=0)
        size = np.maximum(high-low, 1)
        factor = min(1.0, (out_w-40)/size[0], (out_h-40)/size[1])
        center = (low+high)*.5
        desired = np.array([out_w*.5, out_h*.5])
        # Keep motion visible rather than fully centering each frame.
        mapped = {k: (p-center)*factor+desired+(center-desired)*.15
                  for k,p in joints.items()}
        return mapped

    def render(self, pose_landmarks, frame_shape: Sequence[int],
               out_w: int = DEFAULT_WINDOW_W,
               out_h: int = DEFAULT_WINDOW_H) -> Tuple[np.ndarray, bool]:
        background = self._background_for_size(out_w, out_h)
        with self.lock:
            if self.empty or self.rgba is None:
                return background, False
            grid = self._neutral_grid(out_w, out_h)
            pose_ok = False
            if pose_landmarks is not None:
                joints, visibility = self._pose_to_output(pose_landmarks, frame_shape, out_w, out_h)
                joints = self._fit_live_avatar(joints, out_w, out_h)
                joints, visibility = self._smooth(joints, visibility)
                pose_ok = self._pose_ready(visibility)
                if pose_ok:
                    bones = self._build_bones(joints)
                    if len(bones) == len(self.bone_specs):
                        grid = self._compute_destination_grid(bones)
            grid = self._fit_grid_in_view(grid, out_w, out_h)
            garment = self._warp_garment(grid, out_w, out_h)
            background = self._draw_avatar(background, grid)
            return self._composite_rgba(background, garment), pose_ok

    def make_preview(self, out_w: int = DEFAULT_WINDOW_W,
                     out_h: int = DEFAULT_WINDOW_H) -> np.ndarray:
        return self.render(None, (out_h, out_w), out_w, out_h)[0]


class UpperBodyCameraWorker(threading.Thread):
    def __init__(self, renderer: GarmentRenderer,
                 output_queue: "queue.Queue",
                 stop_event: threading.Event,
                 camera_index: int,
                 get_render_size) -> None:
        super().__init__(daemon=True)
        self.renderer = renderer
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.camera_index = camera_index
        self.get_render_size = get_render_size
        self.capture: Optional[cv2.VideoCapture] = None

    def stop(self) -> None:
        self.stop_event.set()
        if self.capture is not None:
            try:
                self.capture.release()
            except Exception:
                pass

    def run(self) -> None:
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        self.capture = cv2.VideoCapture(self.camera_index, backend)
        if not self.capture.isOpened():
            self.capture.release()
            self.capture = cv2.VideoCapture(self.camera_index)

        if not self.capture.isOpened():
            self.output_queue.put((
                "error",
                "Unable to open the camera. Close other camera apps or change the camera index."
            ))
            return

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        try:
            import mediapipe as mp
            if not hasattr(mp, "solutions"):
                raise RuntimeError("Install mediapipe==0.10.21 in Python 3.10/3.11.")
        except Exception as exc:
            self.capture.release()
            self.output_queue.put(("error", f"MediaPipe unavailable: {exc}"))
            return

        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose

        try:
            with mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=0.4,
                min_tracking_confidence=0.4,
            ) as pose:
                while not self.stop_event.is_set():
                    ok, frame = self.capture.read()
                    if not ok:
                        self.output_queue.put(("error", "Unable to read a camera frame."))
                        break

                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = pose.process(rgb)

                    camera_preview = frame.copy()
                    if result.pose_landmarks:
                        mp_drawing.draw_landmarks(
                            camera_preview,
                            result.pose_landmarks,
                            UPPER_BODY_CONNECTIONS,
                        )

                    width, height = self.get_render_size()
                    output, pose_ok = self.renderer.render(
                        result.pose_landmarks, frame.shape, width, height
                    )

                    try:
                        while True:
                            self.output_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self.output_queue.put(("frame", output, camera_preview, pose_ok))
        except Exception as exc:
            self.output_queue.put(("error", f"Pose tracking error: {exc}"))
        finally:
            if self.capture is not None:
                self.capture.release()



class GarmentCalibrationWindow(tk.Toplevel):
    """Drag garment rig points onto the matching pixels of the source image."""

    MAX_CANVAS_W = 700
    MAX_CANVAS_H = 760
    CANVAS_MARGIN = 60
    HANDLE_RADIUS = 10
    SHORT_LABELS = {
        "shoulder_l": "L shoulder",
        "shoulder_r": "R shoulder",
        "elbow_l": "L elbow",
        "elbow_r": "R elbow",
        "wrist_l": "L wrist",
        "wrist_r": "R wrist",
        "hip_l": "L hip",
        "hip_r": "R hip",
    }
    RIG_EDGES = [
        ("shoulder_l", "shoulder_r"),
        ("shoulder_l", "elbow_l"),
        ("elbow_l", "wrist_l"),
        ("shoulder_r", "elbow_r"),
        ("elbow_r", "wrist_r"),
        ("shoulder_l", "hip_l"),
        ("shoulder_r", "hip_r"),
        ("hip_l", "hip_r"),
    ]

    def __init__(self, app) -> None:
        super().__init__(app.root)
        self.app = app
        self.title("Manual Garment Fit - Drag Vertices")
        self.configure(bg="#20252b")
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self.cancel)

        rgba = self.app.renderer.get_source_rgba()
        source_w, source_h = self.app.renderer.get_source_size()
        if rgba is None or source_w <= 0 or source_h <= 0:
            self.destroy()
            return

        self.source_w = source_w
        self.source_h = source_h
        self.points: Dict[str, np.ndarray] = self.app.renderer.get_source_joints()
        if any(key not in self.points for key in CALIBRATION_KEYS):
            self.points = self.app.renderer.get_default_joints()

        self.display_scale = min(
            self.MAX_CANVAS_W / source_w,
            self.MAX_CANVAS_H / source_h,
        )
        self.display_w = max(1, int(round(source_w * self.display_scale)))
        self.display_h = max(1, int(round(source_h * self.display_scale)))
        margin = self.CANVAS_MARGIN
        self.canvas_w = max(420, self.display_w + margin * 2)
        self.canvas_h = max(600, self.display_h + margin * 2)
        self.offset_x = (self.canvas_w - self.display_w) // 2
        self.offset_y = (self.canvas_h - self.display_h) // 2

        display_rgb = self._make_display_image(rgba)
        resized = cv2.resize(
            display_rgb,
            (self.display_w, self.display_h),
            interpolation=cv2.INTER_NEAREST,
        )
        self.display_photo = ImageTk.PhotoImage(Image.fromarray(resized))

        self.canvas = tk.Canvas(
            self,
            width=self.canvas_w,
            height=self.canvas_h,
            bg="#20252b",
            highlightthickness=0,
        )
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
        self.canvas.create_image(
            self.offset_x,
            self.offset_y,
            anchor=tk.NW,
            image=self.display_photo,
        )
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        panel = tk.Frame(self, bg="#f6fbe9", width=260)
        panel.pack(side=tk.RIGHT, fill=tk.Y)
        panel.pack_propagate(False)

        tk.Label(
            panel,
            text="Manual Garment Fit",
            bg="#f6fbe9",
            fg="#162010",
            font=("Segoe UI", 14, "bold"),
        ).pack(anchor="w", padx=12, pady=(14, 4))
        tk.Label(
            panel,
            text=(
                "Drag each colored handle to the matching place on the "
                "garment image. The colored lines show the rig that will "
                "follow the detected upper-body skeleton."
            ),
            bg="#f6fbe9",
            fg="#203020",
            justify=tk.LEFT,
            wraplength=230,
        ).pack(anchor="w", padx=12, pady=(0, 10))

        for key, label, color in CALIBRATION_POINTS:
            row = tk.Frame(panel, bg="#f6fbe9")
            row.pack(anchor="w", padx=12, pady=2)
            tk.Label(row, text="  ", bg=color, width=2).pack(side=tk.LEFT)
            tk.Label(row, text=label, bg="#f6fbe9", fg="#203020").pack(
                side=tk.LEFT, padx=(7, 0)
            )

        self.hint_var = tk.StringVar(
            value="Tip: drag the handles, then click Apply Fit."
        )
        tk.Label(
            panel,
            textvariable=self.hint_var,
            bg="#f6fbe9",
            fg="#3b4b35",
            justify=tk.LEFT,
            wraplength=230,
        ).pack(anchor="w", padx=12, pady=(12, 8))

        ttk.Button(
            panel,
            text="Reset Default",
            command=self.reset_default,
        ).pack(fill=tk.X, padx=12, pady=3)
        ttk.Button(
            panel,
            text="Apply Fit",
            command=self.apply_fit,
        ).pack(fill=tk.X, padx=12, pady=3)
        ttk.Button(
            panel,
            text="Cancel",
            command=self.cancel,
        ).pack(fill=tk.X, padx=12, pady=3)

        self.active_key: Optional[str] = None
        self.redraw_rig()
        self.update_idletasks()
        self.geometry(f"{self.canvas_w + 260}x{max(self.canvas_h, 560)}")

    @staticmethod
    def _make_display_image(rgba: np.ndarray) -> np.ndarray:
        background = np.full(rgba.shape[:2] + (3,), (55, 55, 60), dtype=np.uint8)
        alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
        foreground = rgba[:, :, :3].astype(np.float32)
        composited = foreground * alpha + background.astype(np.float32) * (1.0 - alpha)
        bgr = np.clip(composited, 0, 255).astype(np.uint8)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def source_to_canvas(self, point: np.ndarray) -> Tuple[float, float]:
        x = self.offset_x + float(point[0]) * self.display_scale
        y = self.offset_y + float(point[1]) * self.display_scale
        padding = self.HANDLE_RADIUS + 6
        x = float(np.clip(x, padding, self.canvas_w - padding))
        y = float(np.clip(y, padding, self.canvas_h - padding))
        return x, y

    def canvas_to_source(self, x: float, y: float) -> np.ndarray:
        source_x = (x - self.offset_x) / self.display_scale
        source_y = (y - self.offset_y) / self.display_scale
        return np.array(
            [
                np.clip(source_x, 0.0, self.source_w - 1.0),
                np.clip(source_y, 0.0, self.source_h - 1.0),
            ],
            dtype=np.float32,
        )

    def redraw_rig(self) -> None:
        self.canvas.delete("rig")
        for first, second in self.RIG_EDGES:
            if first not in self.points or second not in self.points:
                continue
            x1, y1 = self.source_to_canvas(self.points[first])
            x2, y2 = self.source_to_canvas(self.points[second])
            self.canvas.create_line(
                x1,
                y1,
                x2,
                y2,
                fill="#f7f7f7",
                width=3,
                tags="rig",
            )

        for key, _label, color in CALIBRATION_POINTS:
            x, y = self.source_to_canvas(self.points[key])
            radius = self.HANDLE_RADIUS
            self.canvas.create_oval(
                x - radius,
                y - radius,
                x + radius,
                y + radius,
                fill=color,
                outline="#ffffff",
                width=2,
                tags="rig",
            )
            self.canvas.create_text(
                x + 12,
                y - 12,
                text=self.SHORT_LABELS[key],
                fill="#ffffff",
                anchor=tk.SW,
                font=("Segoe UI", 9, "bold"),
                tags="rig",
            )

    def on_press(self, event) -> None:
        closest_key = None
        closest_distance = self.HANDLE_RADIUS + 10.0
        for key in CALIBRATION_KEYS:
            x, y = self.source_to_canvas(self.points[key])
            distance = math.hypot(event.x - x, event.y - y)
            if distance < closest_distance:
                closest_distance = distance
                closest_key = key
        self.active_key = closest_key
        if closest_key is not None:
            self.hint_var.set(f"Moving {self.SHORT_LABELS[closest_key]}...")

    def on_drag(self, event) -> None:
        if self.active_key is None:
            return
        self.points[self.active_key] = self.canvas_to_source(event.x, event.y)
        self.redraw_rig()

    def on_release(self, _event) -> None:
        if self.active_key is not None:
            self.hint_var.set(
                f"Updated {self.SHORT_LABELS[self.active_key]}. "
                "Click Apply Fit when finished."
            )
        self.active_key = None

    def reset_default(self) -> None:
        self.points = self.app.renderer.get_default_joints()
        self.redraw_rig()
        self.hint_var.set("Default positions restored.")

    def apply_fit(self) -> bool:
        if self.app.apply_manual_calibration(self.points):
            self.app.calibration_window = None
            self.destroy()
            return True
        return False

    def cancel(self) -> None:
        self.app.calibration_window = None
        self.destroy()

class GarmentMotionApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("White Avatar Garment Motion Mirror v6.0")
        self.win_w = DEFAULT_WINDOW_W
        self.win_h = DEFAULT_WINDOW_H
        self.root.geometry(f"{self.win_w}x{self.win_h}")
        self.root.resizable(True, True)
        self.root.configure(bg="#00ff00")

        self.renderer = GarmentRenderer()
        self.frame_queue: "queue.Queue" = queue.Queue(maxsize=2)
        self.stop_event = threading.Event()
        self.camera_worker: Optional[UpperBodyCameraWorker] = None
        self.calibration_window: Optional[GarmentCalibrationWindow] = None
        self.current_frame = self.renderer.make_preview(self.win_w, self.win_h)
        self.photo = None
        self.camera_photo = None
        self.status_var = tk.StringVar(
            value="Upload a garment image that uses a connected green-screen background."
        )

        self.canvas = tk.Canvas(
            self.root,
            width=self.win_w,
            height=self.win_h,
            highlightthickness=0,
            bg="#00ff00",
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.image_item = self.canvas.create_image(0, 0, anchor=tk.NW)
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        self._show_frame(self.current_frame)

        self._build_control_window()
        self._build_camera_window()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(25, self._poll_frames)

    def get_render_size(self) -> Tuple[int, int]:
        return max(100, self.win_w), max(100, self.win_h)

    def on_canvas_resize(self, event) -> None:
        if event.width > 10 and event.height > 10:
            if abs(self.win_w - event.width) > 5 or abs(self.win_h - event.height) > 5:
                self.win_w = event.width
                self.win_h = event.height
                if self.camera_worker is None or not self.camera_worker.is_alive():
                    preview = self.renderer.make_preview(self.win_w, self.win_h)
                    self._show_frame(preview)

    def _build_control_window(self) -> None:
        self.control_window = tk.Toplevel(self.root)
        self.control_window.title("Controls")
        self.control_window.geometry("470x335")
        self.control_window.resizable(False, False)
        self.control_window.configure(bg="#f6fbe9")
        self.control_window.protocol("WM_DELETE_WINDOW", self.on_close)

        panel = tk.Frame(self.control_window, bg="#f6fbe9")
        panel.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        row1 = tk.Frame(panel, bg="#f6fbe9")
        row1.pack(padx=5, pady=6, fill=tk.X)
        ttk.Button(
            row1,
            text="Upload Garment Image",
            command=self.upload_image,
        ).pack(side=tk.LEFT, padx=(0, 7))
        self.camera_button = ttk.Button(
            row1,
            text="Start Camera",
            command=self.toggle_camera,
        )
        self.camera_button.pack(side=tk.LEFT, padx=(0, 7))
        ttk.Button(
            row1,
            text="Save Current Frame",
            command=self.save_frame,
        ).pack(side=tk.LEFT)

        row2 = tk.Frame(panel, bg="#f6fbe9")
        row2.pack(padx=5, pady=6, fill=tk.X)
        ttk.Label(row2, text="Camera index:").pack(side=tk.LEFT)
        self.camera_index_var = tk.IntVar(value=0)
        ttk.Spinbox(
            row2,
            from_=0,
            to=8,
            width=4,
            textvariable=self.camera_index_var,
        ).pack(side=tk.LEFT, padx=(6, 12))
        ttk.Button(
            row2,
            text="Use Original Size",
            command=self.use_original_size,
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(
            row2,
            text="Manual Fit",
            command=self.open_calibration,
        ).pack(side=tk.LEFT)

        row3 = tk.Frame(panel, bg="#f6fbe9")
        row3.pack(padx=5, pady=6, fill=tk.X)
        ttk.Label(row3, text="Garment scale:").pack(side=tk.LEFT)
        self.scale_var = tk.DoubleVar(value=1.0)
        self.scale_slider = ttk.Scale(
            row3,
            from_=0.5,
            to=2.5,
            variable=self.scale_var,
            command=self.on_scale_change,
        )
        self.scale_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 5))
        self.scale_value_label = ttk.Label(row3, text="1.0x", width=5)
        self.scale_value_label.pack(side=tk.LEFT)

        row4 = tk.Frame(panel, bg="#f6fbe9")
        row4.pack(padx=5, pady=6, fill=tk.X)
        ttk.Label(row4, text="Motion strength:").pack(side=tk.LEFT)
        self.motion_var = tk.DoubleVar(value=0.85)
        self.motion_slider = ttk.Scale(
            row4,
            from_=0.0,
            to=1.0,
            variable=self.motion_var,
            command=self.on_motion_change,
        )
        self.motion_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 5))
        self.motion_value_label = ttk.Label(row4, text="85%", width=5)
        self.motion_value_label.pack(side=tk.LEFT)

        row5 = tk.Frame(panel, bg="#f6fbe9")
        row5.pack(padx=5, pady=6, fill=tk.X)
        ttk.Label(row5, text="Background:").pack(side=tk.LEFT)
        ttk.Button(
            row5,
            text="Upload Background",
            command=self.upload_background,
        ).pack(side=tk.LEFT, padx=(6, 8))
        ttk.Button(
            row5,
            text="Green Background",
            command=self.use_green_background,
        ).pack(side=tk.LEFT)

        self.status_label = tk.Label(
            panel,
            textvariable=self.status_var,
            bg="#f6fbe9",
            fg="#203020",
            justify=tk.LEFT,
            anchor="w",
            wraplength=430,
        )
        self.status_label.pack(padx=5, pady=(10, 5), fill=tk.X)

    def _build_camera_window(self) -> None:
        self.camera_window = tk.Toplevel(self.root)
        self.camera_window.title("Live Camera and Upper-Body Pose")
        self.camera_window.geometry("360x270")
        self.camera_window.resizable(False, False)
        self.camera_window.protocol("WM_DELETE_WINDOW", self.on_close)

        self.camera_canvas = tk.Canvas(
            self.camera_window,
            width=360,
            height=270,
            bg="#222222",
        )
        self.camera_canvas.pack(fill=tk.BOTH, expand=True)
        self.camera_image_item = self.camera_canvas.create_image(0, 0, anchor=tk.NW)

    def open_calibration(self) -> None:
        if self.renderer.empty:
            messagebox.showinfo(
                "No Garment Loaded",
                "Upload a garment image before opening Manual Fit.",
            )
            return
        if (
            self.calibration_window is not None
            and self.calibration_window.winfo_exists()
        ):
            self.calibration_window.lift()
            self.calibration_window.focus_force()
            return
        self.calibration_window = GarmentCalibrationWindow(self)

    def apply_manual_calibration(self, points: Dict[str, np.ndarray]) -> bool:
        if not self.renderer.set_source_joints(points):
            messagebox.showerror(
                "Invalid Fit",
                "The left and right shoulders must be separated and in the "
                "correct left-to-right order.",
            )
            return False
        self.status_var.set(
            "Manual garment fit applied. The uploaded pixels now follow the "
            "user-defined upper-body vertices."
        )
        if self.camera_worker is None or not self.camera_worker.is_alive():
            self._show_frame(self.renderer.make_preview(self.win_w, self.win_h))
        return True

    def use_original_size(self) -> None:
        if self.renderer.empty:
            messagebox.showinfo(
                "No Garment Loaded",
                "Upload a garment image first.",
            )
            return
        with self.renderer.lock:
            source_w = self.renderer.source_w
            source_h = self.renderer.source_h
        self.win_w, self.win_h = source_w, source_h
        self.root.geometry(f"{source_w}x{source_h}")
        self._show_frame(self.renderer.make_preview(source_w, source_h))
        self.status_var.set(
            f"Main window set to the original garment size: {source_w} x {source_h}."
        )

    def on_scale_change(self, value: str) -> None:
        scale_value = float(value)
        self.scale_value_label.configure(text=f"{scale_value:.1f}x")
        self.renderer.set_scale(scale_value)
        if self.camera_worker is None or not self.camera_worker.is_alive():
            self._show_frame(self.renderer.make_preview(self.win_w, self.win_h))

    def on_motion_change(self, value: str) -> None:
        strength = float(value)
        self.motion_value_label.configure(text=f"{strength * 100:.0f}%")
        self.renderer.set_motion_strength(strength)

    def _show_frame(self, frame_bgr: np.ndarray) -> None:
        self.current_frame = frame_bgr.copy()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.itemconfigure(self.image_item, image=self.photo)

    def _show_camera_frame(self, camera_bgr: np.ndarray) -> None:
        resized = cv2.resize(camera_bgr, (360, 270), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self.camera_photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.camera_canvas.itemconfigure(self.camera_image_item, image=self.camera_photo)

    def upload_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Select a garment image with a green-screen background",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        if (
            self.calibration_window is not None
            and self.calibration_window.winfo_exists()
        ):
            self.calibration_window.destroy()
            self.calibration_window = None

        image = load_image_unicode(path)
        if image is None:
            messagebox.showerror(
                "Read Error",
                "The selected image could not be read. Please use PNG, JPG, or BMP.",
            )
            return

        success = self.renderer.set_garment(image)
        if not success:
            messagebox.showerror(
                "No Garment Detected",
                "No connected green background was detected, or the garment area is too small.\n"
                "Use a background close to R=0, G=255, B=0.",
            )
            return

        self.status_var.set(
            "Garment loaded. Use Manual Fit to place the garment vertices on "
            "the upload, then start the camera."
        )
        if self.camera_worker is None or not self.camera_worker.is_alive():
            self._show_frame(self.renderer.make_preview(self.win_w, self.win_h))
        self.root.after(150, self.open_calibration)

    def upload_background(self) -> None:
        path = filedialog.askopenfilename(
            title="Select a background image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        image = load_image_unicode(path)
        if image is None or not self.renderer.set_background(image):
            messagebox.showerror(
                "Background Error",
                "The selected background image could not be loaded.",
            )
            return

        self.status_var.set(
            "Background image loaded. It will fill the output window and "
            "appear behind the garment."
        )
        if self.camera_worker is None or not self.camera_worker.is_alive():
            self._show_frame(self.renderer.make_preview(self.win_w, self.win_h))

    def use_green_background(self) -> None:
        self.renderer.use_green_background()
        self.status_var.set("The output background has been reset to RGB(0, 255, 0).")
        if self.camera_worker is None or not self.camera_worker.is_alive():
            self._show_frame(self.renderer.make_preview(self.win_w, self.win_h))

    def toggle_camera(self) -> None:
        if self.camera_worker is not None and self.camera_worker.is_alive():
            self.stop_camera()
            self.status_var.set("Camera stopped.")
            self._show_frame(self.renderer.make_preview(self.win_w, self.win_h))
            return

        if self.renderer.empty:
            messagebox.showinfo(
                "No Garment Loaded",
                "Upload a garment image with a green-screen background first.",
            )
            return

        self.stop_event = threading.Event()
        self.renderer.reset_pose()
        self.camera_worker = UpperBodyCameraWorker(
            self.renderer,
            self.frame_queue,
            self.stop_event,
            int(self.camera_index_var.get()),
            self.get_render_size,
        )
        self.camera_worker.start()
        self.camera_button.configure(text="Stop Camera")
        self.status_var.set("Opening the camera...")

    def stop_camera(self) -> None:
        self.stop_event.set()
        if self.camera_worker is not None:
            self.camera_worker.stop()
            self.camera_worker.join(timeout=1.5)
        self.camera_worker = None
        self.camera_button.configure(text="Start Camera")

    def _poll_frames(self) -> None:
        latest = None
        try:
            while True:
                latest = self.frame_queue.get_nowait()
        except queue.Empty:
            pass

        if latest is not None:
            if latest[0] == "frame":
                _, frame, camera_frame, pose_ok = latest
                self._show_frame(frame)
                self._show_camera_frame(camera_frame)
                if pose_ok:
                    self.status_var.set(
                        "Upper-body pose detected. The white avatar and garment are following your upper body."
                    )
                else:
                    self.status_var.set(
                        "Both shoulders are not clearly visible. Adjust the camera or step back slightly."
                    )
            elif latest[0] == "error":
                self.status_var.set(latest[1])
                self.stop_camera()

        self.root.after(25, self._poll_frames)

    def save_frame(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save Current Frame",
            defaultextension=".png",
            initialfile="upper_body_garment.png",
            filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg")],
        )
        if not path:
            return
        if save_image_unicode(path, self.current_frame):
            self.status_var.set(f"Frame saved: {path}")
        else:
            messagebox.showerror(
                "Save Error",
                "The image could not be saved. Check the destination folder permissions.",
            )

    def on_close(self) -> None:
        self.stop_camera()
        if (
            self.calibration_window is not None
            and self.calibration_window.winfo_exists()
        ):
            self.calibration_window.destroy()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    GarmentMotionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
