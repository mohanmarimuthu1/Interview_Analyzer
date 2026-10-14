"""MediaPipe face pipeline: landmarks -> aligned crop, eye crops."""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
_CACHE_DIR = Path.home() / ".cache" / "interviewlens"
_MODEL_CACHE = _CACHE_DIR / "face_landmarker.task"

# fmt: off
# FaceLandmarker (478-point) eye contour indices — same subset as FaceMesh
_LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
_RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
# fmt: on

FACE_CROP_SIZE = 224
EYE_CROP_SIZE = 64


@dataclass
class FaceResult:
    crop: np.ndarray  # (224, 224, 3) uint8 BGR, roll-aligned
    left_eye: np.ndarray  # (64, 64, 3) uint8 BGR
    right_eye: np.ndarray  # (64, 64, 3) uint8 BGR
    landmarks: np.ndarray  # (N, 2) float32 pixel coordinates


def _ensure_model(model_path: Path = _MODEL_CACHE) -> str:
    """Download face_landmarker.task to cache if not present."""
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading face landmarker model -> {model_path}")
        urllib.request.urlretrieve(_MODEL_URL, model_path)
    return str(model_path)


class FacePipeline:
    """Single-face landmark detection and crop extraction.

    Downloads the MediaPipe face landmarker model on first use (~25 MB, cached
    in ~/.cache/interviewlens/).  Use as a context manager or call .close().
    """

    def __init__(
        self,
        model_path: str | None = None,
        static_mode: bool = False,
        max_faces: int = 1,
    ) -> None:
        resolved = model_path or _ensure_model()
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=resolved),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=max_faces,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._detector = mp_vision.FaceLandmarker.create_from_options(opts)

    def __call__(self, frame_bgr: np.ndarray) -> FaceResult | None:
        """Process one BGR frame. Returns None if no face detected."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_img)

        if not result.face_landmarks:
            return None

        lm = result.face_landmarks[0]
        pts = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)

        return FaceResult(
            crop=_aligned_crop(frame_bgr, pts, FACE_CROP_SIZE),
            left_eye=_eye_crop(frame_bgr, pts, _LEFT_EYE, EYE_CROP_SIZE),
            right_eye=_eye_crop(frame_bgr, pts, _RIGHT_EYE, EYE_CROP_SIZE),
            landmarks=pts,
        )

    def close(self) -> None:
        self._detector.close()

    def __enter__(self) -> FacePipeline:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _aligned_crop(img: np.ndarray, pts: np.ndarray, size: int) -> np.ndarray:
    left_center = pts[_LEFT_EYE].mean(axis=0)
    right_center = pts[_RIGHT_EYE].mean(axis=0)

    dy = right_center[1] - left_center[1]
    dx = right_center[0] - left_center[0]
    angle = float(np.degrees(np.arctan2(dy, dx)))

    eye_mid = ((left_center + right_center) / 2).astype(np.float32)
    M = cv2.getRotationMatrix2D(tuple(eye_mid.tolist()), angle, 1.0)

    aligned = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    # rotate landmarks to find tight bounding box
    ones = np.ones((len(pts), 1), dtype=np.float32)
    rot_pts = (M @ np.hstack([pts, ones]).T).T
    x1, y1 = rot_pts.min(axis=0).astype(int)
    x2, y2 = rot_pts.max(axis=0).astype(int)

    pad = max(int((x2 - x1) * 0.12), 4)
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(img.shape[1], x2 + pad)
    y2 = min(img.shape[0], y2 + pad)

    face = aligned[y1:y2, x1:x2]
    if face.size == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    return cv2.resize(face, (size, size))


def _eye_crop(img: np.ndarray, pts: np.ndarray, indices: list[int], size: int) -> np.ndarray:
    eye_pts = pts[indices]
    cx, cy = eye_pts.mean(axis=0)
    half = (
        max(
            float(eye_pts[:, 0].max() - eye_pts[:, 0].min()),
            float(eye_pts[:, 1].max() - eye_pts[:, 1].min()),
        )
        * 0.9
    )
    half = max(half, 8.0)

    x1 = max(0, int(cx - half))
    y1 = max(0, int(cy - half))
    x2 = min(img.shape[1], int(cx + half))
    y2 = min(img.shape[0], int(cy + half))

    patch = img[y1:y2, x1:x2]
    if patch.size == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    return cv2.resize(patch, (size, size))
