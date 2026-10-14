"""Tests for vision.face — skipped when mediapipe/cv2 are not installed."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("mediapipe")
pytest.importorskip("cv2")

from interviewlens.vision.face import (  # noqa: E402
    _LEFT_EYE,
    _RIGHT_EYE,
    EYE_CROP_SIZE,
    FACE_CROP_SIZE,
    FacePipeline,
    FaceResult,
    _aligned_crop,
    _eye_crop,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _blank(h: int = 480, w: int = 640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _fake_pts(h: int = 480, w: int = 640) -> np.ndarray:
    """478 landmark points spread across the frame center."""
    rng = np.random.default_rng(0)
    pts = rng.uniform(
        low=[w * 0.2, h * 0.2],
        high=[w * 0.8, h * 0.8],
        size=(478, 2),
    ).astype(np.float32)
    # force eye centers to be horizontally aligned for stable alignment test
    pts[_LEFT_EYE[0]] = [w * 0.4, h * 0.45]
    pts[_RIGHT_EYE[0]] = [w * 0.6, h * 0.45]
    return pts


def _make_mock_pipeline() -> FacePipeline:
    """Return a FacePipeline backed by a no-op mock detector."""
    mock_detector = MagicMock()
    mock_detector.detect.return_value.face_landmarks = []

    with patch(
        "interviewlens.vision.face.mp_vision.FaceLandmarker.create_from_options",
        return_value=mock_detector,
    ):
        pipe = FacePipeline(model_path="fake.task")
    return pipe


# ---------------------------------------------------------------------------
# Utility function tests (no model needed)
# ---------------------------------------------------------------------------


def test_aligned_crop_shape():
    img = _blank(480, 640)
    pts = _fake_pts(480, 640)
    crop = _aligned_crop(img, pts, FACE_CROP_SIZE)
    assert crop.shape == (FACE_CROP_SIZE, FACE_CROP_SIZE, 3)
    assert crop.dtype == np.uint8


def test_eye_crop_shape():
    img = _blank(480, 640)
    pts = _fake_pts(480, 640)
    for indices in (_LEFT_EYE, _RIGHT_EYE):
        patch_img = _eye_crop(img, pts, indices, EYE_CROP_SIZE)
        assert patch_img.shape == (EYE_CROP_SIZE, EYE_CROP_SIZE, 3)
        assert patch_img.dtype == np.uint8


def test_face_result_fields():
    dummy = np.zeros((224, 224, 3), dtype=np.uint8)
    lm = np.zeros((478, 2), dtype=np.float32)
    r = FaceResult(crop=dummy, left_eye=dummy, right_eye=dummy, landmarks=lm)
    assert r.crop.shape == (224, 224, 3)
    assert r.landmarks.shape == (478, 2)


# ---------------------------------------------------------------------------
# Pipeline tests (mock detector, no download)
# ---------------------------------------------------------------------------


def test_no_face_returns_none():
    pipe = _make_mock_pipeline()
    with pipe:
        result = pipe(_blank())
    assert result is None


def test_pipeline_context_manager():
    pipe = _make_mock_pipeline()
    with pipe:
        assert pipe is not None


def test_tiny_frame_no_crash():
    """Very small frames should not raise."""
    pipe = _make_mock_pipeline()
    with pipe:
        result = pipe(np.zeros((32, 32, 3), dtype=np.uint8))
    assert result is None
