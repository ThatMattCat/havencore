"""Multi-factor face quality scoring.

Returns a score in [0, 1] combining four signals about a detected face:

  bbox_area     — bigger faces have more pixels to embed reliably; saturates
                  at 25% of the frame area (a face bigger than that is just
                  "max-area").
  sharpness     — Laplacian variance of the face crop; low variance means
                  blurry. Saturates at variance=300 (typical "very sharp"
                  point for face crops at 640px detection size).
  pose          — eye symmetry around the nose, derived from the 5-point
                  landmarks insightface returns. Frontal ≈ 1.0, strong yaw
                  toward 0.
  brightness    — mean grayscale of the face crop, peaking at mid-tone (127).
                  Penalizes both over- and under-exposed crops.

Weights are tuning knobs, not deployment knobs — left as constants here.
With QUALITY_FLOOR=0.40 and IMPROVEMENT_QUALITY_FLOOR=0.70 (the env
defaults), a "usable" face roughly clears the floor when at least three of
the four signals are decent, and a "learn from it" face needs all four to be
strong.
"""

import logging

import cv2
import numpy as np


logger = logging.getLogger("face-recognition.quality")


W_AREA = 0.30
W_SHARPNESS = 0.30
W_POSE = 0.25
W_BRIGHTNESS = 0.15

# Saturation point for bbox area: face occupying >=25% of frame is "max".
AREA_SATURATION = 0.25
# Saturation point for Laplacian variance — a sharp face crop at the
# detection resolution typically lands around this value.
SHARPNESS_SATURATION = 300.0


def _crop(frame: np.ndarray, bbox: np.ndarray) -> np.ndarray | None:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in bbox[:4])
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def _area_score(frame: np.ndarray, bbox: np.ndarray) -> float:
    h, w = frame.shape[:2]
    frame_area = float(h * w)
    if frame_area <= 0:
        return 0.0
    bw = max(0.0, float(bbox[2] - bbox[0]))
    bh = max(0.0, float(bbox[3] - bbox[1]))
    ratio = (bw * bh) / frame_area
    return float(min(ratio / AREA_SATURATION, 1.0))


def _sharpness_score(crop_gray: np.ndarray) -> float:
    var = float(cv2.Laplacian(crop_gray, cv2.CV_64F).var())
    return float(min(var / SHARPNESS_SATURATION, 1.0))


def _pose_score(kps: np.ndarray) -> float:
    # insightface 5-point order: left_eye, right_eye, nose, left_mouth, right_mouth.
    if kps is None or len(kps) < 3:
        return 0.0
    left_eye, right_eye, nose = kps[0], kps[1], kps[2]
    d_left = float(np.linalg.norm(left_eye - nose))
    d_right = float(np.linalg.norm(right_eye - nose))
    denom = d_left + d_right
    if denom <= 1e-6:
        return 0.0
    asymmetry = abs(d_left - d_right) / denom
    return float(max(0.0, 1.0 - asymmetry))


def _brightness_score(crop_gray: np.ndarray) -> float:
    mean = float(crop_gray.mean())
    return float(max(0.0, 1.0 - abs(mean - 127.0) / 127.0))


def score_face(frame: np.ndarray, face) -> float:
    """Compute a quality score in [0, 1] for one detected face.

    `face` is an insightface Face object — uses `.bbox` and `.kps`.
    Returns 0.0 if the bbox can't be cropped (degenerate detection).
    """
    crop = _crop(frame, face.bbox)
    if crop is None:
        return 0.0
    crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    area = _area_score(frame, face.bbox)
    sharpness = _sharpness_score(crop_gray)
    pose = _pose_score(face.kps)
    brightness = _brightness_score(crop_gray)

    score = (
        W_AREA * area
        + W_SHARPNESS * sharpness
        + W_POSE * pose
        + W_BRIGHTNESS * brightness
    )
    score = float(max(0.0, min(score, 1.0)))
    logger.debug(
        "quality score=%.3f (area=%.2f sharp=%.2f pose=%.2f bright=%.2f)",
        score, area, sharpness, pose, brightness,
    )
    return score
