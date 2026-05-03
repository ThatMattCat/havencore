"""Multi-factor face quality scoring.

Returns a score in [0, 1] combining four signals about a detected face:

  bbox_area     — bigger faces have more pixels to embed reliably; saturates
                  at AREA_SATURATION_PX absolute pixels (~200×200), since
                  ArcFace's input is fixed 112×112 and a face crop with at
                  least that many pixels has all the info the embedder can
                  use. Frame-relative ratio scoring penalized panoramic and
                  wide-angle cameras unfairly — same face is "small" in a
                  7680×2160 panorama but pixel-rich for embedding.
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

# Saturation point for bbox area in *absolute* pixel count, not a ratio of
# the frame. ArcFace resizes every face crop to 112×112 (12,544 px) before
# embedding, so a face crop above ~200×200 has all the information ArcFace
# can use; bigger crops add no value and a face-area-as-fraction-of-frame
# metric punishes wide-angle and panoramic cameras unfairly. 40000 ≈ 200×200,
# slightly above the embedder input so there's headroom before saturation.
AREA_SATURATION_PX = 40000.0
# Saturation point for Laplacian variance — a sharp face crop typically
# lands around this value. Resolution-independent in practice because the
# Laplacian operator's variance scales with both signal and noise.
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


def _area_score(bbox: np.ndarray) -> float:
    bw = max(0.0, float(bbox[2] - bbox[0]))
    bh = max(0.0, float(bbox[3] - bbox[1]))
    return float(min((bw * bh) / AREA_SATURATION_PX, 1.0))


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

    area = _area_score(face.bbox)
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
    bw = int(round(face.bbox[2] - face.bbox[0]))
    bh = int(round(face.bbox[3] - face.bbox[1]))
    logger.info(
        "quality score=%.3f (area=%.2f sharp=%.2f pose=%.2f bright=%.2f) "
        "face=%dx%d det=%.2f",
        score, area, sharpness, pose, brightness,
        bw, bh, float(getattr(face, "det_score", 0.0)),
    )
    return score
