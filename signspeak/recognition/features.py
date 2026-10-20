"""Turn hand landmarks into features that capture what makes BSL letters different.

Most BSL fingerspelling is about where one hand touches the other (vowels are
pointing at a fingertip, many consonants are a shape resting on the other palm).
So besides each hand's own shape, the features include the distance from every
fingertip of one hand to every fingertip of the other.

Training stores the *raw* landmarks, not these features, so the feature recipe
can improve later without anyone having to retrain.
"""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

N_LANDMARKS = 21
WRIST = 0
PALM = (0, 5, 9, 13, 17)  # rigid points used to measure hand size
KEY_A = (0, 4, 8, 12, 16, 20, 9)  # first hand: wrist, 5 fingertips, palm centre
KEY_B = (4, 8, 12, 16, 20, 9)  # second hand: 5 fingertips, palm centre

HAND_DIM = N_LANDMARKS * 3
CROSS_DIM = len(KEY_A) * len(KEY_B)
# [hand 1 | hand 2 | hand-to-hand distances | wrist offset (x, y) | presence (2)]
FEATURE_DIM = 2 * HAND_DIM + CROSS_DIM + 2 + 2

# Relative weights of each group in the distance the classifier uses
Z_WEIGHT = 0.35  # MediaPipe depth is noisy
CROSS_WEIGHT = 1.6  # contact between hands is the most telling cue in BSL
OFFSET_WEIGHT = 1.0
PRESENCE_WEIGHT = 3.0  # one-handed and two-handed signs should never be confused


class HandLike(Protocol):
    label: str
    points: np.ndarray


def to_metric(points: np.ndarray, aspect: float) -> np.ndarray:
    """MediaPipe x, y are fractions of width and height. Put x, y, z on one scale."""
    pts = np.asarray(points, dtype=np.float32).copy()
    pts[:, 0] *= aspect
    pts[:, 2] *= aspect
    return pts


def hands_to_raw(hands: Sequence[HandLike], aspect: float = 16 / 9) -> tuple[np.ndarray, int]:
    """Pack up to two hands into a (2, 21, 3) array, ordered left to right on screen.

    Ordering by position is far steadier than MediaPipe's left/right labels,
    which often flip when hands touch or overlap, exactly what BSL does.
    """
    pts = [to_metric(h.points, aspect) for h in list(hands)[:2]]
    pts.sort(key=lambda p: float(p[WRIST, 0]))
    raw = np.zeros((2, N_LANDMARKS, 3), dtype=np.float32)
    for i, p in enumerate(pts):
        raw[i] = p
    return raw, len(pts)


def hand_scale(points: np.ndarray) -> float:
    """Size of the palm, robust to the hand turning towards the camera."""
    palm = points[list(PALM), :2]
    return float(np.mean(np.linalg.norm(palm - palm.mean(axis=0), axis=1)))


def normalise_hand(points: np.ndarray, scale: float) -> np.ndarray:
    if scale < 1e-6:
        return np.zeros(HAND_DIM, dtype=np.float32)
    out = (points - points[WRIST]) / scale
    out[:, 2] *= Z_WEIGHT
    return out.ravel()


def features_from_raw(raw: np.ndarray, n_hands: int) -> np.ndarray | None:
    if n_hands <= 0:
        return None
    vec = np.zeros(FEATURE_DIM, dtype=np.float32)
    scales = [hand_scale(raw[i]) for i in range(n_hands)]
    for i in range(n_hands):
        vec[i * HAND_DIM : (i + 1) * HAND_DIM] = normalise_hand(raw[i], scales[i])

    o = 2 * HAND_DIM
    if n_hands == 2:
        s = max((scales[0] + scales[1]) / 2, 1e-6)
        a = raw[0][list(KEY_A), :2]
        b = raw[1][list(KEY_B), :2]
        cross = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2) / s
        # Close contacts matter; how far apart two distant fingers are does not
        vec[o : o + CROSS_DIM] = CROSS_WEIGHT * np.minimum(cross, 8.0).ravel()
        vec[o + CROSS_DIM : o + CROSS_DIM + 2] = (
            OFFSET_WEIGHT * (raw[1, WRIST, :2] - raw[0, WRIST, :2]) / s
        )
    vec[-2] = PRESENCE_WEIGHT
    vec[-1] = PRESENCE_WEIGHT if n_hands == 2 else 0.0
    return vec


def build_features(hands: Sequence[HandLike], aspect: float = 16 / 9) -> np.ndarray | None:
    raw, n = hands_to_raw(hands, aspect)
    return features_from_raw(raw, n)


def augment(raw: np.ndarray, n_hands: int, copies: int, rng: np.random.Generator) -> np.ndarray:
    """Make slightly rotated and jittered copies of a sample.

    Gives the classifier the small variations you'd naturally produce on a
    different day, so 40 captured frames go a lot further.
    """
    out = np.repeat(raw[None], copies, axis=0)
    live = out[:, :n_hands]
    centre = live[:, :, :, :2].reshape(copies, -1, 2).mean(axis=1)[:, None, None, :]
    angles = rng.uniform(-0.14, 0.14, copies)  # about +/- 8 degrees
    cos, sin = np.cos(angles)[:, None, None], np.sin(angles)[:, None, None]
    xy = live[..., :2] - centre
    rotated = np.stack(
        [xy[..., 0] * cos - xy[..., 1] * sin, xy[..., 0] * sin + xy[..., 1] * cos], axis=-1
    )
    live[..., :2] = rotated * rng.uniform(0.93, 1.07, copies)[:, None, None, None] + centre
    scale = np.mean([hand_scale(raw[i]) for i in range(n_hands)]) if n_hands else 0.0
    live += rng.normal(0, 0.025 * scale, live.shape).astype(np.float32)
    out[:, :n_hands] = live
    return out
