"""Turn raw hand landmarks into a feature vector that ignores where you stand and how big you are.

BSL fingerspelling is two-handed, so the vector covers both hands and how they sit
relative to each other. Each hand is:
  * moved so the wrist is the origin,
  * scaled so wrist -> middle-finger knuckle is length 1.
The offset between the two wrists is scaled by the right hand's size.
"""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

N_LANDMARKS = 21
WRIST = 0
MIDDLE_MCP = 9
HAND_DIM = N_LANDMARKS * 3
# [right hand | left hand | wrist offset (x, y, z) | right present | left present]
FEATURE_DIM = 2 * HAND_DIM + 3 + 2

_SLOTS = ("Right", "Left")


class HandLike(Protocol):
    label: str
    points: np.ndarray


def _to_metric(points: np.ndarray, aspect: float) -> np.ndarray:
    """MediaPipe x and y are fractions of width and height. Put them on the same scale."""
    pts = np.asarray(points, dtype=np.float32).copy()
    pts[:, 0] *= aspect
    pts[:, 2] *= aspect  # z uses roughly the same scale as x
    return pts


def hand_size(points: np.ndarray) -> float:
    return float(np.linalg.norm(points[MIDDLE_MCP, :2] - points[WRIST, :2]))


def normalise_hand(points: np.ndarray) -> np.ndarray:
    """(21, 3) metric points -> flat (63,) vector, wrist-centred and size-normalised."""
    centred = points - points[WRIST]
    size = hand_size(points)
    if size < 1e-6:
        return np.zeros(HAND_DIM, dtype=np.float32)
    return (centred / size).ravel().astype(np.float32)


def assign_slots(hands: Sequence[HandLike]) -> list[HandLike | None]:
    """Put hands into [right, left] slots.

    MediaPipe sometimes labels both hands the same. When that happens the hand
    further to the right of the (mirrored) image is treated as the right hand.
    """
    slots: list[HandLike | None] = [None, None]
    hands = list(hands)[:2]
    labels = [h.label for h in hands]
    if len(hands) == 2 and labels[0] == labels[1]:
        hands.sort(key=lambda h: -float(np.mean(h.points[:, 0])))
        return [hands[0], hands[1]]
    for hand in hands:
        idx = _SLOTS.index(hand.label) if hand.label in _SLOTS else 0
        if slots[idx] is None:
            slots[idx] = hand
        else:
            slots[1 - idx] = hand
    return slots


def build_features(hands: Sequence[HandLike], aspect: float = 16 / 9) -> np.ndarray | None:
    """Return a FEATURE_DIM vector, or None when no hands are visible."""
    if not hands:
        return None

    right, left = assign_slots(hands)
    vec = np.zeros(FEATURE_DIM, dtype=np.float32)
    metric = {}

    for slot, hand in enumerate((right, left)):
        if hand is None:
            continue
        pts = _to_metric(hand.points, aspect)
        metric[slot] = pts
        vec[slot * HAND_DIM : (slot + 1) * HAND_DIM] = normalise_hand(pts)
        vec[2 * HAND_DIM + 3 + slot] = 1.0

    if 0 in metric and 1 in metric:
        ref = hand_size(metric[0]) or 1.0
        offset = (metric[1][WRIST] - metric[0][WRIST]) / ref
        vec[2 * HAND_DIM : 2 * HAND_DIM + 3] = offset

    return vec
