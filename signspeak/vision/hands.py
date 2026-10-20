"""MediaPipe hand tracking, wrapped so the rest of the app only sees plain numpy arrays."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

try:
    import mediapipe as mp

    _mp_hands = mp.solutions.hands
    _mp_draw = mp.solutions.drawing_utils
    _mp_styles = mp.solutions.drawing_styles
except (ImportError, AttributeError) as exc:  # pragma: no cover - depends on install
    raise ImportError(
        "MediaPipe with the 'solutions' API is required. Install the pinned version: "
        "pip install mediapipe==0.10.14"
    ) from exc


@dataclass(frozen=True)
class HandObservation:
    label: str  # "Left" or "Right", from the signer's point of view (with a mirrored frame)
    score: float
    points: np.ndarray  # (21, 3) x, y in 0..1 image space, z relative depth


class HandTracker:
    def __init__(
        self,
        max_hands: int = 2,
        model_complexity: int = 0,
        detection_confidence: float = 0.6,
        tracking_confidence: float = 0.5,
    ) -> None:
        self._hands = _mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            model_complexity=model_complexity,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._last_raw = None

    def process(self, frame_bgr: np.ndarray) -> list[HandObservation]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False  # lets MediaPipe skip a copy
        results = self._hands.process(rgb)
        self._last_raw = results

        if not results.multi_hand_landmarks:
            return []

        observations: list[HandObservation] = []
        for landmarks, handedness in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            cls = handedness.classification[0]
            points = np.array(
                [(lm.x, lm.y, lm.z) for lm in landmarks.landmark], dtype=np.float32
            )
            observations.append(HandObservation(cls.label, float(cls.score), points))
        return observations

    def draw(self, frame_bgr: np.ndarray) -> None:
        """Draw the landmarks from the most recent `process` call."""
        results = self._last_raw
        if results is None or not results.multi_hand_landmarks:
            return
        for landmarks in results.multi_hand_landmarks:
            _mp_draw.draw_landmarks(
                frame_bgr,
                landmarks,
                _mp_hands.HAND_CONNECTIONS,
                _mp_styles.get_default_hand_landmarks_style(),
                _mp_styles.get_default_hand_connections_style(),
            )

    def close(self) -> None:
        self._hands.close()
