"""Webcam capture on a background thread so the UI never waits on the camera."""

from __future__ import annotations

import sys
import threading
import time

import cv2
import numpy as np


class Camera:
    """Keeps only the newest frame. `read()` never blocks for long."""

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720) -> None:
        # DirectShow opens much faster on Windows. Other platforms use the default backend.
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(index)

        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open camera {index}. Check it is plugged in and not in use "
                "by another app (Teams, Zoom, the browser), or try --camera 1."
            )

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            raise RuntimeError(f"Camera {index} opened but returned no image.")

        self._cap = cap
        self._frame: np.ndarray = frame
        self._frame_id = 0
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        failures = 0
        while self._running:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                failures += 1
                if failures > 50:
                    self._running = False
                time.sleep(0.01)
                continue
            failures = 0
            with self._lock:
                self._frame = frame
                self._frame_id += 1

    @property
    def alive(self) -> bool:
        return self._running

    def read(self) -> tuple[int, np.ndarray]:
        """Return (frame_id, frame). The id only changes when a new frame arrives."""
        with self._lock:
            return self._frame_id, self._frame

    def release(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        self._cap.release()

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, *_exc) -> None:
        self.release()
