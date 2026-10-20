"""Speech -> fingerspelling: letter pictures and a non-blocking player.

The old version showed letters with a blocking loop, which froze the camera while
it spelled. This player is driven by the clock, so the main loop just asks
"which letter now?" every frame.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ..text import ALPHABET

CARD_SIZE = 320
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def _fit_square(img: np.ndarray, size: int) -> np.ndarray:
    """Letterbox an image onto a square card without stretching it."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        img = (img[:, :, :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
    h, w = img.shape[:2]
    scale = size / max(h, w)
    resized = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))),
                         interpolation=cv2.INTER_AREA)
    card = np.full((size, size, 3), 255, dtype=np.uint8)
    y, x = (size - resized.shape[0]) // 2, (size - resized.shape[1]) // 2
    card[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return card


def _placeholder(letter: str, size: int) -> np.ndarray:
    """A clean letter card used until a real BSL photo is added for that letter."""
    card = np.full((size, size, 3), (246, 243, 240), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = size / 70
    (tw, th), _ = cv2.getTextSize(letter, font, scale, 6)
    cv2.putText(card, letter, ((size - tw) // 2, (size + th) // 2 - 10), font, scale,
                (90, 60, 40), 6, cv2.LINE_AA)
    note = f"add {letter}.png"
    (nw, _), _ = cv2.getTextSize(note, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(card, note, ((size - nw) // 2, size - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (150, 140, 130), 1, cv2.LINE_AA)
    return card


class LetterLibrary:
    def __init__(self, folders: list[Path], size: int = CARD_SIZE) -> None:
        self.size = size
        self.cards: dict[str, np.ndarray] = {}
        self.missing: list[str] = []
        for letter in ALPHABET:
            img = self._find(folders, letter)
            if img is None:
                self.missing.append(letter)
                self.cards[letter] = _placeholder(letter, size)
            else:
                self.cards[letter] = _fit_square(img, size)

    @staticmethod
    def _find(folders: list[Path], letter: str) -> np.ndarray | None:
        for folder in folders:
            if not folder.is_dir():
                continue
            for name in (letter, letter.lower()):
                for ext in IMAGE_EXTS:
                    path = folder / f"{name}{ext}"
                    if path.exists():
                        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                        if img is not None:
                            return img
        return None

    def __getitem__(self, letter: str) -> np.ndarray:
        return self.cards[letter]


class FingerspellPlayer:
    def __init__(
        self,
        letter_hold: float = 0.8,
        space_hold: float = 0.5,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.letter_hold = letter_hold
        self.space_hold = space_hold
        self._clock = clock
        self._tokens: list[str] = []
        self._starts: list[float] = []
        self._end = 0.0

    def play(self, tokens: list[str]) -> None:
        now = self._clock()
        self._tokens = list(tokens)
        self._starts = []
        t = now
        for tok in self._tokens:
            self._starts.append(t)
            t += self.space_hold if tok == " " else self.letter_hold
        self._end = t

    def stop(self) -> None:
        self._tokens, self._starts, self._end = [], [], 0.0

    @property
    def playing(self) -> bool:
        return bool(self._tokens) and self._clock() < self._end

    def current(self) -> tuple[int, str] | None:
        """(index, token) being shown right now, or None when finished."""
        if not self.playing:
            return None
        now = self._clock()
        idx = 0
        for i, start in enumerate(self._starts):
            if start <= now:
                idx = i
            else:
                break
        return idx, self._tokens[idx]

    @property
    def tokens(self) -> list[str]:
        return self._tokens
