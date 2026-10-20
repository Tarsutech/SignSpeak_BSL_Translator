"""Turns a noisy per-frame stream of letter guesses into typed text.

Rules:
  * a letter is typed once it has been the guess for `stable_frames` frames in a row;
  * holding a sign never types it twice, and a brief flicker of a different guess
    mid-hold does not count as letting go. To type a double letter (the LL in HELLO),
    drop or change your hand for a moment and sign it again;
  * when no hands are seen for `space_after` seconds, one space is added.
"""

from __future__ import annotations

import time
from typing import Callable


class LetterStabiliser:
    def __init__(
        self,
        stable_frames: int = 8,
        space_after: float = 1.5,
        release_frames: int = 4,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.stable_frames = stable_frames
        self.space_after = space_after
        self.release_frames = release_frames
        self._clock = clock
        self.text = ""
        self._candidate: str | None = None
        self._count = 0
        self._locked: str | None = None  # letter already typed for the current hold
        self._hands_gone_since: float | None = None

    @property
    def candidate(self) -> str | None:
        return self._candidate

    @property
    def progress(self) -> float:
        """0..1, how close the current guess is to being typed."""
        if self._candidate is None or self._candidate == self._locked:
            return 0.0
        return min(1.0, self._count / self.stable_frames)

    def update(self, label: str | None, hands_present: bool) -> str | None:
        """Feed one frame. Returns what was typed on this frame (a letter or " "), if anything."""
        now = self._clock()

        if not hands_present:
            self._reset_hold()
            if self._hands_gone_since is None:
                self._hands_gone_since = now
            elif now - self._hands_gone_since >= self.space_after and self._needs_space():
                self.text += " "
                return " "
            return None

        self._hands_gone_since = None

        if label == self._candidate:
            self._count += 1
        else:
            self._candidate, self._count = label, 1

        if (
            self._locked is not None
            and label != self._locked
            and self._count >= self.release_frames
        ):
            self._locked = None

        if label is None or label == self._locked:
            return None

        if self._count >= self.stable_frames:
            self._locked = label
            self.text += label
            return label
        return None

    def _needs_space(self) -> bool:
        return bool(self.text) and not self.text.endswith(" ")

    def _reset_hold(self) -> None:
        self._candidate = None
        self._count = 0
        self._locked = None

    def add_space(self) -> None:
        if self._needs_space():
            self.text += " "

    def backspace(self) -> None:
        self.text = self.text[:-1]

    def clear(self) -> None:
        self.text = ""
        self._reset_hold()
        self._hands_gone_since = None
