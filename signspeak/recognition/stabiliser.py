"""Turns a noisy per-frame stream of letter guesses into typed text.

Rules:
  * the last `window` frames vote; a letter is typed once it wins at least `need`
    of them. Odd wrong frames no longer reset the count, so steady signs type
    quickly and shaky ones still get there;
  * holding a sign never types it twice. To type a double letter (the LL in
    HELLO), drop or change your hands for a moment and sign it again;
  * when no hands are seen for `space_after` seconds, one space is added.
"""

from __future__ import annotations

import time
from collections import Counter, deque
from typing import Callable


class LetterStabiliser:
    def __init__(
        self,
        window: int = 10,
        need: int = 7,
        space_after: float = 1.5,
        release: int = 5,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not 0 < need <= window:
            raise ValueError("need must be between 1 and window")
        self.need = need
        self.space_after = space_after
        self.release = release
        self._clock = clock
        self.text = ""
        self._history: deque[str | None] = deque(maxlen=window)
        self._locked: str | None = None  # letter already typed for the current hold
        self._hands_gone_since: float | None = None

    def _leader(self) -> tuple[str | None, int]:
        counts = Counter(l for l in self._history if l is not None)
        if not counts:
            return None, 0
        return counts.most_common(1)[0]

    @property
    def candidate(self) -> str | None:
        return self._leader()[0]

    @property
    def progress(self) -> float:
        """0..1, how close the leading guess is to being typed."""
        label, votes = self._leader()
        if label is None or label == self._locked:
            return 0.0
        return min(1.0, votes / self.need)

    def update(self, label: str | None, hands_present: bool) -> str | None:
        """Feed one frame. Returns what was typed on this frame (a letter or " "), if anything."""
        now = self._clock()

        if not hands_present:
            self._history.clear()
            self._locked = None
            if self._hands_gone_since is None:
                self._hands_gone_since = now
            elif now - self._hands_gone_since >= self.space_after and self._needs_space():
                self.text += " "
                return " "
            return None

        self._hands_gone_since = None
        self._history.append(label)

        if self._locked is not None:
            recent = list(self._history)[-self.release :]
            if len(recent) == self.release and recent.count(self._locked) <= self.release // 2:
                # Let go: forget the old votes too, or they'd retype the letter at once
                self._locked = None
                self._history.clear()
                self._history.extend(recent)

        leader, votes = self._leader()
        if leader is not None and leader != self._locked and votes >= self.need:
            self._locked = leader
            self.text += leader
            return leader
        return None

    def _needs_space(self) -> bool:
        return bool(self.text) and not self.text.endswith(" ")

    def add_space(self) -> None:
        if self._needs_space():
            self.text += " "

    def backspace(self) -> None:
        self.text = self.text[:-1]

    def clear(self) -> None:
        self.text = ""
        self._history.clear()
        self._locked = None
        self._hands_gone_since = None
