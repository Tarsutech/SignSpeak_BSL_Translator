"""App logic: modes, keys, recognition and training. No camera or window code here,
which keeps it easy to test."""

from __future__ import annotations

import time
from collections import deque
from typing import Callable, Sequence

import numpy as np

from .config import Settings
from .recognition.classifier import SignModel
from .recognition.features import HandLike, features_from_raw, hands_to_raw
from .recognition.stabiliser import LetterStabiliser
from .text import ALPHABET, text_to_tokens
from .ui.fingerspell import FingerspellPlayer, LetterLibrary
from .ui.hud import HudState

KEY_ESC = 27
KEY_ENTER = (10, 13)
KEY_BACKSPACE = (8, 127)
KEY_SPACE = 32
MESSAGE_SECONDS = 3.5


class Controller:
    def __init__(
        self,
        settings: Settings,
        model: SignModel,
        library: LetterLibrary,
        speaker=None,
        listener=None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.s = settings
        self.model = model
        self.library = library
        self.speaker = speaker
        self.listener = listener
        self._clock = clock

        self.mode = "translate"
        self.stabiliser = LetterStabiliser(
            settings.vote_window, settings.vote_need, settings.space_after, clock=clock
        )
        self.player = FingerspellPlayer(settings.letter_hold, settings.space_hold, clock=clock)
        self.heard_text = ""
        self.show_landmarks = True
        self.show_help = True

        self._guess: str | None = None
        self._confidence = 0.0
        self._hands = 0
        self._capturing: str | None = None
        self._capture_raw: list[np.ndarray] = []
        self._capture_hands = 0
        self._smooth: deque[np.ndarray] = deque(maxlen=settings.smoothing_frames)
        self._smooth_hands = 0
        self._runner_up: str | None = None
        self._forget_armed = False
        self._message = ""
        self._message_level = "info"
        self._message_until = 0.0
        self._fps = 0.0
        self._last_tick: float | None = None

        if len(model) == 0:
            self.notify("No letters trained yet. Press T to teach SignSpeak your signing.", "warn")

    # ---- messages -----------------------------------------------------------
    def notify(self, text: str, level: str = "info") -> None:
        self._message, self._message_level = text, level
        self._message_until = self._clock() + MESSAGE_SECONDS

    # ---- per frame ----------------------------------------------------------
    def update(self, hands: Sequence[HandLike], aspect: float) -> None:
        now = self._clock()
        if self._last_tick is not None:
            dt = max(now - self._last_tick, 1e-6)
            self._fps = 1 / dt if self._fps == 0 else self._fps * 0.9 + (1 / dt) * 0.1
        self._last_tick = now

        raw, n = hands_to_raw(hands, aspect)
        self._hands = n

        if self.mode == "train":
            self._update_training(raw, n)
        else:
            self._update_translation(raw, n)

        self._poll_speech()

    def _smoothed(self, raw: np.ndarray, n: int) -> np.ndarray:
        """Average the last few frames of landmarks to calm MediaPipe's jitter.
        Restarts whenever the number of hands changes, so it never blends
        a one-handed frame into a two-handed one."""
        if n != self._smooth_hands:
            self._smooth.clear()
            self._smooth_hands = n
        self._smooth.append(raw)
        return np.mean(self._smooth, axis=0)

    def _update_translation(self, raw: np.ndarray, n: int) -> None:
        label, self._runner_up = None, None
        self._confidence = 0.0
        if n:
            features = features_from_raw(self._smoothed(raw, n), n)
            pred = self.model.predict(features)
            if pred is not None and pred.confidence >= self.s.min_confidence:
                label, self._confidence = pred.label, pred.confidence
                self._runner_up = pred.runner_up
        else:
            self._smooth.clear()
            self._smooth_hands = 0
        self._guess = label
        self.stabiliser.update(label, hands_present=n > 0)

    def _update_training(self, raw: np.ndarray, n: int) -> None:
        if self._capturing is None or n == 0:
            return
        # Every frame of one recording must show the same number of hands,
        # otherwise a moment where tracking lost a hand would poison the letter
        if not self._capture_raw:
            self._capture_hands = n  # the first frame decides one or two hands
        elif n != self._capture_hands:
            return
        self._capture_raw.append(raw)
        if len(self._capture_raw) >= self.s.samples_per_capture:
            letter = self._capturing
            self.model.add(letter, np.stack(self._capture_raw),
                           np.full(len(self._capture_raw), n))
            self._capturing, self._capture_raw = None, []
            self._save()
            self.notify(f"Learned {letter} ({self.model.counts().get(letter, 0)} samples)", "good")

    def _poll_speech(self) -> None:
        if self.speaker is not None and self.speaker.last_error:
            self.notify(self.speaker.last_error, "bad")
            self.speaker.last_error = None
        if self.listener is None:
            return
        while not self.listener.results.empty():
            result = self.listener.results.get_nowait()
            if result.error:
                self.notify(result.error, "bad")
                continue
            self.heard_text = result.text or ""
            tokens = text_to_tokens(self.heard_text)
            if tokens:
                self.player.play(tokens)
            else:
                self.notify("Nothing to fingerspell in that phrase", "warn")

    def _save(self) -> None:
        try:
            self.model.save(self.s.model_path)
        except OSError as exc:
            self.notify(f"Couldn't save training data: {exc}", "bad")

    # ---- keys -----------------------------------------------------------
    def handle_key(self, key: int) -> bool:
        """Returns False when the app should quit."""
        if key < 0 or key == 255:
            return True
        if self.mode == "train":
            self._train_key(key)
            return True
        return self._translate_key(key)

    def _translate_key(self, key: int) -> bool:
        ch = chr(key).lower() if 0 <= key < 256 else ""
        if ch == "q" or key == KEY_ESC:
            return False
        if key == KEY_SPACE:
            self.stabiliser.add_space()
        elif key in KEY_BACKSPACE:
            self.stabiliser.backspace()
        elif ch == "c":
            self._speak()
        elif ch == "v":
            self._listen()
        elif ch == "r":
            self.stabiliser.clear()
            self.heard_text = ""
            self.player.stop()
            self.notify("Cleared")
        elif ch == "t":
            self.mode = "train"
            self.player.stop()
            self.notify("Training mode. Hold a sign and press its letter.", "info")
        elif ch == "l":
            self.show_landmarks = not self.show_landmarks
        elif ch == "h":
            self.show_help = not self.show_help
        return True

    def _train_key(self, key: int) -> None:
        if key == KEY_ESC or key in KEY_ENTER:
            self._capturing, self._capture_raw = None, []
            self._forget_armed = False
            self.mode = "translate"
            self.stabiliser.clear()
            n = len(self.model.labels)
            self.notify(f"Back to translating. {n} letter{'s' if n != 1 else ''} trained.", "good")
            return
        if key in KEY_BACKSPACE:
            self._forget_armed = True
            self.notify("Press the letter to forget", "warn")
            return
        ch = chr(key).upper() if 0 <= key < 256 else ""
        if ch not in ALPHABET or len(ch) != 1:
            return
        if self._forget_armed:
            self._forget_armed = False
            removed = self.model.remove(ch)
            self._save()
            self.notify(f"Forgot {ch} ({removed} samples)", "warn")
            return
        if self._capturing:
            return
        if self._hands == 0:
            self.notify("Show your hands first, then press the letter", "warn")
            return
        self._capturing, self._capture_raw = ch, []
        self.notify(f"Recording {ch}... hold it steady", "info")

    def _speak(self) -> None:
        text = self.stabiliser.text.strip()
        if not text:
            self.notify("Nothing to say yet. Sign some letters first.", "warn")
        elif self.speaker is None:
            self.notify("Text to speech isn't available", "bad")
        else:
            self.speaker.say(text)

    def _listen(self) -> None:
        if self.listener is None:
            self.notify("Voice input isn't available", "bad")
        elif not self.listener.start():
            self.notify("Already listening", "warn")

    # ---- view -----------------------------------------------------------
    def hud_state(self) -> HudState:
        now = self._clock()
        current = self.player.current()
        card, index = None, -1
        if current is not None:
            index, tok = current
            card = self.library[tok] if tok != " " else None
            if card is None:  # keep the previous letter's space visible as a blank
                card = np.full((self.library.size, self.library.size, 3), 246, np.uint8)

        listen_state = None
        if self.listener is not None and self.listener.busy:
            listen_state = self.listener.state.value

        return HudState(
            mode=self.mode,
            fps=self._fps,
            signed_text=self.stabiliser.text,
            heard_text=self.heard_text,
            message=self._message if now < self._message_until else "",
            message_level=self._message_level,
            guess=self._guess,
            confidence=self._confidence,
            progress=self.stabiliser.progress,
            hands=self._hands,
            model_letters=len(self.model.labels),
            train_counts=self.model.counts(),
            train_target=self.s.samples_per_capture,
            capturing=self._capturing,
            capture_progress=len(self._capture_raw) / self.s.samples_per_capture,
            runner_up=self._runner_up,
            listen_state=listen_state,
            speaking=bool(self.speaker and self.speaker.busy),
            card=card,
            spell_tokens=self.player.tokens if current else [],
            spell_index=index,
            show_help=self.show_help,
        )
