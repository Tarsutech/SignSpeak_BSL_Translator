"""Speech to text on a background thread.

The old version re-measured background noise on every press, which added a second
of dead air each time. Here it is measured once, then the listener is ready instantly.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from enum import Enum


class ListenState(Enum):
    IDLE = "idle"
    CALIBRATING = "calibrating"
    LISTENING = "listening"
    RECOGNISING = "recognising"


@dataclass(frozen=True)
class SpeechResult:
    text: str | None
    error: str | None


class SpeechListener:
    def __init__(self, timeout: float = 5.0, phrase_time_limit: float = 8.0) -> None:
        self.timeout = timeout
        self.phrase_time_limit = phrase_time_limit
        self.state = ListenState.IDLE
        self.results: queue.Queue[SpeechResult] = queue.Queue()
        self._recogniser = None
        self._microphone = None
        self._calibrated = False
        self._lock = threading.Lock()

    @property
    def busy(self) -> bool:
        return self.state is not ListenState.IDLE

    def start(self) -> bool:
        """Begin listening for one phrase. Returns False if already listening."""
        with self._lock:
            if self.busy:
                return False
            self.state = ListenState.CALIBRATING
        threading.Thread(target=self._listen_once, name="stt", daemon=True).start()
        return True

    def _setup(self):
        import speech_recognition as sr

        if self._recogniser is None:
            self._recogniser = sr.Recognizer()
            self._recogniser.dynamic_energy_threshold = True
            self._recogniser.pause_threshold = 0.6
            self._microphone = sr.Microphone()
        return sr

    def _listen_once(self) -> None:
        try:
            result = self._capture_and_recognise()
        except ImportError:
            result = SpeechResult(
                None, "Voice input needs SpeechRecognition and PyAudio installed"
            )
        except Exception as exc:  # noqa: BLE001 - surface anything to the UI
            result = SpeechResult(None, f"Voice input error: {exc}")
        self.results.put(result)
        self.state = ListenState.IDLE

    def _capture_and_recognise(self) -> SpeechResult:
        sr = self._setup()
        try:
            with self._microphone as source:
                if not self._calibrated:
                    self.state = ListenState.CALIBRATING
                    self._recogniser.adjust_for_ambient_noise(source, duration=0.8)
                    self._calibrated = True
                self.state = ListenState.LISTENING
                audio = self._recogniser.listen(
                    source,
                    timeout=self.timeout,
                    phrase_time_limit=self.phrase_time_limit,
                )
        except sr.WaitTimeoutError:
            return SpeechResult(None, "Didn't hear anything")
        except OSError as exc:
            return SpeechResult(None, f"Microphone error: {exc}")

        self.state = ListenState.RECOGNISING
        try:
            return SpeechResult(self._recogniser.recognize_google(audio, language="en-GB"), None)
        except sr.UnknownValueError:
            return SpeechResult(None, "Couldn't make out the words")
        except sr.RequestError as exc:
            return SpeechResult(None, f"Speech service unreachable (needs internet): {exc}")
