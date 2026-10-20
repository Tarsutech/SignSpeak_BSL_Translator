"""Text to speech on a background thread, so speaking never freezes the camera."""

from __future__ import annotations

import queue
import shutil
import subprocess
import sys
import threading


def _speak_with_system(text: str) -> bool:
    """Fallback that uses the OS's own voice. Returns True if it ran."""
    try:
        if sys.platform.startswith("win"):
            safe = text.replace("'", "''")
            script = (
                "Add-Type -AssemblyName System.Speech; "
                "(New-Object System.Speech.Synthesis.SpeechSynthesizer)"
                f".Speak('{safe}')"
            )
            subprocess.run(
                ["powershell", "-NoProfile", "-Command", script],
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            return True
        if sys.platform == "darwin" and shutil.which("say"):
            subprocess.run(["say", text], check=False)
            return True
        for cmd in ("espeak-ng", "espeak"):
            if shutil.which(cmd):
                subprocess.run([cmd, text], check=False)
                return True
    except OSError:
        pass
    return False


class Speaker:
    def __init__(self, rate: int = 160) -> None:
        self.rate = rate
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._busy = threading.Event()
        self.last_error: str | None = None
        self._thread = threading.Thread(target=self._worker, name="tts", daemon=True)
        self._thread.start()

    @property
    def busy(self) -> bool:
        return self._busy.is_set() or not self._queue.empty()

    def say(self, text: str) -> None:
        text = text.strip()
        if text:
            self._queue.put(text)

    def _speak_pyttsx3(self, text: str) -> None:
        import pyttsx3

        # A fresh engine per phrase avoids pyttsx3's known hang on the second
        # runAndWait() call, which is what the old PowerShell fallback was papering over.
        engine = pyttsx3.init()
        try:
            engine.setProperty("rate", self.rate)
            engine.say(text)
            engine.runAndWait()
        finally:
            engine.stop()

    def _worker(self) -> None:
        while True:
            text = self._queue.get()
            if text is None:
                return
            self._busy.set()
            try:
                self._speak_pyttsx3(text)
                self.last_error = None
            except Exception as exc:  # noqa: BLE001 - any engine failure falls back
                if not _speak_with_system(text):
                    self.last_error = f"Text to speech unavailable: {exc}"
            finally:
                self._busy.clear()

    def close(self) -> None:
        self._queue.put(None)
