"""All tunable settings in one place."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Settings:
    # Camera
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    mirror: bool = True  # selfie view; also makes MediaPipe's handedness labels correct

    # Hand tracking (model_complexity 0 is fastest, 1 is more accurate)
    max_hands: int = 2
    model_complexity: int = 0
    detection_confidence: float = 0.6
    tracking_confidence: float = 0.5

    # Recognition
    knn_k: int = 5
    min_confidence: float = 0.6  # share of neighbour votes the winning letter needs
    max_distance: float = 4.0  # reject poses far from anything trained
    max_samples_per_letter: int = 300

    # Turning a stream of predictions into text
    stable_frames: int = 8  # frames a letter must hold before it is typed
    space_after: float = 1.5  # seconds with no hands before a space is added

    # Training
    samples_per_capture: int = 40

    # Speech
    voice: bool = True  # microphone input and spoken output
    tts_rate: int = 160
    listen_timeout: float = 5.0
    phrase_time_limit: float = 8.0

    # Speech -> fingerspelling playback
    letter_hold: float = 0.8  # seconds each letter is shown
    space_hold: float = 0.5

    # Files
    model_path: Path = PROJECT_ROOT / "data" / "model.npz"
    letters_dir: Path = PROJECT_ROOT / "assets" / "bsl_letters"

    window_name: str = "SignSpeak - BSL Translator"
