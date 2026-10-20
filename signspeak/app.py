"""Entry point: wires the camera, hand tracker, controller and window together."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from .config import PROJECT_ROOT, Settings
from .controller import Controller
from .recognition.classifier import KNNClassifier
from .ui.fingerspell import LetterLibrary
from .ui.hud import Hud


def parse_args(argv: list[str] | None = None) -> Settings:
    s = Settings()
    p = argparse.ArgumentParser(description="SignSpeak: BSL fingerspelling translator")
    p.add_argument("--camera", type=int, default=s.camera_index, help="webcam index (default 0)")
    p.add_argument("--width", type=int, default=s.frame_width)
    p.add_argument("--height", type=int, default=s.frame_height)
    p.add_argument("--accurate", action="store_true",
                   help="use the heavier hand model (more accurate, slower)")
    p.add_argument("--model", type=Path, default=s.model_path, help="training data file")
    p.add_argument("--letters", type=Path, default=s.letters_dir,
                   help="folder of BSL letter pictures named A.png ... Z.png")
    p.add_argument("--no-voice", action="store_true", help="disable microphone and speech")
    a = p.parse_args(argv)
    s.camera_index, s.frame_width, s.frame_height = a.camera, a.width, a.height
    s.model_complexity = 1 if a.accurate else 0
    s.model_path, s.letters_dir = a.model, a.letters
    s.voice = not a.no_voice
    return s


def main(argv: list[str] | None = None) -> int:
    settings = parse_args(argv)

    # Imported here so `--help` works even before the heavy packages are installed
    from .vision.camera import Camera
    from .vision.hands import HandTracker

    model = KNNClassifier.load(settings.model_path, settings.knn_k,
                               settings.max_samples_per_letter)
    # Also accept the old repo's bsl_letters/ folder
    library = LetterLibrary([settings.letters_dir, PROJECT_ROOT / "bsl_letters"])
    if library.missing:
        print(f"Using placeholder cards for {len(library.missing)} letters with no picture "
              f"in {settings.letters_dir}")

    speaker = listener = None
    if settings.voice:
        from .audio.stt import SpeechListener
        from .audio.tts import Speaker

        speaker = Speaker(settings.tts_rate)
        listener = SpeechListener(settings.listen_timeout, settings.phrase_time_limit)

    controller = Controller(settings, model, library, speaker, listener)
    hud = Hud()
    tracker = HandTracker(settings.max_hands, settings.model_complexity,
                          settings.detection_confidence, settings.tracking_confidence)

    try:
        camera = Camera(settings.camera_index, settings.frame_width, settings.frame_height)
    except RuntimeError as exc:
        print(exc)
        tracker.close()
        return 1

    cv2.namedWindow(settings.window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(settings.window_name, settings.frame_width, settings.frame_height)

    last_id = -1
    camera_died = False
    try:
        while True:
            if not camera.alive:
                camera_died = True
                break
            frame_id, raw = camera.read()
            if frame_id == last_id:
                # No new frame yet: stay responsive to keys without redoing work
                if not controller.handle_key(cv2.waitKey(5) & 0xFF):
                    break
                continue
            last_id = frame_id

            frame = cv2.flip(raw, 1) if settings.mirror else raw.copy()
            h, w = frame.shape[:2]
            hands = tracker.process(frame)
            controller.update(hands, aspect=w / h)

            if controller.show_landmarks:
                tracker.draw(frame)
            hud.draw(frame, controller.hud_state())
            cv2.imshow(settings.window_name, frame)

            if not controller.handle_key(cv2.waitKey(1) & 0xFF):
                break
            if cv2.getWindowProperty(settings.window_name, cv2.WND_PROP_VISIBLE) < 1:
                break  # window closed with the X button
    finally:
        camera.release()
        tracker.close()
        if speaker:
            speaker.close()
        cv2.destroyAllWindows()

    if camera_died:
        print("The camera stopped sending images.")
        return 1
    return 0
