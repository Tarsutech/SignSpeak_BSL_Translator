"""Everything drawn on top of the camera image. Pure drawing: no app logic in here."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..text import ALPHABET, wrap_text

# Colours are BGR
INK = (245, 242, 238)
MUTED = (175, 168, 160)
PANEL = (34, 28, 24)
ACCENT = (196, 164, 38)  # teal-blue
GOOD = (120, 200, 90)
WARN = (60, 170, 240)
BAD = (80, 80, 225)

FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_SMALL = cv2.FONT_HERSHEY_SIMPLEX
AA = cv2.LINE_AA


@dataclass
class HudState:
    mode: str = "translate"  # "translate" or "train"
    fps: float = 0.0
    signed_text: str = ""
    heard_text: str = ""
    message: str = ""
    message_level: str = "info"  # info | good | warn | bad
    guess: str | None = None
    confidence: float = 0.0
    progress: float = 0.0
    hands: int = 0
    model_letters: int = 0
    train_counts: dict[str, int] = field(default_factory=dict)
    train_target: int = 40
    capturing: str | None = None
    capture_progress: float = 0.0
    listen_state: str | None = None
    speaking: bool = False
    card: np.ndarray | None = None
    spell_tokens: list[str] = field(default_factory=list)
    spell_index: int = -1
    show_help: bool = True


def _panel(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, alpha: float = 0.72,
           color: tuple[int, int, int] = PANEL) -> None:
    """Translucent rectangle. Blends only the region, not the whole frame."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    overlay = np.empty_like(roi)
    overlay[:] = color
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, dst=roi)


def _text(img, text, org, scale, color=INK, thickness=1, font=FONT) -> None:
    cv2.putText(img, text, org, font, scale, color, thickness, AA)


def _text_width(text: str, scale: float, thickness: int = 1, font=FONT) -> int:
    return cv2.getTextSize(text, font, scale, thickness)[0][0]


def _level_colour(level: str) -> tuple[int, int, int]:
    return {"good": GOOD, "warn": WARN, "bad": BAD}.get(level, INK)


class Hud:
    def __init__(self) -> None:
        self._scale = 1.0

    def draw(self, frame: np.ndarray, s: HudState) -> None:
        h, w = frame.shape[:2]
        self._scale = k = w / 1280
        pad = int(18 * k)

        self._top_bar(frame, s, w, pad, k)
        bottom_h = self._bottom_panel(frame, s, w, h, pad, k)

        if s.mode == "train":
            self._train_panel(frame, s, w, h, pad, k, bottom_h)
        else:
            self._guess_badge(frame, s, pad, k)

        if s.card is not None:
            self._card_panel(frame, s, w, pad, k)

    # ---- pieces -----------------------------------------------------------
    def _top_bar(self, img, s: HudState, w, pad, k) -> None:
        bar_h = int(52 * k)
        _panel(img, 0, 0, w, bar_h, 0.6)
        mode = "TRAIN" if s.mode == "train" else "TRANSLATE"
        pill_col = WARN if s.mode == "train" else ACCENT
        tw = _text_width(mode, 0.6 * k, 1)
        cv2.rectangle(img, (pad, int(12 * k)), (pad + tw + int(24 * k), bar_h - int(12 * k)),
                      pill_col, -1, AA)
        _text(img, mode, (pad + int(12 * k), int(33 * k)), 0.6 * k, PANEL, 1)

        x = pad + tw + int(40 * k)
        _text(img, "SignSpeak", (x, int(34 * k)), 0.75 * k, INK, 1)
        x += _text_width("SignSpeak", 0.75 * k) + int(16 * k)
        _text(img, "BSL fingerspelling", (x, int(34 * k)), 0.5 * k, MUTED, 1, FONT_SMALL)

        right = f"{s.fps:4.0f} fps   hands {s.hands}   letters trained {s.model_letters}/26"
        _text(img, right, (w - pad - _text_width(right, 0.5 * k, 1, FONT_SMALL), int(33 * k)),
              0.5 * k, MUTED, 1, FONT_SMALL)

    def _guess_badge(self, img, s: HudState, pad, k) -> None:
        size = int(118 * k)
        x0, y0 = pad, int(70 * k)
        _panel(img, x0, y0, x0 + size, y0 + size + int(30 * k), 0.7)
        cx, cy, r = x0 + size // 2, y0 + size // 2, int(46 * k)

        if s.model_letters == 0:
            _text(img, "?", (cx - int(12 * k), cy + int(16 * k)), 1.5 * k, MUTED, 2)
            _text(img, "press T", (x0 + int(22 * k), y0 + size + int(18 * k)), 0.45 * k,
                  MUTED, 1, FONT_SMALL)
            return

        cv2.circle(img, (cx, cy), r, (70, 62, 56), max(2, int(5 * k)), AA)
        if s.progress > 0:
            cv2.ellipse(img, (cx, cy), (r, r), -90, 0, 360 * s.progress, ACCENT,
                        max(2, int(5 * k)), AA)
        letter = s.guess or "-"
        col = INK if s.guess else MUTED
        lw = _text_width(letter, 1.6 * k, 2)
        _text(img, letter, (cx - lw // 2, cy + int(19 * k)), 1.6 * k, col, 2)

        label = f"{s.confidence * 100:3.0f}% sure" if s.guess else "no match"
        _text(img, label, (x0 + int(14 * k), y0 + size + int(18 * k)), 0.45 * k, MUTED, 1,
              FONT_SMALL)

    def _card_panel(self, img, s: HudState, w, pad, k) -> None:
        card_size = int(260 * k)
        card = cv2.resize(s.card, (card_size, card_size), interpolation=cv2.INTER_AREA)
        x1 = w - pad
        x0 = x1 - card_size
        y0 = int(70 * k)
        _panel(img, x0 - int(10 * k), y0 - int(10 * k), x1 + int(10 * k),
               y0 + card_size + int(62 * k), 0.8)
        img[y0 : y0 + card_size, x0:x1] = card

        # Word strip with the current letter highlighted
        tokens = s.spell_tokens
        if not tokens:
            return
        window = 14
        start = max(0, min(s.spell_index - window // 2, len(tokens) - window))
        shown = tokens[start : start + window]
        scale = 0.62 * k
        cell = int(18 * k)
        total = cell * len(shown)
        x = x0 + (card_size - total) // 2
        y = y0 + card_size + int(36 * k)
        for i, tok in enumerate(shown):
            is_now = start + i == s.spell_index
            ch = "_" if tok == " " else tok
            col = ACCENT if is_now else (INK if start + i < s.spell_index else MUTED)
            _text(img, ch, (x + i * cell, y), scale, col, 2 if is_now else 1)

    def _train_panel(self, img, s: HudState, w, h, pad, k, bottom_h) -> None:
        cols = 13
        cell_w, cell_h = int(58 * k), int(54 * k)
        grid_w = cols * cell_w
        x0 = pad
        y0 = int(70 * k)
        _panel(img, x0, y0, x0 + grid_w + int(20 * k), y0 + 2 * cell_h + int(76 * k), 0.78)
        _text(img, "Hold a sign, then press its letter key to record it", (x0 + int(12 * k),
              y0 + int(26 * k)), 0.55 * k, INK, 1)

        for i, letter in enumerate(ALPHABET):
            cx = x0 + int(10 * k) + (i % cols) * cell_w
            cy = y0 + int(42 * k) + (i // cols) * cell_h
            n = s.train_counts.get(letter, 0)
            done = n >= s.train_target
            box_col = GOOD if done else ((90, 80, 72) if n == 0 else WARN)
            if s.capturing == letter:
                box_col = ACCENT
            cv2.rectangle(img, (cx, cy), (cx + cell_w - int(6 * k), cy + cell_h - int(6 * k)),
                          box_col, max(1, int(2 * k)), AA)
            _text(img, letter, (cx + int(8 * k), cy + int(27 * k)), 0.7 * k, INK, 1)
            _text(img, str(n), (cx + int(8 * k), cy + int(43 * k)), 0.38 * k, MUTED, 1,
                  FONT_SMALL)
            if s.capturing == letter:
                bw = int((cell_w - int(6 * k)) * s.capture_progress)
                cv2.rectangle(img, (cx, cy + cell_h - int(10 * k)),
                              (cx + bw, cy + cell_h - int(6 * k)), ACCENT, -1)

        tip = ("Recording... keep the sign steady and move slightly so it learns small variations"
               if s.capturing else
               "Aim for 40+ samples per letter. To redo one: Backspace, then its key")
        _text(img, tip, (x0 + int(12 * k), y0 + 2 * cell_h + int(62 * k)), 0.45 * k, MUTED, 1,
              FONT_SMALL)

    def _bottom_panel(self, img, s: HudState, w, h, pad, k) -> int:
        line_h = int(36 * k)
        max_chars = max(10, int((w - 2 * pad) / (17 * k)))
        signed_lines = wrap_text(s.signed_text + "|", max_chars)[-2:]
        heard_lines = wrap_text(s.heard_text, max_chars)[-1:] if s.heard_text else []

        rows = 1 + len(signed_lines) + (1 + len(heard_lines) if heard_lines else 0)
        help_h = int(30 * k) if s.show_help else 0
        msg_h = int(28 * k) if s.message else 0
        panel_h = int(24 * k) + rows * line_h + help_h + msg_h
        y0 = h - panel_h
        _panel(img, 0, y0, w, h, 0.78)

        y = y0 + int(30 * k)
        _text(img, "SIGNED", (pad, y), 0.45 * k, ACCENT, 1, FONT_SMALL)
        status = self._audio_status(s)
        if status:
            _text(img, status[0], (w - pad - _text_width(status[0], 0.5 * k, 1, FONT_SMALL), y),
                  0.5 * k, status[1], 1, FONT_SMALL)
        for line in signed_lines:
            y += line_h
            _text(img, line, (pad, y - int(6 * k)), 0.95 * k, INK, 1)

        if heard_lines:
            y += line_h
            _text(img, "HEARD", (pad, y - int(6 * k)), 0.45 * k, ACCENT, 1, FONT_SMALL)
            for line in heard_lines:
                y += line_h
                _text(img, line, (pad, y - int(10 * k)), 0.8 * k, (200, 225, 190), 1)

        if s.message:
            y += msg_h
            _text(img, s.message, (pad, y - int(4 * k)), 0.52 * k, _level_colour(s.message_level),
                  1, FONT_SMALL)

        if s.show_help:
            y += help_h
            keys = (
                "A-Z record letter   Backspace then letter: forget it   Enter/Esc done"
                if s.mode == "train"
                else "Space space   Backspace delete   C speak   V voice to signs   "
                     "R reset   T train   L landmarks   H help   Q quit"
            )
            _text(img, keys, (pad, y - int(2 * k)), 0.45 * k, MUTED, 1, FONT_SMALL)
        return panel_h

    @staticmethod
    def _audio_status(s: HudState) -> tuple[str, tuple[int, int, int]] | None:
        if s.listen_state == "calibrating":
            return "Checking background noise...", WARN
        if s.listen_state == "listening":
            return "Listening - speak now", BAD
        if s.listen_state == "recognising":
            return "Working out what you said...", WARN
        if s.speaking:
            return "Speaking...", GOOD
        return None
