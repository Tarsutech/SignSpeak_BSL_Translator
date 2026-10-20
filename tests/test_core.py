"""Tests for the parts that don't need a camera or microphone. Run: python -m pytest"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from signspeak.recognition.classifier import KNNClassifier
from signspeak.recognition.features import FEATURE_DIM, HAND_DIM, build_features
from signspeak.recognition.stabiliser import LetterStabiliser
from signspeak.text import text_to_tokens, wrap_text
from signspeak.ui.fingerspell import FingerspellPlayer


@dataclass
class FakeHand:
    label: str
    points: np.ndarray


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def make_hand(label: str, seed: int, offset=(0.3, 0.5), scale=0.1) -> FakeHand:
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-1, 1, (21, 3)).astype(np.float32)
    pts[0] = 0
    pts[9] = (0, -1, 0)
    pts = pts * scale
    pts[:, 0] += offset[0]
    pts[:, 1] += offset[1]
    return FakeHand(label, pts)


# ---- features -------------------------------------------------------------
def test_features_ignore_position_and_size():
    a = make_hand("Right", 1, offset=(0.3, 0.5), scale=0.1)
    b = make_hand("Right", 1, offset=(0.6, 0.3), scale=0.2)
    fa, fb = build_features([a]), build_features([b])
    assert fa.shape == (FEATURE_DIM,)
    np.testing.assert_allclose(fa, fb, atol=1e-5)


def test_features_slot_by_handedness_and_order_independent():
    r, l = make_hand("Right", 1, (0.6, 0.5)), make_hand("Left", 2, (0.3, 0.5))
    f1, f2 = build_features([r, l]), build_features([l, r])
    np.testing.assert_allclose(f1, f2)
    assert f1[-2:].tolist() == [1.0, 1.0]
    only_left = build_features([l])
    assert np.all(only_left[:HAND_DIM] == 0) and only_left[-2:].tolist() == [0.0, 1.0]


def test_features_duplicate_labels_resolved_by_position():
    a, b = make_hand("Right", 1, (0.7, 0.5)), make_hand("Right", 2, (0.2, 0.5))
    f = build_features([b, a])
    np.testing.assert_allclose(f[:HAND_DIM], build_features([a])[:HAND_DIM], atol=1e-5)


def test_no_hands_gives_none():
    assert build_features([]) is None


# ---- classifier ---------------------------------------------------------
def test_knn_learns_and_round_trips(tmp_path):
    model = KNNClassifier(k=3)
    rng = np.random.default_rng(0)
    centres = {"A": rng.normal(size=FEATURE_DIM), "B": rng.normal(size=FEATURE_DIM)}
    for label, c in centres.items():
        model.add(label, c + rng.normal(scale=0.05, size=(20, FEATURE_DIM)))

    pred = model.predict(centres["B"])
    assert pred.label == "B" and pred.confidence > 0.99

    path = tmp_path / "model.npz"
    model.save(path)
    loaded = KNNClassifier.load(path, k=3)
    assert loaded.counts() == {"A": 20, "B": 20}
    assert loaded.predict(centres["A"]).label == "A"

    assert loaded.remove("A") == 20 and loaded.labels == ["B"]


def test_knn_caps_samples_and_rejects_bad_shape():
    model = KNNClassifier(max_samples_per_label=10)
    model.add("A", np.zeros((25, FEATURE_DIM)))
    assert model.counts() == {"A": 10}
    with pytest.raises(ValueError):
        model.add("A", np.zeros((1, 5)))
    assert KNNClassifier().predict(np.zeros(FEATURE_DIM)) is None


# ---- stabiliser -----------------------------------------------------------
def test_letter_typed_once_per_hold():
    st = LetterStabiliser(stable_frames=3, space_after=1.0, clock=FakeClock())
    typed = [st.update("A", True) for _ in range(10)]
    assert typed.count("A") == 1 and st.text == "A"


def test_flicker_does_not_retype_but_release_does():
    st = LetterStabiliser(stable_frames=3, release_frames=4, clock=FakeClock())
    for _ in range(3):
        st.update("L", True)
    st.update(None, True)  # one-frame flicker
    for _ in range(5):
        st.update("L", True)
    assert st.text == "L"
    for _ in range(4):
        st.update(None, True)  # real release
    for _ in range(3):
        st.update("L", True)
    assert st.text == "LL"


def test_space_after_hands_gone():
    clock = FakeClock()
    st = LetterStabiliser(stable_frames=2, space_after=1.0, clock=clock)
    st.update("H", True)
    st.update("H", True)
    st.update(None, False)
    clock.t = 0.5
    assert st.update(None, False) is None
    clock.t = 1.2
    assert st.update(None, False) == " "
    clock.t = 5
    assert st.update(None, False) is None  # only one space
    assert st.text == "H "


def test_stabiliser_editing():
    st = LetterStabiliser(clock=FakeClock())
    st.add_space()
    assert st.text == ""  # no leading space
    st.text = "HI"
    st.add_space()
    st.add_space()
    assert st.text == "HI "
    st.backspace()
    assert st.text == "HI"
    st.clear()
    assert st.text == ""


# ---- text and playback ------------------------------------------------------
def test_text_to_tokens():
    assert text_to_tokens("Hi, Talha!") == list("HI") + [" "] + list("TALHA")
    assert text_to_tokens("  room 101  ") == list("ROOM")
    assert text_to_tokens("") == []


def test_wrap_text():
    assert wrap_text("hello there world", 11) == ["hello there", "world"]
    assert wrap_text("abcdefghij", 4) == ["abcd", "efgh", "ij"]
    assert wrap_text("", 5) == [""]


def test_player_is_time_driven():
    clock = FakeClock()
    p = FingerspellPlayer(letter_hold=1.0, space_hold=0.5, clock=clock)
    p.play(["A", " ", "B"])
    assert p.current() == (0, "A")
    clock.t = 1.2
    assert p.current() == (1, " ")
    clock.t = 1.6
    assert p.current() == (2, "B")
    clock.t = 2.6
    assert p.current() is None and not p.playing
