"""Tests for the parts that don't need a camera or microphone. Run: python -m pytest"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from signspeak.recognition.classifier import LegacyModelError, SignModel
from signspeak.recognition.features import (
    CROSS_DIM,
    FEATURE_DIM,
    HAND_DIM,
    augment,
    build_features,
    hands_to_raw,
)
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
    np.testing.assert_allclose(fa, fb, atol=1e-4)


def test_hands_ordered_by_position_not_label():
    """MediaPipe's left/right labels flip when hands touch; order must not."""
    r, l = make_hand("Right", 1, (0.6, 0.5)), make_hand("Left", 2, (0.3, 0.5))
    flipped_r, flipped_l = FakeHand("Left", r.points), FakeHand("Right", l.points)
    f1 = build_features([r, l])
    np.testing.assert_allclose(f1, build_features([l, r]))
    np.testing.assert_allclose(f1, build_features([flipped_r, flipped_l]))
    assert f1[-1] > 0 and build_features([l])[-1] == 0


def test_cross_distances_see_contact():
    left, right = make_hand("Left", 2, (0.3, 0.5)), make_hand("Right", 1, (0.6, 0.5))
    near = FakeHand("Right", right.points.copy())
    near.points[:, 0] -= 0.25
    o = 2 * HAND_DIM
    assert build_features([left, near])[o:o + CROSS_DIM].min() < \
        build_features([left, right])[o:o + CROSS_DIM].min()


def test_no_hands_gives_none():
    assert build_features([]) is None


def test_augment_keeps_shape_and_hand_count():
    raw, n = hands_to_raw([make_hand("Right", 1)])
    out = augment(raw, n, 5, np.random.default_rng(0))
    assert out.shape == (5, 2, 21, 3) and np.all(out[:, 1] == 0)
    assert not np.allclose(out[0], raw)


# ---- model ----------------------------------------------------------------
def _poses(n_letters=8):
    letters = []
    for i in range(n_letters):
        two = i % 2 == 0
        hands = [make_hand("Right", 100 + i, (0.6, 0.5))]
        if two:
            hands.append(make_hand("Left", 200 + i, (0.4, 0.5)))
        letters.append((chr(65 + i), hands))
    return letters


def _jitter(hands, rng, noise, shift, scale, angle):
    out = []
    for h in hands:
        p = h.points.copy()
        c = p[:, :2].mean(axis=0)
        ca, sa = np.cos(angle), np.sin(angle)
        xy = (p[:, :2] - c) @ np.array([[ca, sa], [-sa, ca]]) * scale + c + shift
        p[:, :2] = xy + rng.normal(0, noise, xy.shape)
        out.append(FakeHand(h.label, p))
    return out


def _record(model, letters, rng, n=40):
    for label, hands in letters:
        raws, nh = zip(*(hands_to_raw(_jitter(hands, rng, 0.002, 0, 1, 0)) for _ in range(n)))
        model.add(label, np.stack(raws), np.array(nh))


def test_model_recognises_new_takes_of_trained_letters():
    rng = np.random.default_rng(1)
    letters = _poses()
    model = SignModel()
    _record(model, letters, rng)
    correct = total = 0
    for label, hands in letters:
        for _ in range(30):  # a different day: moved, closer, slightly tilted, shakier
            test = _jitter(hands, rng, 0.004, rng.uniform(-0.1, 0.1, 2),
                           rng.uniform(0.8, 1.25), rng.uniform(-0.1, 0.1))
            pred = model.predict(build_features(test))
            correct += pred is not None and pred.label == label
            total += 1
    assert correct / total > 0.95


def test_model_round_trips_and_caps(tmp_path):
    rng = np.random.default_rng(0)
    model = SignModel(max_samples_per_label=30)
    _record(model, _poses(3), rng, n=40)
    assert model.counts() == {"A": 30, "B": 30, "C": 30}
    path = tmp_path / "model.npz"
    model.save(path)
    loaded = SignModel.load(path, max_samples_per_label=30)
    assert loaded.counts() == model.counts()
    feats = build_features(_poses(3)[1][1])
    assert loaded.predict(feats).label == model.predict(feats).label == "B"
    assert loaded.remove("A") == 30 and loaded.labels == ["B", "C"]


def test_old_model_file_is_detected(tmp_path):
    path = tmp_path / "model.npz"
    np.savez(path, x=np.zeros((1, 5)), y=np.array(["A"]))
    with pytest.raises(LegacyModelError):
        SignModel.load(path)


def test_empty_model_predicts_nothing():
    assert SignModel().predict(np.zeros(FEATURE_DIM, np.float32)) is None


# ---- stabiliser -----------------------------------------------------------
def test_letter_typed_once_per_hold():
    st = LetterStabiliser(window=5, need=3, clock=FakeClock())
    typed = [st.update("A", True) for _ in range(20)]
    assert typed.count("A") == 1 and st.text == "A"


def test_odd_wrong_frames_do_not_block_typing():
    st = LetterStabiliser(window=10, need=7, clock=FakeClock())
    for label in ["B", "B", "R", "B", "B", None, "B", "B", "B"]:
        st.update(label, True)
    assert st.text == "B"


def test_flicker_does_not_retype_but_release_does():
    st = LetterStabiliser(window=10, need=7, release=5, clock=FakeClock())
    for _ in range(8):
        st.update("L", True)
    st.update(None, True)  # one-frame flicker
    for _ in range(5):
        st.update("L", True)
    assert st.text == "L"
    for _ in range(5):
        st.update("O", True)  # changed hands for a moment
    for _ in range(9):
        st.update("L", True)
    assert st.text == "LL"


def test_changing_letter_does_not_retype_previous_one():
    st = LetterStabiliser(window=10, need=7, release=5, clock=FakeClock())
    for label in ["A"] * 12 + ["B"] * 12:
        st.update(label, True)
    assert st.text == "AB"


def test_dropping_hands_allows_double_letters():
    st = LetterStabiliser(window=5, need=3, space_after=10, clock=FakeClock())
    for _ in range(4):
        st.update("L", True)
    st.update(None, False)
    for _ in range(4):
        st.update("L", True)
    assert st.text == "LL"


def test_space_after_hands_gone():
    clock = FakeClock()
    st = LetterStabiliser(window=2, need=2, space_after=1.0, clock=clock)
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
