"""The sign model: your recorded examples plus a k-nearest-neighbours classifier.

Why kNN: it learns from a few dozen samples per letter, trains instantly, needs no
extra libraries, and redoing one letter never disturbs the others.

The model stores raw landmarks. Features (and augmented copies) are rebuilt from
them, so improving the feature recipe never forces a retrain.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .features import FEATURE_DIM, N_LANDMARKS, augment, features_from_raw

FORMAT_VERSION = 2


class LegacyModelError(ValueError):
    """The file was saved by the first version, which stored features only."""


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float  # 0..1 share of weighted neighbour votes
    distance: float  # distance to the closest example of the winning letter
    runner_up: str | None = None


class SignModel:
    def __init__(
        self,
        k: int = 7,
        max_samples_per_label: int = 200,
        augment_copies: int = 3,
        reject_factor: float = 20.0,
        seed: int = 0,
    ) -> None:
        self.k = k
        self.max_samples_per_label = max_samples_per_label
        self.augment_copies = augment_copies
        self.reject_factor = reject_factor
        self._seed = seed
        self._raw: dict[str, np.ndarray] = {}  # label -> (m, 2, 21, 3)
        self._nh: dict[str, np.ndarray] = {}  # label -> (m,) hands per sample
        self._feat: dict[str, np.ndarray] = {}  # label -> features incl. augmented copies
        self._x: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self._spread: float | None = None

    # ---- data -------------------------------------------------------------
    def __len__(self) -> int:
        return sum(len(v) for v in self._nh.values())

    @property
    def labels(self) -> list[str]:
        return sorted(self._raw)

    def counts(self) -> dict[str, int]:
        return {label: len(nh) for label, nh in sorted(self._nh.items())}

    def add(self, label: str, raws: np.ndarray, n_hands: np.ndarray) -> None:
        raws = np.asarray(raws, dtype=np.float32).reshape(-1, 2, N_LANDMARKS, 3)
        n_hands = np.asarray(n_hands, dtype=np.int8).reshape(-1)
        if len(raws) != len(n_hands):
            raise ValueError("raws and n_hands must be the same length")
        if label in self._raw:
            raws = np.concatenate([self._raw[label], raws])
            n_hands = np.concatenate([self._nh[label], n_hands])
        extra = len(raws) - self.max_samples_per_label
        if extra > 0:  # keep the newest examples
            raws, n_hands = raws[extra:], n_hands[extra:]
        self._raw[label], self._nh[label] = raws, n_hands
        self._feat[label] = self._featurise(label, raws, n_hands)
        self._invalidate()

    def remove(self, label: str) -> int:
        removed = len(self._nh.get(label, ()))
        for d in (self._raw, self._nh, self._feat):
            d.pop(label, None)
        self._invalidate()
        return removed

    def clear(self) -> None:
        self._raw.clear(), self._nh.clear(), self._feat.clear()
        self._invalidate()

    def _featurise(self, label: str, raws: np.ndarray, n_hands: np.ndarray) -> np.ndarray:
        # Seeded per label so reloading gives exactly the same model
        rng = np.random.default_rng([self._seed, sum(map(ord, label))])
        rows = []
        for raw, n in zip(raws, n_hands):
            rows.append(features_from_raw(raw, int(n)))
            if self.augment_copies and n:
                for aug in augment(raw, int(n), self.augment_copies, rng):
                    rows.append(features_from_raw(aug, int(n)))
        return np.stack(rows).astype(np.float32) if rows else np.empty((0, FEATURE_DIM))

    def _invalidate(self) -> None:
        self._x = self._y = None
        self._spread = None

    def _matrix(self) -> tuple[np.ndarray, np.ndarray]:
        if self._x is None:
            if self._feat:
                self._x = np.concatenate(list(self._feat.values()))
                self._y = np.concatenate(
                    [np.full(len(f), label, dtype="<U8") for label, f in self._feat.items()]
                )
            else:
                self._x = np.empty((0, FEATURE_DIM), dtype=np.float32)
                self._y = np.empty((0,), dtype="<U8")
            self._x = np.ascontiguousarray(self._x, dtype=np.float32)
            self._sq_norms = np.einsum("ij,ij->i", self._x, self._x)
        return self._x, self._y

    def _typical_spread(self) -> float:
        """How far apart examples of the same letter usually are. Used to spot
        poses that aren't any trained letter."""
        if self._spread is None:
            x, y = self._matrix()
            rng = np.random.default_rng(self._seed)
            dists = []
            for label in self._feat:
                rows = np.flatnonzero(y == label)
                if len(rows) < 3:
                    continue
                for i in rng.choice(rows, size=min(20, len(rows)), replace=False):
                    d = np.linalg.norm(x[rows] - x[i], axis=1)
                    d.sort()
                    dists.append(d[min(self.k, len(d) - 1)])
            self._spread = float(np.median(dists)) if dists else np.inf
        return self._spread

    # ---- inference --------------------------------------------------------
    def predict(self, features: np.ndarray) -> Prediction | None:
        x, y = self._matrix()
        if len(y) == 0:
            return None
        # |x - f|^2 = |x|^2 - 2 x.f + |f|^2, about 5x faster than subtracting every row
        f = np.asarray(features, dtype=np.float32)
        sq = self._sq_norms - 2.0 * (x @ f) + float(f @ f)
        dists = np.sqrt(np.maximum(sq, 0.0))
        k = min(self.k, len(dists))
        nearest = np.argpartition(dists, k - 1)[:k]

        weights = 1.0 / (dists[nearest] + 1e-3)
        votes: dict[str, float] = {}
        closest: dict[str, float] = {}
        for i, w in zip(nearest, weights):
            label = str(y[i])
            votes[label] = votes.get(label, 0.0) + float(w)
            closest[label] = min(closest.get(label, np.inf), float(dists[i]))

        ranked = sorted(votes, key=votes.__getitem__, reverse=True)
        best = ranked[0]
        if closest[best] > self.reject_factor * self._typical_spread():
            return None
        return Prediction(
            best,
            votes[best] / sum(votes.values()),
            closest[best],
            ranked[1] if len(ranked) > 1 else None,
        )

    # ---- persistence ------------------------------------------------------
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        labels = self.labels
        raw = (np.concatenate([self._raw[l] for l in labels]) if labels
               else np.empty((0, 2, N_LANDMARKS, 3), np.float32))
        nh = np.concatenate([self._nh[l] for l in labels]) if labels else np.empty(0, np.int8)
        y = np.concatenate([np.full(len(self._nh[l]), l, "<U8") for l in labels]) if labels \
            else np.empty(0, "<U8")
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez_compressed(tmp, version=FORMAT_VERSION, raw=raw, n_hands=nh, y=y)
        tmp.replace(path)  # atomic, so a crash mid-save never corrupts the model

    @classmethod
    def load(cls, path: Path, **kwargs) -> "SignModel":
        model = cls(**kwargs)
        path = Path(path)
        if not path.exists():
            return model
        with np.load(path) as data:
            if "raw" not in data.files:
                raise LegacyModelError(f"{path} was saved by the old version")
            raw, nh, y = data["raw"], data["n_hands"], data["y"]
        for label in np.unique(y):
            mask = y == label
            model.add(str(label), raw[mask], nh[mask])
        return model
