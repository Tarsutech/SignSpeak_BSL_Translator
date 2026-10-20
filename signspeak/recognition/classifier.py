"""A small k-nearest-neighbours classifier trained on your own signing.

Why kNN: it learns from a few dozen samples per letter, trains instantly, needs no
extra libraries, and adding or redoing one letter never disturbs the others.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .features import FEATURE_DIM


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float  # 0..1 share of weighted neighbour votes
    distance: float  # distance to the closest sample of the winning label


class KNNClassifier:
    def __init__(self, k: int = 5, max_samples_per_label: int = 300, seed: int = 0) -> None:
        self.k = k
        self.max_samples_per_label = max_samples_per_label
        self._x = np.empty((0, FEATURE_DIM), dtype=np.float32)
        self._y = np.empty((0,), dtype="<U8")
        self._rng = np.random.default_rng(seed)

    # ---- data -------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._y)

    @property
    def labels(self) -> list[str]:
        return sorted(set(self._y.tolist()))

    def counts(self) -> dict[str, int]:
        labels, counts = np.unique(self._y, return_counts=True)
        return {str(l): int(c) for l, c in zip(labels, counts)}

    def add(self, label: str, samples: np.ndarray) -> None:
        samples = np.atleast_2d(np.asarray(samples, dtype=np.float32))
        if samples.shape[1] != FEATURE_DIM:
            raise ValueError(f"Expected {FEATURE_DIM} features, got {samples.shape[1]}")
        self._x = np.vstack([self._x, samples])
        self._y = np.concatenate([self._y, np.full(len(samples), label, dtype="<U8")])
        self._cap(label)

    def _cap(self, label: str) -> None:
        idx = np.flatnonzero(self._y == label)
        extra = len(idx) - self.max_samples_per_label
        if extra <= 0:
            return
        drop = self._rng.choice(idx, size=extra, replace=False)
        keep = np.ones(len(self._y), dtype=bool)
        keep[drop] = False
        self._x, self._y = self._x[keep], self._y[keep]

    def remove(self, label: str) -> int:
        keep = self._y != label
        removed = int((~keep).sum())
        self._x, self._y = self._x[keep], self._y[keep]
        return removed

    def clear(self) -> None:
        self._x = self._x[:0]
        self._y = self._y[:0]

    # ---- inference --------------------------------------------------------
    def predict(self, features: np.ndarray) -> Prediction | None:
        if len(self._y) == 0:
            return None
        dists = np.linalg.norm(self._x - features, axis=1)
        k = min(self.k, len(dists))
        nearest = np.argpartition(dists, k - 1)[:k]

        weights = 1.0 / (dists[nearest] + 1e-6)
        votes: dict[str, float] = {}
        closest: dict[str, float] = {}
        for i, w in zip(nearest, weights):
            label = str(self._y[i])
            votes[label] = votes.get(label, 0.0) + float(w)
            closest[label] = min(closest.get(label, np.inf), float(dists[i]))

        best = max(votes, key=votes.__getitem__)
        return Prediction(best, votes[best] / sum(votes.values()), closest[best])

    # ---- persistence ------------------------------------------------------
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez_compressed(tmp, x=self._x, y=self._y)
        tmp.replace(path)  # atomic, so a crash mid-save never corrupts the model

    @classmethod
    def load(cls, path: Path, k: int = 5, max_samples_per_label: int = 300) -> "KNNClassifier":
        model = cls(k=k, max_samples_per_label=max_samples_per_label)
        path = Path(path)
        if not path.exists():
            return model
        with np.load(path) as data:
            x, y = data["x"], data["y"]
        if x.ndim != 2 or x.shape[1] != FEATURE_DIM:
            raise ValueError(
                f"{path} was made by an older version (feature size {x.shape[-1]}). "
                "Delete it and retrain."
            )
        model._x = x.astype(np.float32)
        model._y = y.astype("<U8")
        return model
