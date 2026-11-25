"""Frozen train/test splits stored as row identifiers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class Split:
    train_ids: np.ndarray
    test_ids: np.ndarray
    seed: int
    kind: str


def make_split(y, kind: str, seed: int = 0, test_size: float = 0.2, official: np.ndarray | None = None) -> Split:
    row_ids = np.arange(len(y))
    if kind == "official":
        if official is None:
            raise ValueError("This dataset has no official split column.")
        train_ids, test_ids = row_ids[official == "train"], row_ids[official == "test"]
    elif kind == "stratified":
        train_ids, test_ids = train_test_split(row_ids, test_size=test_size, stratify=np.asarray(y), random_state=seed)
    else:
        raise ValueError(f"Unknown split kind: {kind}")
    return Split(np.sort(train_ids), np.sort(test_ids), seed, kind)


def stratified_sample(y: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    """Positions of a fixed explanation sample with up to ``per_class`` rows of every class."""
    rng = np.random.default_rng(seed)
    picks = []
    for label in np.unique(y):
        positions = np.flatnonzero(y == label)
        take = min(per_class, len(positions))
        picks.append(rng.choice(positions, size=take, replace=False))
    return np.sort(np.concatenate(picks))
