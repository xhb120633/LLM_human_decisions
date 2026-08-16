"""Invertible target transforms for scale-robust Text2Decision training."""

from __future__ import annotations

import numpy as np


MONETARY_INDICES = [0, 1, 2, 3, 4, 10]


def signed_log_monetary(target: np.ndarray) -> np.ndarray:
    """Convert /1000 monetary targets to signed log1p dollar coordinates."""
    transformed = np.asarray(target, dtype=np.float32).copy()
    dollars = transformed[:, MONETARY_INDICES] * 1000.0
    transformed[:, MONETARY_INDICES] = np.sign(dollars) * np.log1p(
        np.abs(dollars)
    )
    return transformed


def inverse_signed_log_monetary(transformed: np.ndarray) -> np.ndarray:
    """Return signed-log targets to the original /1000 monetary units."""
    target = np.asarray(transformed, dtype=np.float32).copy()
    logged = target[:, MONETARY_INDICES]
    dollars = np.sign(logged) * np.expm1(np.abs(logged))
    target[:, MONETARY_INDICES] = dollars / 1000.0
    return target
