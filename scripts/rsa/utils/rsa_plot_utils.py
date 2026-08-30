"""Shared display-only helpers for saved RSA time courses."""
from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator


DISPLAY_ONLY_NOTE = (
    "Curves are PCHIP-smoothed for visualization only; significance is based "
    "on the original 50 time bins."
)


def pchip_mean_sem(times, mean, sem, *, enabled=True, n_points=500):
    """Densify mean and mean±SEM for display without touching statistics."""
    times = np.asarray(times, dtype=float)
    mean = np.asarray(mean, dtype=float)
    sem = np.asarray(sem, dtype=float)
    if times.ndim != 1 or mean.shape != times.shape or sem.shape != times.shape:
        raise ValueError("times, mean, and sem must be matching 1D arrays")
    if len(times) != 50:
        raise ValueError(f"Expected the original 50 time bins, got {len(times)}")
    if not (np.isfinite(times).all() and np.isfinite(mean).all() and np.isfinite(sem).all()):
        raise ValueError("Non-finite curve values cannot be displayed")
    if not enabled:
        return times, mean, mean - sem, mean + sem
    if n_points < len(times):
        raise ValueError("PCHIP display density cannot be below the original bin count")
    dense_times = np.linspace(times[0], times[-1], int(n_points))
    dense_mean = PchipInterpolator(times, mean)(dense_times)
    dense_lower = PchipInterpolator(times, mean - sem)(dense_times)
    dense_upper = PchipInterpolator(times, mean + sem)(dense_times)
    # Numerical crossings are not expected, but ordering is enforced for fill_between.
    lower = np.minimum(dense_lower, dense_upper)
    upper = np.maximum(dense_lower, dense_upper)
    return dense_times, dense_mean, lower, upper


def contiguous_segments(mask):
    """Return inclusive index bounds from an original-bin boolean mask."""
    mask = np.asarray(mask, dtype=bool)
    padded = np.r_[False, mask, False]
    starts = np.flatnonzero(np.diff(padded.astype(int)) == 1)
    ends = np.flatnonzero(np.diff(padded.astype(int)) == -1) - 1
    return list(zip(starts, ends))
