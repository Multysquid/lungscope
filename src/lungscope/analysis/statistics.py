"""Ensemble statistics and value ranges: pure functions on xarray/numpy data."""

import warnings
from typing import NamedTuple

import numpy as np
import xarray as xr


class Spread(NamedTuple):
    """Statistics over the runs, e.g. per time step; NaN (or -1) where no run has a value."""

    mean: np.ndarray
    std: np.ndarray  # sample standard deviation (ddof = 1), as in "pressure (std)"
    lo: np.ndarray  # lowest value
    hi: np.ndarray  # highest value
    lowest: np.ndarray  # which run has the lowest value
    highest: np.ndarray  # which run has the highest value


def ensemble_mean(values: xr.DataArray) -> xr.DataArray:
    """Mean over the realizations (samples); NaN stays NaN where a value is not defined."""
    return values.mean("realization")


def run_spread(runs: np.ndarray) -> Spread:
    """Mean, standard deviation, lowest and highest value over the runs (the first axis).

    NaN values (a run without a value) are left out.
    """
    runs = np.asarray(runs, dtype=float)
    defined = np.isfinite(runs).any(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # no values, or a single run for the std
        mean, std = np.nanmean(runs, axis=0), np.nanstd(runs, axis=0, ddof=1)
        lo, hi = np.nanmin(runs, axis=0), np.nanmax(runs, axis=0)
    filled = np.where(defined, runs, 0.0)  # nanargmin/nanargmax refuse slices without values
    lowest = np.where(defined, np.nanargmin(filled, axis=0), -1)
    highest = np.where(defined, np.nanargmax(filled, axis=0), -1)
    return Spread(mean, std, lo, hi, lowest, highest)


def symmetric_range(*arrays: xr.DataArray | np.ndarray) -> tuple[float, float]:
    """A range centered on zero that covers every finite value, e.g. for a diverging colormap."""
    lo, hi = value_range(*arrays)
    largest = max(abs(lo), abs(hi))
    return -largest, largest


def value_range(*arrays: xr.DataArray | np.ndarray) -> tuple[float, float]:
    """Smallest and largest finite value over all given arrays, e.g. for a fixed color range."""
    finite = [np.asarray(a, dtype=float) for a in arrays]
    finite = np.concatenate([a[np.isfinite(a)] for a in finite])
    if finite.size == 0:
        return 0.0, 0.0
    return float(finite.min()), float(finite.max())
