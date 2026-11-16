"""Quantities the viewer can show over time, computed from the study dataset.

Each quantity is a value per element and output time, known at both ends of the element; for a
quantity of the whole element both ends hold the same value. NaN marks elements for which the
quantity is not defined. Each quantity also carries every run's value per element (``members``)
for per-element plots. There are no colors here: the views decide how to draw them.
"""

from dataclasses import dataclass

import numpy as np
import xarray as xr

from lungscope.analysis.statistics import ensemble_mean


@dataclass(frozen=True)
class TimeQuantity:
    """A quantity per element and output time, ready to be shown."""

    name: str  # its entry in "Color by"
    label: str  # legend title: the quantity, its unit and what was averaged
    start: np.ndarray  # (time, entity): value at each element's first end, toward the inlet
    end: np.ndarray  # (time, entity): value at its last end
    signed: bool = False  # takes both signs, e.g. a pressure drop that reverses with the flow
    uncertain: bool = False  # describes the spread between the realizations, not a mean
    members: np.ndarray | None = None  # (realization, time, entity): each run's value per element
    member_label: str = ""  # what ``members`` holds, e.g. "pressure"
    member_note: str = ""  # where on an element ``members`` is taken, if that matters
    unit: str = ""


ELEMENT_VALUE_NOTE = "Airways: value at the middle (mean of both ends). Terminal units: at the entrance."


def terminal_units(study: xr.Dataset) -> np.ndarray:
    """Which entities are terminal units: those with a positive reference volume."""
    if "v0" in study.variables and study["v0"].dims == ("entity",):
        v0 = study["v0"].values
    elif "v_0" in study.variables:
        v0 = study["v_0"].isel(realization=0, time=0).values
    else:
        return np.zeros(study.sizes["entity"], dtype=bool)
    v0 = np.asarray(v0, dtype=float)
    return np.isfinite(v0) & (v0 > 0)


def element_values(study: xr.Dataset, start_name: str, end_name: str) -> np.ndarray:
    """Each run's value per element, (realization, time, entity), from values at both ends.

    An airway gets the value at its middle (the mean of both ends); a terminal unit the value at
    its entrance, as in the 3D view.
    """
    start = np.asarray(study[start_name].values, dtype=float)
    end = np.asarray(study[end_name].values, dtype=float)
    values = 0.5 * (start + end)
    units = terminal_units(study)
    values[..., units] = start[..., units]
    return values


def mean_pressure(study: xr.Dataset) -> TimeQuantity:
    """Mean over the realizations of the pressure at both ends of every element."""
    unit = _unit(study, "p_1")
    return TimeQuantity(
        name="pressure (mean)",
        label=f"pressure [{unit}], mean of {study.sizes['realization']} samples",
        start=ensemble_mean(study["p_1"]).values,
        end=ensemble_mean(study["p_2"]).values,
        members=element_values(study, "p_1", "p_2"),
        member_label="pressure",
        member_note=ELEMENT_VALUE_NOTE,
        unit=unit,
    )


def mean_pressure_differential(study: xr.Dataset) -> TimeQuantity:
    """Mean over the realizations of the pressure drop p_1 - p_2 across each airway.

    Not defined for terminal units: their p_1 - p_2 is their whole driving pressure (hundreds of
    Pa against fractions of a Pa in the airways), a different quantity.
    """
    unit = _unit(study, "p_1")
    units = terminal_units(study)
    differential = study["p_1"] - study["p_2"]
    runs = np.array(differential.values, dtype=float)
    runs[..., units] = np.nan
    drop = np.array(ensemble_mean(differential).values, dtype=float)
    drop[:, units] = np.nan
    return TimeQuantity(
        name="pressure differential (mean)",
        label=f"pressure differential p_1 − p_2 [{unit}], mean of {study.sizes['realization']} samples",
        start=drop,
        end=drop,
        signed=True,
        members=runs,
        member_label="pressure differential p_1 − p_2",
        unit=unit,
    )


def pressure_std(study: xr.Dataset) -> TimeQuantity:
    """Standard deviation over the realizations of the pressure at both ends of every element.

    The sample standard deviation (ddof = 1), as QUEENS uses for its variance.
    """
    unit = _unit(study, "p_1")
    return TimeQuantity(
        name="pressure (std)",
        label=f"pressure standard deviation [{unit}], {study.sizes['realization']} samples",
        start=study["p_1"].std("realization", ddof=1).values,
        end=study["p_2"].std("realization", ddof=1).values,
        uncertain=True,
        members=element_values(study, "p_1", "p_2"),
        member_label="pressure",
        member_note=ELEMENT_VALUE_NOTE,
        unit=unit,
    )


def time_quantities(study: xr.Dataset) -> list[TimeQuantity]:
    """Everything the viewer can show over time, in display order."""
    if not {"p_1", "p_2"} <= set(study.data_vars):
        return []
    return [mean_pressure(study), mean_pressure_differential(study), pressure_std(study)]


def _unit(study: xr.Dataset, name: str) -> str:
    return study[name].attrs.get("units", "")
