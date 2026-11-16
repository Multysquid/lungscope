"""Statistical analyses that operate independently of the application UI.

Pure functions on the study dataset from ``data.load_study``. Inputs and outputs are xarray
objects, numpy arrays or plain numbers. Nothing here opens files or draws, and nothing imports
``data`` or ``views``.

Public interface
- ``ensemble_mean(values)``: mean over the realizations; NaN ("not defined") stays NaN.
- ``run_spread(runs)``: ``Spread`` over the runs (first axis): mean, standard deviation, lowest
  and highest value, and which run has them.
- ``value_range(*arrays)``: smallest and largest finite value.
- ``symmetric_range(*arrays)``: a range centered on zero that covers every finite value.
- ``terminal_units(study)``: which entities are terminal units (positive reference volume).
- ``element_values(study, start_name, end_name)``: each run's value per element (realization,
  time, entity): the middle of an airway, the entrance of a terminal unit.
- ``TimeQuantity``: a quantity per element and output time, known at both element ends.
  ``start`` and ``end`` are arrays of shape (time, entity), NaN where the quantity is not
  defined; ``signed`` says it takes both signs, ``uncertain`` that it is a spread, not a mean.
  ``members`` holds every run's value per element (realization, time, entity), described by
  ``member_label``, ``member_note`` (where on the element) and ``unit``.
- ``mean_pressure(study)``, ``mean_pressure_differential(study)``: means over the realizations.
- ``pressure_std(study)``: standard deviation of the pressure over the realizations (ddof = 1).
- ``time_quantities(study)``: all quantities that can be shown over time, in display order.
"""

from lungscope.analysis.quantities import (
    TimeQuantity,
    element_values,
    mean_pressure,
    mean_pressure_differential,
    pressure_std,
    terminal_units,
    time_quantities,
)
from lungscope.analysis.statistics import (
    Spread,
    ensemble_mean,
    run_spread,
    symmetric_range,
    value_range,
)

__all__ = [
    "Spread",
    "TimeQuantity",
    "element_values",
    "ensemble_mean",
    "mean_pressure",
    "mean_pressure_differential",
    "pressure_std",
    "run_spread",
    "symmetric_range",
    "terminal_units",
    "time_quantities",
    "value_range",
]
