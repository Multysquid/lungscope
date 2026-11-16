"""The analysis runs on a small synthetic dataset, without any files."""

import numpy as np
import xarray as xr

from lungscope import analysis


def synthetic_study() -> xr.Dataset:
    """3 realizations x 4 times x 3 entities: two airways, then one terminal unit."""
    shape = (3, 4, 3)
    p_1 = np.arange(np.prod(shape), dtype=float).reshape(shape)
    p_2 = p_1 - np.array([0.5, 0.25, 600.0])  # the unit's p_1 - p_2 is its whole driving pressure
    dims = ("realization", "time", "entity")
    study = xr.Dataset(
        {"p_1": (dims, p_1), "p_2": (dims, p_2)},
        coords={
            "realization": [0, 1, 2],
            "time": [0.25, 0.5, 0.75, 1.0],
            "entity": [0, 1, 2],
            "v0": ("entity", [0.0, 0.0, 1000.0]),
        },
    )
    study["p_1"].attrs["units"] = study["p_2"].attrs["units"] = "Pa"
    return study


def test_ensemble_mean_keeps_nan():
    values = xr.DataArray([[1.0, np.nan], [3.0, np.nan]], dims=("realization", "entity"))
    mean = analysis.ensemble_mean(values)
    assert mean.dims == ("entity",)
    assert mean.values[0] == 2.0
    assert np.isnan(mean.values[1])


def test_ranges():
    values = np.array([[-0.2, 0.1], [np.nan, 0.05]])
    assert analysis.value_range(values) == (-0.2, 0.1)
    assert analysis.symmetric_range(values) == (-0.2, 0.2)
    assert analysis.value_range(np.full(3, np.nan)) == (0.0, 0.0)


def test_terminal_units():
    assert analysis.terminal_units(synthetic_study()).tolist() == [False, False, True]


def test_mean_pressure():
    study = synthetic_study()
    quantity = analysis.mean_pressure(study)
    assert quantity.start.shape == (4, 3)
    assert not quantity.signed
    assert not quantity.uncertain
    np.testing.assert_allclose(quantity.start, study["p_1"].mean("realization"))
    np.testing.assert_allclose(quantity.end, study["p_2"].mean("realization"))
    assert "[Pa]" in quantity.label
    assert "3 samples" in quantity.label


def test_mean_pressure_differential():
    quantity = analysis.mean_pressure_differential(synthetic_study())
    assert quantity.signed
    np.testing.assert_allclose(quantity.start[:, :2], np.tile([0.5, 0.25], (4, 1)))
    assert np.isnan(quantity.start[:, 2]).all()  # not defined for the terminal unit
    assert quantity.members.shape == (3, 4, 3)
    np.testing.assert_allclose(quantity.members[..., :2], np.tile([0.5, 0.25], (3, 4, 1)))
    assert np.isnan(quantity.members[..., 2]).all()


def test_element_values():
    study = synthetic_study()
    values = analysis.element_values(study, "p_1", "p_2")
    p_1 = study["p_1"].values
    np.testing.assert_allclose(values[..., :2], p_1[..., :2] - [0.25, 0.125])  # airways: the middle
    np.testing.assert_allclose(values[..., 2], p_1[..., 2])  # terminal unit: the entrance


def test_pressure_std():
    study = synthetic_study()
    quantity = analysis.pressure_std(study)
    assert quantity.uncertain
    assert not quantity.signed
    # the realizations' pressures are 12 apart: sample standard deviation 12 (ddof = 1)
    np.testing.assert_allclose(quantity.start, 12.0)
    np.testing.assert_allclose(quantity.end, 12.0)
    assert "standard deviation [Pa]" in quantity.label
    np.testing.assert_allclose(quantity.members, analysis.element_values(study, "p_1", "p_2"))
    assert quantity.member_label == "pressure"
    assert quantity.unit == "Pa"


def test_run_spread():
    runs = np.array([[1.0, 5.0, np.nan], [3.0, np.nan, np.nan], [2.0, 4.0, np.nan]])
    spread = analysis.run_spread(runs)
    np.testing.assert_allclose(spread.mean[:2], [2.0, 4.5])
    np.testing.assert_allclose(spread.std[:2], [1.0, np.sqrt(0.5)])
    np.testing.assert_allclose(spread.lo[:2], [1.0, 4.0])
    np.testing.assert_allclose(spread.hi[:2], [3.0, 5.0])
    assert spread.lowest.tolist() == [0, 2, -1]  # -1: no run has a value
    assert spread.highest.tolist() == [1, 0, -1]
    assert np.isnan([spread.mean[2], spread.std[2], spread.lo[2], spread.hi[2]]).all()


def test_time_quantities():
    quantities = analysis.time_quantities(synthetic_study())
    names = [quantity.name for quantity in quantities]
    assert names == ["pressure (mean)", "pressure differential (mean)", "pressure (std)"]
    assert [quantity.uncertain for quantity in quantities] == [False, False, True]
    assert all(quantity.members.shape == (3, 4, 3) for quantity in quantities)
    assert analysis.time_quantities(synthetic_study().drop_vars("p_2")) == []
