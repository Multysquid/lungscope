"""Geometry, picking, colors, the scene and the run chart on a tiny synthetic tree, no browser."""

import numpy as np
import pytest
import pyvista as pv

from lungscope.analysis import TimeQuantity
from lungscope.views.coloring import FIELD_ARRAY, UNCERTAINTY_COLORMAP, colormap_colors
from lungscope.views.geometry import build_geometry
from lungscope.views.picking import ElementPicker
from lungscope.views.run_chart import figure_at, run_chart
from lungscope.views.scene import LungScene, _field


@pytest.fixture
def tree() -> pv.UnstructuredGrid:
    """An airway (radius 2) from z = 0 down to z = -10, then a terminal unit down to z = -15."""
    points = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -10.0], [0.0, 0.0, -15.0]])
    cells = np.array([2, 0, 1, 2, 1, 2])
    mesh = pv.UnstructuredGrid(cells, np.full(2, pv.CellType.LINE, dtype=np.uint8), points)
    mesh.cell_data["radius"] = [2.0, 0.0]  # 4C's input gives terminal units a placeholder zero
    mesh.cell_data["v0"] = [0.0, 1000.0]
    return mesh


def test_terminal_units_are_tubes_like_their_airway(tree):
    units = build_geometry(tree).units
    assert set(np.unique(units.cell_data["element_index"])) == {1}
    bounds = np.reshape(units.bounds, (3, 2))  # x, y, z: lowest and highest
    # the radius of the airway it continues, and its own length: no sphere of its volume
    np.testing.assert_allclose(bounds[:2], [[-2.0, 2.0], [-2.0, 2.0]], atol=0.05)
    np.testing.assert_allclose(bounds[2], [-15.0, -10.0], atol=1e-6)


def test_picking_finds_the_first_element_along_the_line(tree):
    geometry = build_geometry(tree)
    picker = ElementPicker({"airways": geometry.airways, "units": geometry.units})
    assert picker.pick([-50, 0, -5], [50, 0, -5]) == 0  # through the airway
    assert picker.pick([-50, 0, -14], [50, 0, -14]) == 1  # through the terminal unit
    assert picker.pick([-50, 0, -14], [50, 0, -14], ["airways"]) is None  # units hidden
    assert picker.pick([0, 0, 50], [0, 0, -50]) == 0  # down the axis: the airway comes first
    assert picker.pick([0, 0, -50], [0, 0, 50]) == 1  # up the axis: the terminal unit first
    assert picker.pick([-50, 9, -5], [50, 9, -5]) is None  # beside the tree


def test_uncertainty_colormap_runs_from_light_to_dark():
    ends = colormap_colors(2, UNCERTAINTY_COLORMAP)  # lowest and highest value, as hex
    light, dark = (sum(int(color[i : i + 2], 16) for i in (1, 3, 5)) for color in ends)
    assert light > dark + 300
    assert light < 3 * 250  # not white: the most certain elements still show


def test_field_ranges(tree):
    start = np.array([[0.0, 0.5], [1.0, 2.0]])
    end = np.array([[0.5, 0.0], [1.5, 900.0]])  # the terminal unit's last end is not drawn
    units = np.array([False, True])
    mean = _field(TimeQuantity("p", "p", start, end), units)
    assert (mean.lo, mean.hi) == (0.0, 2.0)
    spread = _field(TimeQuantity("s", "s", start + 1, end + 1, uncertain=True), units)
    assert (spread.lo, spread.hi, spread.cmap) == (0.0, 3.0, UNCERTAINTY_COLORMAP)  # from zero up


def drawn(actor) -> np.ndarray:
    """The field values an actor draws: those of its mapper's input, which the browser view gets."""
    actor.mapper.GetInputAlgorithm().Update()
    return np.asarray(pv.wrap(actor.mapper.GetInputDataObject(0, 0)).point_data[FIELD_ARRAY])


def test_scene_colors_paints_and_picks(tree):
    members = np.array([[[1.0, 2.0], [3.0, 4.0]], [[1.5, 2.5], [3.5, 4.5]]])  # (run, time, element)
    mean = members.mean(axis=0)
    pressure = TimeQuantity("pressure (mean)", "pressure [Pa]", mean, mean + 1, members=members, unit="Pa")
    scene = LungScene(tree, times=[0.1, 0.2], quantities=[pressure])
    assert scene.options == ["Solid color", "radius", "v0", "pressure (mean)"]
    assert scene.color_by == "Solid color"  # no generation array to start with
    assert [group["items"] for group in scene.groups][1:] == [["pressure (mean)"], []]
    assert scene.has_units

    assert scene.show("pressure (mean)")["title"] == "pressure [Pa]"
    assert scene.changes_over_time
    scene.paint(1)
    (airway, *_), (unit, *_) = scene._painted
    values = drawn(airway)
    assert values.min() == pytest.approx(3.25) and values.max() == pytest.approx(4.25)  # p_1 to p_2
    np.testing.assert_allclose(drawn(unit), 4.25)  # a terminal unit: its entrance value throughout

    click = {"mode": "local", "ray": [[-50, 0, -14], [50, 0, -14]]}  # through the terminal unit
    assert scene.element_at(click) == 1
    scene.set_units_visible(False)
    assert not unit.GetVisibility()
    assert scene.element_at(click) is None
    assert scene.element_at(None) is None

    assert scene.show("radius")["title"] == "radius"
    assert not scene.changes_over_time


def test_scene_without_quantities_has_nothing_to_pick(tree):
    scene = LungScene(tree)
    assert scene.groups == [{"title": "Constant over time", "items": ["Solid color", "radius", "v0"]}]
    assert scene.element_at({"mode": "local", "ray": [[-50, 0, -5], [50, 0, -5]]}) is None


def test_run_chart():
    times = np.array([0.1, 0.2, 0.3])
    runs = np.array([[1.0, 2.0, 3.0], [1.1, 2.2, 3.3], [0.9, 1.8, 2.7]])
    chart = run_chart(times, runs, "pressure", "Pa", revision="7/pressure")
    assert len(chart.figure["data"]) == 8  # per panel: two band edges, the runs, the mean
    assert chart.figure["layout"]["uirevision"] == "7/pressure"
    assert "template" not in chart.figure["layout"]
    runs_trace = chart.figure["data"][2]
    assert runs_trace["y"].count(None) == 3  # the runs are one line with a gap after each run
    assert chart.readouts[1]["time"] == "t = 0.20 s"
    assert [item["label"] for item in chart.readouts[1]["items"]] == ["mean", "std", "min–max"]
    figure = figure_at(chart, 2)
    assert [shape["x0"] for shape in figure["layout"]["shapes"]] == [0.3, 0.3]
    assert "shapes" not in chart.figure["layout"]  # the cached figure stays without a marker
    assert run_chart(times, np.full((3, 3), np.nan), "pressure", "Pa", revision="x") is None
