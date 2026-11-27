"""The app's pages and the handlers that link their views, without a browser.

``server.state.ready()`` runs every change handler once, as ``server.start()`` does, and
``send`` applies a change the way one arrives from the browser.
"""

import asyncio
import re

import numpy as np
import pytest
import pyvista as pv
import xarray as xr
from trame.app import get_server

from lungscope.analysis import time_quantities
from lungscope.app import build_starter, build_viewer, main
from lungscope.views import LungScene, entity_mesh, figure_at, run_chart
from lungscope.views.coloring import FIELD_ARRAY, end_field_values

THROUGH_AIRWAY_1 = {"mode": "local", "ray": [[2.5, -50, -12.5], [2.5, 50, -12.5]]}  # (0,0,-10)-(5,0,-15)
THROUGH_UNIT_3 = {"mode": "local", "ray": [[6.5, -50, -16.5], [6.5, 50, -16.5]]}  # (5,0,-15)-(7,0,-17)


def tiny_study() -> xr.Dataset:
    """2 runs x 3 times, like data.load_study gives: a trachea splitting into two airways, each
    ending in a terminal unit."""
    ends = np.array(
        [
            [[0, 0, 0], [0, 0, -10]],
            [[0, 0, -10], [5, 0, -15]],
            [[0, 0, -10], [-5, 0, -16]],
            [[5, 0, -15], [7, 0, -17]],
            [[-5, 0, -16], [-7, 0, -18]],
        ],
        dtype=float,
    )
    rng = np.random.default_rng(0)
    p_1 = 100 + rng.normal(0, 0.01, (2, 3, 5))
    dims = ("realization", "time", "entity")
    study = xr.Dataset(
        {"p_1": (dims, p_1), "p_2": (dims, p_1 - 0.1)},
        coords={
            "realization": [0, 1],
            "time": [0.1, 0.2, 0.3],
            "entity": np.arange(5),
            "endpoints": (("entity", "end", "xyz"), ends),
            "block_id": ("entity", [1, 1, 1, 2, 2]),
            "generation": ("entity", [0, 1, 1, -1, -1]),
            "radius": ("entity", [2.0, 1.5, 1.4, 0.0, 0.0]),
            "v0": ("entity", [0.0, 0.0, 0.0, 1000.0, 1000.0]),
        },
    )
    study["p_1"].attrs["units"] = study["p_2"].attrs["units"] = "Pa"
    return study


def viewer(name: str, started: bool = True):
    """The viewer page of tiny_study on its own server, and its scene."""
    server = get_server(name, client_type="vue3")
    study = tiny_study()
    scene = LungScene(entity_mesh(study), study["time"].values, time_quantities(study))
    layout = build_viewer(server, scene, "tiny (2 samples)")
    if started:
        server.state.ready()
    return server, scene, layout


def send(state, **changes):
    """Apply a change as it comes from the browser."""
    state.update(changes)
    state.flush()


def click(server, event, moved=False):
    """A click in the 3D view, as the view reports it."""
    server.controller.trigger_fn("click_element")(event, moved)
    server.state.flush()


def drawn(scene) -> list[np.ndarray]:
    """The field values each colored part draws (its mapper's input, which the browser gets)."""
    values = []
    for actor, *_ in scene._painted:
        actor.mapper.GetInputAlgorithm().Update()
        values.append(np.asarray(pv.wrap(actor.mapper.GetInputDataObject(0, 0)).point_data[FIELD_ARRAY]))
    return values


def test_viewer_page():
    server, _, layout = viewer("test_viewer_page", started=False)
    state = server.state
    assert state.color_by == "generation"
    assert state.legend["title"] == "generation"
    assert [group["title"] for group in state.color_groups] == [
        "Constant over time",
        "Changing over time",
        "Uncertain and changing over time",
    ]
    assert state.color_groups[2]["items"] == ["pressure (std)"]
    assert state.pick_modes == ["click"]
    assert state.pick is None
    page = layout.html
    for part in ("Color by", "Terminal units", "pick_chart", "t = {{ times[time_index].toFixed(2) }} s"):
        assert part in page
    assert "vtk-remote-local-view" in page  # the 3D view
    view_id = re.search(r'viewId="([^"]+)"', page).group(1)
    assert view_id in state  # its state lives on this page's server, not on trame's default one


def test_viewer_page_for_a_mesh_has_no_time_bar_and_no_picking():
    server = get_server("test_viewer_mesh", client_type="vue3")
    scene = LungScene(entity_mesh(tiny_study()))
    layout = build_viewer(server, scene, "lung.vtu")
    assert server.state.pick_modes == []
    assert [group["title"] for group in server.state.color_groups] == ["Constant over time"]
    assert "time_index" not in layout.html


def test_starter_page():
    server = get_server("test_starter_page", client_type="vue3")
    page = build_starter(server).html
    assert "Starter scene" in page


def test_no_page_keeps_trames_footer():
    """trame's page layouts bring a footer with links to trame and Kitware; no page shows it.
    Without a time bar the footer is hidden and out of the app layout, so it keeps no space free."""
    starter = build_starter(get_server("test_footer_starter", client_type="vue3"))
    mesh_scene = LungScene(entity_mesh(tiny_study()))
    mesh = build_viewer(get_server("test_footer_mesh", client_type="vue3"), mesh_scene, "lung.vtu")
    _, _, study = viewer("test_footer_study", started=False)
    for layout in (starter, mesh, study):
        assert "Kitware" not in layout.html
        assert "Powered by trame" not in layout.html
    for layout in (starter, mesh):
        footer = layout.footer.html
        assert 'style="display: none"' in footer
        assert " app" not in footer
    assert "time_index" in study.footer.html  # the study's footer is its time bar


def test_popup_follows_the_click_the_time_step_and_the_coloring():
    server, scene, _ = viewer("test_popup")
    state = server.state
    click(server, THROUGH_AIRWAY_1, moved=True)  # the end of a drag that rotated the view
    assert state.pick is None
    click(server, THROUGH_AIRWAY_1)
    assert state.pick_element == 1
    assert state.pick["title"] == "Airway 1 · generation 1"
    assert state.pick["subtitle"] == "pressure [Pa] of 2 runs"  # the coloring is constant over time
    pressure = scene.charted["pressure (mean)"]
    chart = run_chart(scene.times, pressure.members[:, :, 1], "pressure", "Pa", revision="x")
    assert [trace["y"] for trace in state.pick_chart["data"]] == [trace["y"] for trace in figure_at(chart, 0)["data"]]
    assert state.pick["readout"]["time"] == "t = 0.10 s"

    send(state, time_index=2)
    assert state.pick["readout"]["time"] == "t = 0.30 s"
    assert {shape["x0"] for shape in state.pick_chart["layout"]["shapes"]} == {0.3}

    send(state, color_by="pressure differential (mean)")
    assert state.legend["title"].startswith("pressure differential p_1 − p_2 [Pa]")
    assert state.pick["subtitle"] == "pressure differential p_1 − p_2 [Pa] of 2 runs"
    click(server, THROUGH_UNIT_3)
    assert state.pick["title"] == "Terminal unit 3"
    assert state.pick["missing"]  # no pressure drop across a terminal unit

    send(state, pick=None, pick_element=None)  # the × button
    assert state.pick is None


def test_time_step_recolors_and_switches_hide():
    server, scene, _ = viewer("test_paint")
    state = server.state
    send(state, color_by="pressure (mean)")
    send(state, time_index=2)
    for got, (_, _, element, position) in zip(drawn(scene), scene._painted, strict=True):
        np.testing.assert_allclose(got, end_field_values(scene.fields["pressure (mean)"], 2, element, position))

    send(state, show_units=False)
    assert not any(actor.GetVisibility() for actor in scene._actors["units"])
    click(server, THROUGH_UNIT_3)  # hidden terminal units cannot be picked
    assert state.pick_element is None
    assert scene.has_lung
    send(state, show_lung=False)
    assert not scene.lung_actor.GetVisibility()


def test_playback_runs_to_the_last_step_and_stops():
    server, _, _ = viewer("test_play")
    state = server.state

    async def play():
        send(state, time_index=1)
        send(state, play=1)
        for _ in range(40):
            await asyncio.sleep(0.05)
            if not state.play:
                break
        return state.time_index, state.play

    assert asyncio.run(play()) == (2, 0)


def test_mistyped_options_do_not_open_the_starter_scene(monkeypatch, capsys):
    # Under pytest, trame's parser reads its arguments from TRAME_ARGS instead of sys.argv. Should
    # the starter scene open anyway, --server and --timeout 1 make it stop at once.
    monkeypatch.setenv("TRAME_ARGS", "--stdy output/random_field/study.json --server --timeout 1 --port 0")
    with pytest.raises(SystemExit) as stop:
        main()
    assert stop.value.code == 2
    assert "unrecognized arguments: --stdy output/random_field/study.json" in capsys.readouterr().err
