"""Entry points for the local LungScope application.

    lungscope --study output/random_field/study.json   lung viewer for a forward-UQ study
    lungscope --mesh examples/random_field/lung.vtu     lung viewer for a single mesh file
    lungscope                                           the starter scene

    lungscope --study ... --lobes examples/random_field_large
        the lung drawn from lobe masks (lobe_*.nii.gz) instead of the illustrative one
    lungscope-fit-lobes examples/random_field_large --mesh examples/random_field_large/lung.vtu
        fit where the lobe masks sit on the mesh and write lobes.json next to them

trame's own options apply as well, e.g. ``--port 8080``, or ``--server`` to open no browser.

The app reads the files through ``data`` and computes the quantities through ``analysis``. It
builds the trame layout and coordinates the selections between the views: the coloring
("Color by"), the time step (time bar and playback) and the clicked element, whose runs a popup
plots. A click opens the popup with the quantity shown, or the pressure while the coloring does
not change over time; the popup follows the time step and the coloring until it is closed.
"""

import argparse
import asyncio
import math
import os
from pathlib import Path

from trame.app import asynchronous, get_server
from trame.ui.vuetify3 import SinglePageLayout, SinglePageWithDrawerLayout
from trame.widgets import html
from trame.widgets import vuetify3 as v3

from lungscope.analysis import fit_lobes, terminal_unit_centres, time_quantities
from lungscope.data import load_lobes, load_mesh, load_study, read_input, save_lobe_transform
from lungscope.views import (
    LungScene,
    add_demo_view,
    add_legend,
    add_run_chart,
    add_scene_view,
    element_title,
    entity_mesh,
    figure_at,
    lobe_surface,
    run_chart,
)

INK = "#52514e"
MUTED = "#898781"
DRAWER_WIDTH = 260
POPUP_WIDTH = 480
CHART_HEIGHT = 340
DEFAULT_SPEED = 5  # time steps per second
MAX_SPEED = 20
FAST = 4  # the fast buttons play this many times faster than the speed slider
MAX_FPS = 20  # playback checks the clock this often; slower drawing skips time steps
PLAY_BUTTONS = (  # play mode, icon, label, mirrored icon
    (-2, "mdi-rewind", "Reverse fast", False),
    (-1, "mdi-play", "Reverse", True),
    (0, "mdi-stop", "Stop", False),
    (1, "mdi-play", "Play", False),
    (2, "mdi-fast-forward", "Play fast", False),
)


def main() -> None:
    """Start the browser application: the lung viewer for --study or --mesh, else the starter scene."""
    server = get_server(client_type="vue3")
    source = server.cli.add_mutually_exclusive_group()
    source.add_argument(
        "--study", metavar="PATH", help="study.json of a forward-UQ study (all completed samples, with time)"
    )
    source.add_argument("--mesh", metavar="PATH", help="a single lung model file (.vtu with line elements)")
    server.cli.add_argument(
        "--lobes",
        metavar="DIR",
        help="folder with lobe masks (lobe_*.nii.gz) and lobes.json, which places them on the mesh; "
        "the lung is drawn from them",
    )
    args, unknown = server.cli.parse_known_args()
    given = args.study if args.study is not None else args.mesh
    if given is None:
        # trame's own options are known to the parser; anything else is a mistake, such as a
        # mistyped --study or a path without it, which must not quietly open the starter scene
        if unknown:
            server.cli.error(f"unrecognized arguments: {' '.join(unknown)}")
        if args.lobes is not None:
            server.cli.error("--lobes needs --study or --mesh")
        build_starter(server)
    else:
        path = Path(given).expanduser().resolve()
        if not path.is_file():
            server.cli.error(f"file not found: {path}")
        lung = None
        if args.lobes is not None:  # before the study, which can take minutes to load
            try:
                lobes = load_lobes(args.lobes)
            except (FileNotFoundError, ValueError) as error:
                server.cli.error(str(error))
            placed = [lobes.mesh_from_index(lobe) for lobe in lobes.lobes]
            lung = lobe_surface([lobe.mask for lobe in lobes.lobes], placed)
        if args.study is not None:
            study = load_study(path)
            scene = LungScene(entity_mesh(study), study["time"].values, time_quantities(study), lung=lung)
            build_viewer(server, scene, f"{path.parent.name} ({study.sizes['realization']} samples)")
        else:
            build_viewer(server, LungScene(load_mesh(path), lung=lung), path.name)
    server.start()


def fit_lobes_main() -> None:
    """lungscope-fit-lobes: fit where lobe masks sit on a lung model's mesh, write lobes.json.

    For masks that come without the transform to the mesh, e.g. the CT lobes of a PneumoGen tree.
    An existing lobes.json, which may hold a known transform, is only replaced with --force.
    """
    parser = argparse.ArgumentParser(
        prog="lungscope-fit-lobes",
        description="Fit the affine transform that puts a lung model's terminal units into its lobe masks "
        "(lobe_*.nii.gz), and write it to lobes.json in the masks' folder for lungscope --lobes.",
    )
    parser.add_argument("lobes", metavar="DIR", help="folder with the lobe masks")
    parser.add_argument(
        "--mesh", metavar="PATH", required=True, help="the lung model (.vtu with line elements, v0 on terminal units)"
    )
    parser.add_argument("--force", action="store_true", help="replace an existing lobes.json")
    args = parser.parse_args()
    mesh = Path(args.mesh).expanduser().resolve()
    if not mesh.is_file():
        parser.error(f"file not found: {mesh}")
    try:
        lobes = load_lobes(args.lobes, placed=False)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    if lobes.mesh_from_world is not None and not args.force:
        parser.error(f"{lobes.folder} already has a lobes.json; --force replaces it")
    endpoints, arrays = read_input(mesh)
    volume = arrays.get("v0", arrays.get("v_0"))
    units = terminal_unit_centres(endpoints, volume) if volume is not None else []
    if len(units) < 4:
        parser.error(f"{mesh}: too few terminal units (elements with v0 > 0) to fit")
    print(f"Fitting {len(units)} terminal units into {len(lobes.lobes)} lobes ...", flush=True)
    fit = fit_lobes(units, [lobe.mask for lobe in lobes.lobes], [lobe.world_from_index for lobe in lobes.lobes])
    record = {
        "method": "lungscope-fit-lobes: affine transform that puts the terminal units into the lobes "
        "(iterative closest points from 48 axis orders)",
        "note": "fitted, not given: replace mesh_from_world with the transform the masks belong to, if known",
        "mesh": os.path.relpath(mesh, lobes.folder),
        "terminal_units": len(units),
        "inside": round(fit.inside, 4),
        "similarity_inside": round(fit.similarity_inside, 4),
        "units_per_lobe": {lobe.name: count for lobe, count in zip(lobes.lobes, fit.per_lobe, strict=True)},
    }
    path = save_lobe_transform(lobes.folder, fit.mesh_from_world, record)
    print(f"{fit.inside:.1%} of the terminal units are inside a lobe (without stretching: {fit.similarity_inside:.1%}).")
    print(f"Wrote {path}")


def build_starter(server) -> SinglePageLayout:
    """The starter page: a generated VTK scene to check the setup without simulation files."""
    with SinglePageLayout(server) as layout:
        layout.title.set_text("LungScope")
        _remove_footer(layout)

        with layout.content, html.Main(classes="pa-6"):
            html.H2("Starter scene", classes="text-h6")
            html.P(
                "Rotate or zoom the generated VTK geometry to check your setup.",
                classes="text-body-2 mb-4",
            )
            add_demo_view()
    return layout


def build_viewer(server, scene: LungScene, title: str) -> SinglePageWithDrawerLayout:
    """The lung viewer page for one scene, and the state that links its views.

    State: the coloring (``color_by``), the time step (``time_index``, with ``play`` and
    ``speed``), the visible parts (``show_units``, ``show_lung``) and the clicked element
    (``pick_element``, None while the popup is closed). Each change redraws the views it
    concerns.
    """
    state = server.state
    times = scene.times
    last = len(times) - 1
    charted = scene.charted
    state.trame__title = "LungScope"
    state.times = [float(t) for t in times]
    state.time_index = 0
    state.play = 0  # -2 reverse fast, -1 reverse, 0 stopped, 1 play, 2 play fast
    state.speed = DEFAULT_SPEED
    state.color_groups = scene.groups
    state.color_by = scene.color_by
    state.legend = scene.legend
    state.has_units = state.show_units = scene.has_units
    state.has_lung = state.show_lung = scene.has_lung
    state.pick_element = None  # the element the popup shows, None while it is closed
    state.pick = None  # the popup's text: title, subtitle, note, readout, missing
    state.pick_chart = {"data": [], "layout": {}}  # its Plotly figure
    state.pick_modes = ["click"] if charted else []
    state.pick_from = [0, 0]  # where the pointer went down in the 3D view
    state.pick_moved = False  # whether it moved from there before it came up (a drag)
    popup = {"key": None, "chart": None}  # the chart shown: for which element and quantity

    @server.trigger("click_element")
    def on_click(event=None, moved=False, *_) -> None:
        """A click in the 3D view that ended no drag: show the element under it in the popup."""
        if moved:
            return
        element = scene.element_at(event)
        if element is not None:
            state.pick_element = element

    def update_popup() -> None:
        """Fill the popup for the selected element and the current time step, or close it."""
        element = state.pick_element
        if element is None or not charted:
            popup["key"] = popup["chart"] = None
            state.pick = None
            return
        element = int(element)
        quantity = charted.get(state.color_by) or next(iter(charted.values()))
        if popup["key"] != (element, quantity.name):
            popup["key"] = (element, quantity.name)
            popup["chart"] = run_chart(
                times,
                quantity.members[:, :, element],
                quantity.member_label,
                quantity.unit,
                revision=f"{element}/{quantity.name}",
            )
        chart, step = popup["chart"], int(state.time_index)
        unit = f" [{quantity.unit}]" if quantity.unit else ""
        state.pick = {
            "title": element_title(scene.mesh, element, bool(scene.units[element])),
            "subtitle": f"{quantity.member_label}{unit} of {len(quantity.members)} runs",
            "note": quantity.member_note,
            "readout": chart.readouts[step] if chart else None,
            "missing": "" if chart else f"The {quantity.member_label} is not defined for this element.",
        }
        if chart:
            state.pick_chart = figure_at(chart, step)

    @state.change("pick_element")
    def on_pick_element(**_):
        update_popup()

    @state.change("color_by")
    def on_color_by(color_by, **_):
        if color_by in scene.options:
            state.legend = scene.show(color_by)
            scene.update_view()
            update_popup()

    @state.change("show_lung")
    def on_show_lung(show_lung, **_):
        scene.set_lung_visible(show_lung)
        scene.update_view()

    @state.change("show_units")
    def on_show_units(show_units, **_):
        scene.set_units_visible(show_units)
        scene.update_view()

    @state.change("time_index")
    def on_time_index(time_index, **_):
        scene.paint(time_index)
        if scene.changes_over_time:
            scene.update_view()
        if state.pick_element is not None:
            update_popup()

    playback = {"task": None}

    @state.change("play")
    def on_play(play, **_):
        if not play or last < 1:
            return
        index = int(state.time_index)
        if play > 0 and index >= last:  # play again from the start (reverse: from the end)
            state.time_index = 0
        elif play < 0 and index <= 0:
            state.time_index = last
        task = playback["task"]
        if task is None or task.done():
            playback["task"] = asynchronous.create_task(run_playback())

    async def run_playback() -> None:
        """Step through the times while a play mode is on; stop at the first or last step.

        The step follows the elapsed time, so the speed holds even when drawing a frame takes
        longer than the time between two steps; steps are skipped then.
        """
        loop = asyncio.get_running_loop()
        current, direction = int(state.time_index), 0
        position, before = float(current), loop.time()
        while state.play:
            await asyncio.sleep(1 / MAX_FPS)
            mode, now = int(state.play), loop.time()
            if not mode:
                break
            if int(state.time_index) != current or (mode > 0) != (direction > 0):
                # moved with the slider, or the direction changed: go on from the shown step
                current = int(state.time_index)
                position = float(current)
            direction = 1 if mode > 0 else -1
            rate = max(1.0, float(state.speed)) * (FAST if abs(mode) == 2 else 1)  # steps per second
            position += direction * rate * (now - before)
            before = now
            index = math.floor(position) if direction > 0 else math.ceil(position)
            if index == current:
                continue
            with state:
                if (direction > 0 and index >= last) or (direction < 0 and index <= 0):
                    index = last if direction > 0 else 0
                    state.play = 0
                state.time_index = current = index

    switch = {"color": "primary", "density": "compact", "hide_details": True, "classes": "mr-6 flex-grow-0"}
    with SinglePageWithDrawerLayout(server, width=DRAWER_WIDTH) as layout:
        layout.title.set_text(f"LungScope · {title}")
        with layout.toolbar:
            v3.VSpacer()
            v3.VSwitch(v_if="has_units", v_model=("show_units",), label="Terminal units", **switch)
            v3.VSwitch(v_if="has_lung", v_model=("show_lung",), label="Lung", **switch)
        with layout.drawer:
            _color_by_list()
        with (
            layout.content,
            v3.VContainer(fluid=True, classes="pa-0 fill-height", style="position: relative"),
        ):
            add_scene_view(scene, on_click)
            add_legend()
            _popup()
        if last >= 1:
            layout.footer.clear()
            with layout.footer:
                _time_bar(len(times))
        else:
            _remove_footer(layout)
    return layout


def _remove_footer(layout) -> None:
    """Remove the footer that trame's page layouts bring along.

    It links to trame and Kitware ("© 2025 Kitware Inc.") and shows trame's busy indicator. The
    footer element stays in the page but is empty, hidden and out of the app layout, which would
    otherwise keep its height free at the bottom of the page.
    """
    layout.footer.clear()
    layout.footer.app = None
    layout.footer.hide()


def _color_by_list() -> None:
    """Side panel: the "Color by" choices as a scrolling list, in groups (see state.color_groups)."""
    with html.Div(classes="d-flex flex-column fill-height"):
        html.Div("Color by", classes="text-subtitle-1 font-weight-medium px-4 pt-3 pb-1")
        with (
            v3.VList(density="compact", nav=True, classes="pt-0", style="overflow-y: auto; flex: 1 1 0"),
            html.Div(v_for="group in color_groups", key="group.title"),
        ):
            v3.VListSubheader("{{ group.title }}")
            v3.VListItem(
                v_for="name in group.items",
                key="name",
                title=("name",),
                active=("color_by === name",),
                click="color_by = name",
                color="primary",
                rounded="lg",
            )
            v3.VListItem(v_if="!group.items.length", title="none yet", disabled=True)


def _time_bar(count: int) -> None:
    """Bottom bar: playback buttons, the time step slider, and the playback speed."""
    with html.Div(classes="d-flex align-center w-100 py-1", style="gap: 2px"):
        for mode, icon, label, mirrored in PLAY_BUTTONS:
            with v3.VBtn(
                icon=True,
                size="small",
                variant=(f"play === {mode} ? 'tonal' : 'text'",),
                color=(f"play === {mode} ? 'primary' : undefined",),
                click=f"play = {mode}",
                raw_attrs=[f'title="{label}"', f'aria-label="{label}"'],
            ):
                v3.VIcon(icon, **({"style": "transform: scaleX(-1)"} if mirrored else {}))
        v3.VSlider(
            v_model=("time_index",),
            min=0,
            max=count - 1,
            step=1,
            color="primary",
            density="compact",
            hide_details=True,
            classes="mx-4 flex-grow-1",
        )
        html.Div(
            "t = {{ times[time_index].toFixed(2) }} s · step {{ time_index + 1 }} / " + str(count),
            classes="text-body-2 text-no-wrap",
            style="min-width: 170px",
        )
        v3.VDivider(vertical=True, classes="mx-3")
        html.Div("Speed", classes="text-body-2 mr-2")
        v3.VSlider(
            v_model=("speed",),
            min=1,
            max=MAX_SPEED,
            step=1,
            color="primary",
            density="compact",
            hide_details=True,
            style="max-width: 140px; min-width: 100px",
        )
        html.Div("{{ speed }} steps/s", classes="text-body-2 text-no-wrap ml-2 mr-2", style="min-width: 72px")


def _popup() -> None:
    """Floating card, top right: every run of the shown quantity at the clicked element.

    It shows state.pick (text) and state.pick_chart (Plotly figure); × closes it.
    """
    with v3.VCard(
        v_if="pick",
        elevation=3,
        style=(
            f"position: absolute; top: 16px; right: 16px; z-index: 2; width: {POPUP_WIDTH}px; "
            "max-width: calc(100% - 32px)"
        ),
    ):
        with html.Div(classes="d-flex align-start pl-4 pr-1 pt-2"):
            with html.Div(classes="flex-grow-1 pt-1"):
                html.Div("{{ pick.title }}", classes="text-subtitle-1 font-weight-medium")
                html.Div("{{ pick.subtitle }}", classes="text-body-2", style=f"color: {INK}")
            v3.VBtn(
                icon="mdi-close",
                size="small",
                variant="text",
                click="pick = null; pick_element = null",
                raw_attrs=['title="Close"', 'aria-label="Close"'],
            )
        html.Div("{{ pick.missing }}", v_if="pick.missing", classes="text-body-2 px-4 pt-4 pb-6")
        with html.Div(v_else=True, classes="px-2 pb-3"):
            add_run_chart("pick_chart", CHART_HEIGHT)
            with html.Div(v_if="pick.readout", classes="d-flex align-start px-2 pt-1", style="gap: 16px"):
                html.Div("{{ pick.readout.time }}", classes="text-body-2 font-weight-medium text-no-wrap")
                with html.Div(v_for="item in pick.readout.items", key="item.label"):
                    html.Div("{{ item.value }}", classes="text-body-2 text-no-wrap")
                    html.Div("{{ item.label }}", classes="text-caption", style=f"color: {MUTED}")
            html.Div("{{ pick.note }}", v_if="pick.note", classes="text-caption px-2 pt-2", style=f"color: {MUTED}")
