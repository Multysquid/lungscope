"""The 3D scene of a lung model: airway tree, terminal units and the lung surface around them.

``LungScene`` holds the PyVista plotter and draws the scene for the selections the app hands
it: the coloring (``show``), the time step (``paint``), which parts are visible, and which
element a click hits (``element_at``). ``add_scene_view`` puts the scene into the page.
"""

from collections.abc import Sequence

import numpy as np
import pyvista as pv
from trame.widgets import html
from trame_pyvista.ui import get_viewer, plotter_ui

from lungscope.analysis import TimeQuantity, symmetric_range, value_range
from lungscope.views.coloring import (
    COLORMAP,
    DIVERGING_COLORMAP,
    FIELD_ARRAY,
    UNCERTAINTY_COLORMAP,
    EndField,
    color_legend,
    color_options,
    color_parts,
    color_spec,
    element_positions,
    end_field_values,
    field_legend,
    field_style,
)
from lungscope.views.geometry import build_geometry, unit_mask
from lungscope.views.lung_shape import (
    LUNG_COLOR,
    LUNG_OPACITY,
    front_view,
    lung_surface,
)
from lungscope.views.picking import ElementPicker, click_ray

BACKGROUND = "#fcfcfb"
DEFAULT_COLOR_BY = "generation"
CONSTANT, CHANGING, UNCERTAIN = "Constant over time", "Changing over time", "Uncertain and changing over time"
DRAG = 5  # pixels: a pointer that moved further between press and release rotated the view


class LungScene:
    """The 3D scene of one lung model, and the quantities it can show over time.

    ``mesh`` is the line mesh to draw (``entity_mesh`` of a study, or a mesh file). ``times``
    (output times in s) and ``quantities`` (values per element and time, in entity order) come
    from a study; quantities are only offered with at least two output times. ``lung`` is the
    lung surface around the tree in mesh coordinates, e.g. ``lobe_surface`` of lobe masks;
    without it, the illustrative lung of ``lung_surface`` is drawn. The scene starts colored by
    DEFAULT_COLOR_BY (or the first choice), seen from the front.
    """

    def __init__(
        self,
        mesh: pv.UnstructuredGrid,
        times: Sequence[float] | np.ndarray | None = None,
        quantities: Sequence[TimeQuantity] = (),
        lung: pv.PolyData | None = None,
    ) -> None:
        self.mesh = mesh
        self.times = np.zeros(0) if times is None else np.asarray(times, dtype=float)
        self.units = unit_mask(mesh)  # which elements are terminal units
        over_time = len(self.times) > 1
        self.fields = {q.name: _field(q, self.units) for q in quantities} if over_time else {}
        # the quantities that carry every run's values, for the plot of one element
        self.charted = {q.name: q for q in quantities if q.members is not None} if over_time else {}
        constant = color_options(mesh)
        self.options = [*constant, *self.fields]
        self.groups = [{"title": CONSTANT, "items": constant}]  # the "Color by" choices, by group
        if over_time:
            self.groups += [
                {"title": CHANGING, "items": [q.name for q in quantities if not q.uncertain]},
                {"title": UNCERTAIN, "items": [q.name for q in quantities if q.uncertain]},
            ]
        geometry = build_geometry(mesh)
        self.pieces = (("airways", geometry.airways), ("units", geometry.units))
        self.has_units = geometry.units is not None
        self.units_visible = self.has_units
        self.picker = ElementPicker(dict(self.pieces)) if self.charted else None

        self.plotter = pv.Plotter(off_screen=True)
        self.plotter.set_background(BACKGROUND)
        # No depth peeling: on the CPU renderer (llvmpipe) it drops translucent surfaces such as the
        # lung entirely, and the light, uniform lung surfaces do not need it.
        self._actors = {"airways": [], "units": []}
        self._painted = []  # parts colored by a field: (actor, drawn mesh, element and position per point)
        self._field = None
        self._step = 0
        self.color_by = DEFAULT_COLOR_BY if DEFAULT_COLOR_BY in self.options else self.options[0]
        self.legend = self.show(self.color_by)
        self.lung_actor = None
        if lung is None:
            lung = lung_surface(mesh)
        if lung is not None:
            self.lung_actor = self.plotter.add_mesh(
                lung, color=LUNG_COLOR, opacity=LUNG_OPACITY, smooth_shading=True, show_scalar_bar=False
            )
            self.lung_actor.SetPickable(False)  # clicks go through the lung to the airways inside
        self.has_lung = self.lung_actor is not None
        self._front_camera()

    @property
    def changes_over_time(self) -> bool:
        """Whether the shown coloring changes with the time step."""
        return self._field is not None

    def show(self, name: str) -> dict | None:
        """Color the airways and terminal units by one "Color by" choice; returns its legend.

        A quantity over time is drawn at the current time step (see ``paint``).
        """
        for group in self._actors.values():
            for actor in group:
                self.plotter.remove_actor(actor, render=False)
            group.clear()
        self._painted.clear()
        self.color_by = name
        self._field = self.fields.get(name)
        if self._field is not None:
            for group, piece in self.pieces:
                if piece is None or piece.n_cells == 0:
                    continue
                part = piece.copy()
                part.point_data[FIELD_ARRAY] = np.zeros(part.n_points)
                actor = self.plotter.add_mesh(part, render=False, **field_style(self._field))
                # Smooth shading draws a copy with normals; make that copy the mapper's input, so
                # a new array set on it (see paint) reaches the drawing.
                drawn = actor.mapper.dataset
                actor.mapper.dataset = drawn
                positions = element_positions(drawn, self.mesh, at_start=group == "units")
                self._painted.append((actor, drawn, *positions))
                self._add(group, actor)
            self.paint(self._step)
            return field_legend(self._field)
        spec = color_spec(self.mesh, name)
        for group, piece in self.pieces:
            for part in color_parts(spec, piece):
                self._add(group, self.plotter.add_mesh(part.mesh, render=False, **part.kwargs))
        return color_legend(spec)

    def paint(self, step: int) -> None:
        """Put the shown quantity's values of time step ``step`` on the geometry.

        Every step gets a new array instead of overwriting the old one: the browser caches arrays
        by a hash of their content and fetches them later, and the server answers from the array
        object the hash was made for, so an array must not change after it was sent.
        """
        self._step = int(step)
        if self._field is None:
            return
        for actor, drawn, element, position in self._painted:
            drawn.point_data[FIELD_ARRAY] = end_field_values(self._field, self._step, element, position)
            actor.mapper.Update()  # the browser view sends the mapper's input without rendering first

    def set_units_visible(self, visible: bool) -> None:
        self.units_visible = bool(visible)
        for actor in self._actors["units"]:
            actor.SetVisibility(self.units_visible)

    def set_lung_visible(self, visible: bool) -> None:
        if self.lung_actor is not None:
            self.lung_actor.SetVisibility(bool(visible))

    def element_at(self, event) -> int | None:
        """The element a click event of the scene view hits; None if it hits none.

        Only the visible parts can be hit, and the lung surface never hides an element. Without
        quantities to plot there is nothing to pick.
        """
        if self.picker is None or not isinstance(event, dict):
            return None
        ray = click_ray(event, self.plotter)
        if ray is None:
            return None
        return self.picker.pick(*ray, ["airways", "units"] if self.units_visible else ["airways"])

    def update_view(self) -> None:
        """Send the changed scene to the browser."""
        get_viewer(self.plotter).update()

    def _add(self, group: str, actor) -> None:
        if group == "units":
            actor.SetVisibility(self.units_visible)
        self._actors[group].append(actor)

    def _front_camera(self) -> None:
        """Set and fit the camera: upright, seen from the front, trachea up, right lung on the left.

        PyVista otherwise places the camera only on the first render; the browser view takes this
        camera over with the first scene.
        """
        view = front_view(self.mesh)
        if view is None:
            self.plotter.view_isometric()
            return
        direction, up = view
        bounds = np.reshape(self.plotter.bounds, (3, 2))
        focus = bounds.mean(axis=1)
        distance = 3 * np.linalg.norm(bounds[:, 1] - bounds[:, 0])
        self.plotter.camera_position = [tuple(focus - distance * direction), tuple(focus), tuple(up)]
        self.plotter.reset_camera()


def add_scene_view(scene: LungScene, click) -> None:
    """The scene's view in the page, rendered in the browser.

    ``click`` is called with (event, moved) for every click: the view's event (for
    ``LungScene.element_at``) and whether the pointer moved more than DRAG pixels between press
    and release. A drag that rotates the view ends with a click too, so the pointer's way is
    noted in the capture phase, before the view handles the events. The page's state needs
    ``pick_modes`` (["click"] to report clicks), ``pick_from`` and ``pick_moved``.
    """
    with html.Div(
        style="width: 100%; height: 100%",
        raw_attrs=[
            '@pointerdown.capture="pick_from = [$event.clientX, $event.clientY]"',
            (
                '@pointerup.capture="pick_moved = Math.hypot($event.clientX - pick_from[0], '
                f'$event.clientY - pick_from[1]) > {DRAG}"'
            ),
        ],
    ) as container:
        # The view's state and callbacks belong to the page's server; without this, plotter_ui
        # would put them on trame's default server (one viewer per plotter, reused from here on)
        get_viewer(scene.plotter, server=container.server)
        # disable_auto_switch: while new geometry loads, the browser view would otherwise show the
        # server-rendered image for a moment (other size, other framing) on every time step
        plotter_ui(
            scene.plotter,
            default_server_rendering=False,
            disable_auto_switch=True,
            picking_modes=("pick_modes",),
            click=(click, "[$event, pick_moved]"),
        )


def _field(quantity: TimeQuantity, units: np.ndarray) -> EndField:
    """How a quantity is drawn, with a color range over the values that are drawn.

    A spread (``uncertain``) gets the single-hue uncertainty colormap, from zero (certain) up;
    values of both signs a diverging colormap centered on zero; other values the default one.
    A terminal unit shows the value at its first end only (``units``: which elements are
    terminal units), so the value at its last end does not widen the range.
    """
    drawn_end = np.where(units, quantity.start, quantity.end)
    if quantity.uncertain:
        lo, hi, cmap = 0.0, value_range(quantity.start, drawn_end)[1], UNCERTAINTY_COLORMAP
    elif quantity.signed:
        (lo, hi), cmap = symmetric_range(quantity.start, drawn_end), DIVERGING_COLORMAP
    else:
        (lo, hi), cmap = value_range(quantity.start, drawn_end), COLORMAP
    return EndField(quantity.label, quantity.start, quantity.end, lo, hi, cmap=cmap)
