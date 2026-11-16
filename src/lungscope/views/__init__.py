"""Spatial views and linked plots for the trame application.

The views draw what the app selects. They never open files and never import ``data``: the app
hands them the study dataset or mesh, the output times and the quantities from ``analysis``.

Public interface
- ``entity_mesh(study)``: one line cell per entity of a study dataset, with the arrays that are
  the same in every sample; the mesh the scene draws.
- ``LungScene(mesh, times=None, quantities=())``: the 3D scene: airway and terminal-unit tubes,
  joint spheres, and the translucent lung with its lobes around them.
  - What it shows: ``mesh``, ``times`` (output times in s), ``units`` (which elements are
    terminal units), ``charted`` (the quantities with every run's values, by name),
    ``has_units`` and ``has_lung``.
  - The coloring: ``options`` and ``groups`` are the "Color by" choices; ``show(name)`` colors
    by one of them and returns its legend; ``color_by`` and ``legend`` are the current ones;
    ``changes_over_time`` tells whether the coloring follows the time step.
  - ``paint(step)`` draws a time step; ``set_units_visible`` and ``set_lung_visible`` toggle
    parts; ``element_at(event)`` tells which element a click hits.
  - ``update_view()`` sends the changes to the browser; call it after ``show``, ``paint`` or a
    visibility change.
- ``add_scene_view(scene, click)``: the scene's view in the page; ``click(event, moved)``.
- ``add_legend()``: the legend of the coloring (state variable ``legend``).
- ``run_chart(times, runs, label, unit, revision)``: the plot of one element's runs over time;
  a ``RunChart`` with ``figure`` (a Plotly figure as plain data, without a time marker),
  ``times`` and ``readouts`` (one per time step), or None if no run has a value.
  ``figure_at(chart, step)`` marks a time step, ``element_title(mesh, element, is_unit)``
  names the element.
- ``add_run_chart(state_name, height)``: the plot's widget.
- ``add_demo_view()``: the starter scene, which needs no simulation files.

Modules: ``geometry`` (tubes, spheres, nodes), ``lung_shape`` (the lung with its lobes),
``coloring`` ("Color by" choices, legends, quantities over time), ``picking`` (what a click
hits), ``scene``, ``legend``, ``run_chart`` and ``demo``.
"""

from lungscope.views.demo import add_demo_view
from lungscope.views.geometry import entity_mesh
from lungscope.views.legend import add_legend
from lungscope.views.run_chart import (
    RunChart,
    add_run_chart,
    element_title,
    figure_at,
    run_chart,
)
from lungscope.views.scene import LungScene, add_scene_view

__all__ = [
    "LungScene",
    "RunChart",
    "add_demo_view",
    "add_legend",
    "add_run_chart",
    "add_scene_view",
    "element_title",
    "entity_mesh",
    "figure_at",
    "run_chart",
]
