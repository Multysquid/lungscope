"""Which element a click in the 3D view hits: the first drawn cell along the line of sight.

The browser view (local rendering) reports the line of sight through the clicked pixel; for the
server-rendered view it is computed here from the server's camera. Only the geometry handed to
``ElementPicker`` can be hit, so the lung surface drawn around the tree never hides an element.
"""

from collections.abc import Iterable

import numpy as np
import pyvista as pv
from vtkmodules.vtkCommonCore import reference
from vtkmodules.vtkCommonDataModel import vtkStaticCellLocator

TOLERANCE = 1e-6  # for the line-cell intersection tests


class ElementPicker:
    """Finds the element under a click; pieces carry ``element_index`` per cell (build_geometry)."""

    def __init__(self, pieces: dict[str, pv.PolyData | None]) -> None:
        self._pieces = {}
        for name, piece in pieces.items():
            if piece is None or piece.n_cells == 0:
                continue
            locator = vtkStaticCellLocator()
            locator.SetDataSet(piece)
            locator.BuildLocator()
            self._pieces[name] = (locator, np.asarray(piece.cell_data["element_index"]))

    def pick(self, near, far, pieces: Iterable[str] | None = None) -> int | None:
        """Element of the first cell on the line from ``near`` to ``far`` (world coordinates).

        Only the named pieces (all by default) can be hit; None if the line misses them.
        """
        names = list(self._pieces) if pieces is None else [name for name in pieces if name in self._pieces]
        start, stop = [float(v) for v in near], [float(v) for v in far]
        best, first = None, np.inf
        for name in names:
            locator, element = self._pieces[name]
            t, cell = reference(0.0), reference(-1)
            hit = locator.IntersectWithLine(start, stop, TOLERANCE, t, [0.0] * 3, [0.0] * 3, reference(0), cell)
            if hit and float(t) < first:
                best, first = int(element[int(cell)]), float(t)
        return best


def display_ray(renderer, x: float, y: float) -> tuple[np.ndarray, np.ndarray]:
    """Line of sight through a display point (pixels from the bottom left): near and far end."""
    ends = []
    for depth in (0.0, 1.0):
        renderer.SetDisplayPoint(float(x), float(y), depth)
        renderer.DisplayToWorld()
        point = np.asarray(renderer.GetWorldPoint(), dtype=float)
        ends.append(point[:3] / point[3] if point[3] else point[:3])
    return ends[0], ends[1]


def click_ray(event: dict, plotter: pv.Plotter) -> tuple[np.ndarray, np.ndarray] | None:
    """Line of sight of a click event from the 3D view, or None if it has none.

    The browser view sends ``ray`` (near and far end). The server-rendered view sends the pixel,
    counted from the bottom left in a canvas of ``size``, for the server's render window.
    """
    if event.get("mode") == "remote":
        position, size = event.get("position") or {}, event.get("size") or (0, 0)
        if "x" not in position or "y" not in position:
            return None
        width, height = plotter.render_window.GetSize()
        scale_x = width / size[0] if size[0] else 1.0
        scale_y = height / size[1] if size[1] else 1.0
        return display_ray(plotter.renderer, position["x"] * scale_x, position["y"] * scale_y)
    ray = event.get("ray")
    if not ray or len(ray) != 2:
        return None
    return np.asarray(ray[0], dtype=float), np.asarray(ray[1], dtype=float)
