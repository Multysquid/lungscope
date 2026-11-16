"""Colors for the lung model: the "Color by" choices, legends, and fields that change over time.
"""

from dataclasses import dataclass, field

import matplotlib
import numpy as np
import pyvista as pv
from matplotlib.colors import LinearSegmentedColormap

from lungscope.views.geometry import surface

SOLID = "Solid color"
SOLID_COLOR = "#2a78d6"
NOT_DEFINED_COLOR = "#c3c2b7"
NOT_DEFINED_LABEL = "not defined for this element"
COLORMAP = "viridis"
DIVERGING_COLORMAP = "coolwarm"  # blue - neutral gray - red, for values on both sides of zero
# One hue from light to dark for a spread such as a standard deviation: light = certain (no
# spread between the runs), dark = uncertain. It is matplotlib's "Oranges" without its
# near-white end, so that the most certain elements still stand out from the background.
UNCERTAINTY_COLORMAP = "lungscope_uncertainty"
UNCERTAINTY_RAMP = ("Oranges", 0.25, 1.0)
MAX_CLASSES = 8  # integer arrays spanning at most this many values get one color per value
INTERNAL_ARRAYS = {"element_index", "tube_radius", "Normals", "vtkOriginalCellIds", "vtkOriginalPointIds"}
FIELD_ARRAY = "value"  # point array of the geometry that holds a field's values for one time step


def _register_uncertainty_colormap() -> None:
    """Make UNCERTAINTY_COLORMAP known to matplotlib, where PyVista looks colormaps up by name."""
    if UNCERTAINTY_COLORMAP in matplotlib.colormaps:
        return
    name, first, last = UNCERTAINTY_RAMP
    colors = matplotlib.colormaps[name](np.linspace(first, last, 256))
    matplotlib.colormaps.register(LinearSegmentedColormap.from_list(UNCERTAINTY_COLORMAP, colors))


_register_uncertainty_colormap()


def color_options(mesh: pv.UnstructuredGrid) -> list[str]:
    """Names offered in "Color by": solid color plus the scalar cell arrays of the mesh."""
    names = [
        name for name in mesh.cell_data
        if name not in INTERNAL_ARRAYS and np.asarray(mesh.cell_data[name]).ndim == 1
    ]
    return [SOLID, *names]


def not_defined(mesh: pv.UnstructuredGrid, name: str) -> np.ndarray:
    """Elements for which an array has no meaning.

    NaN values, and placeholder zeros: an element type (``block_id``) whose values are all zero
    while other element types have non-zero values, e.g. ``radius`` on terminal units or
    ``elasticity_e`` on airways in 4C's lung input.
    """
    values = np.asarray(mesh.cell_data[name], dtype=float)
    undefined = ~np.isfinite(values)
    if "block_id" in mesh.cell_data and name != "block_id":
        blocks = np.asarray(mesh.cell_data["block_id"])
        for block in np.unique(blocks):
            inside = blocks == block
            if np.all(values[inside] == 0) and np.any(values[~inside] != 0):
                undefined |= inside
    return undefined


def colormap_colors(n: int, cmap: str = COLORMAP) -> list[str]:
    """n colors spread evenly over a colormap (its middle for n = 1), as hex."""
    table = np.asarray(pv.LookupTable(cmap, n_values=256).values)[:, :3]
    picks = [128] if n == 1 else np.rint(np.linspace(0, 255, n)).astype(int)
    return ["#{:02x}{:02x}{:02x}".format(*table[i]) for i in picks]


def _fmt(value: float) -> str:
    return f"{value:.0f}" if float(value).is_integer() else f"{value:.4g}"


def _fmt_range(lo: float, hi: float) -> tuple[str, str]:
    """Labels for the ends of a range, with enough digits to tell them apart (1000.064, 1000.066)."""
    low, high = _fmt(lo), _fmt(hi)
    digits = 4
    while low == high and digits < 15:
        digits += 1
        low, high = f"{lo:.{digits}g}", f"{hi:.{digits}g}"
    return low, high


def nice_ticks(lo: float, hi: float, target: int = 5) -> list[tuple[float, str]]:
    """Tick values on round numbers (steps of 1, 2, 2.5 or 5 x 10^k) within [lo, hi], with labels."""
    raw = (hi - lo) / (target - 1)
    magnitude = 10.0 ** np.floor(np.log10(raw))
    for factor in (1.0, 2.0, 2.5, 5.0, 10.0):
        step = factor * magnitude
        if step >= raw * (1 - 1e-9):
            break
    first = np.ceil(lo / step - 1e-9) * step
    count = int(np.floor((hi - first) / step + 1e-9)) + 1
    decimals = max(0, int(-np.floor(np.log10(step) + 1e-12)) + (1 if factor == 2.5 else 0))
    values = first + step * np.arange(count)
    ticks = [(v, f"{0.0 if abs(v) < 1e-9 * step else v:.{decimals}f}") for v in values]
    if len(ticks) < 2:  # the range is too narrow for round numbers: label its ends
        return list(zip((lo, hi), _fmt_range(lo, hi), strict=True))
    return ticks


@dataclass
class ColorSpec:
    """How one "Color by" choice maps the elements to colors."""

    title: str
    values: np.ndarray | None = None  # per element; None for the solid color
    undefined: np.ndarray | None = None  # per element: shown gray
    lo: float = 0.0
    hi: float = 0.0
    n_classes: int = 0  # > 0: one color per integer from lo to lo + n_classes - 1
    class_colors: list = field(default_factory=list)
    classes: list = field(default_factory=list)  # the values that occur, with their colors


def color_spec(mesh: pv.UnstructuredGrid, name: str) -> ColorSpec:
    if name == SOLID:
        return ColorSpec(SOLID)
    values = np.asarray(mesh.cell_data[name], dtype=float)
    spec = ColorSpec(name, values, not_defined(mesh, name))
    defined = values[~spec.undefined]
    if defined.size == 0:
        return spec
    spec.lo, spec.hi = float(defined.min()), float(defined.max())
    integers = bool(np.all(defined == np.rint(defined)))
    if spec.lo == spec.hi or (integers and spec.hi - spec.lo + 1 <= MAX_CLASSES):
        spec.n_classes = int(spec.hi - spec.lo) + 1 if integers else 1
        spec.class_colors = colormap_colors(spec.n_classes)
        present = np.unique(defined)
        index = (present - spec.lo).astype(int) if integers else np.zeros(len(present), dtype=int)
        spec.classes = [(float(v), spec.class_colors[k]) for v, k in zip(present, index, strict=True)]
    return spec


@dataclass
class Part:
    """A piece of the geometry with the keyword arguments for ``Plotter.add_mesh``."""

    mesh: pv.PolyData
    kwargs: dict


def color_parts(spec: ColorSpec, geometry: pv.PolyData | None) -> list[Part]:
    """Split a piece of geometry into a colored part and a gray (not defined) part."""
    style = {"smooth_shading": True, "show_scalar_bar": False}
    if geometry is None or geometry.n_cells == 0:
        return []
    if spec.values is None:
        return [Part(geometry, {**style, "color": SOLID_COLOR})]
    element = np.asarray(geometry.cell_data["element_index"])
    shown = ~spec.undefined[element]
    parts = []
    if shown.any():
        part = surface(geometry.extract_cells(np.flatnonzero(shown)))
        part_values = spec.values[np.asarray(part.cell_data["element_index"])]
        if spec.n_classes:
            clim = (spec.lo - 0.5, spec.lo + spec.n_classes - 0.5)
            kwargs = {"cmap": spec.class_colors, "n_colors": spec.n_classes, "clim": clim}
        else:
            kwargs = {"cmap": COLORMAP, "clim": (spec.lo, spec.hi)}
        parts.append(Part(part, {**style, "scalars": part_values, "preference": "cell", **kwargs}))
    if not shown.all():
        rest = surface(geometry.extract_cells(np.flatnonzero(~shown)))
        parts.append(Part(rest, {**style, "color": NOT_DEFINED_COLOR}))
    return parts


def color_legend(spec: ColorSpec, cmap: str = COLORMAP) -> dict | None:
    """Legend for the page: a horizontal color bar (gradient or one block per value) with ticks."""
    if spec.values is None:
        return None
    legend = {"title": spec.title, "gradient": None, "steps": [], "ticks": [], "items": []}
    if spec.n_classes:
        n = len(spec.classes)
        legend["steps"] = [{"color": color} for _, color in spec.classes]
        legend["ticks"] = [
            {"pos": 100 * (k + 0.5) / n, "label": _fmt(value)} for k, (value, _) in enumerate(spec.classes)
        ]
    elif (~spec.undefined).any():
        legend["gradient"] = f"linear-gradient(to right, {', '.join(colormap_colors(11, cmap))})"
        legend["ticks"] = [
            {"pos": 100 * (value - spec.lo) / (spec.hi - spec.lo), "label": label}
            for value, label in nice_ticks(spec.lo, spec.hi)
        ]
    if spec.undefined.any():
        legend["items"].append({"color": NOT_DEFINED_COLOR, "label": NOT_DEFINED_LABEL})
    return legend


@dataclass
class EndField:
    """A quantity known at both ends of every element and at every time step, like pressure.

    It is drawn smoothly: along an airway the color runs linearly from the value at its first end
    to the value at its last end, so it is continuous through the joints. A terminal unit shows
    the value at its first end, its entrance, over its whole length: its last end is not further
    down the tree but on the far side of its alveolar volume, where a boundary pressure acts.
    The color range is the same for all steps.
    A quantity of a whole element (like the pressure drop across it) has start == end, so every
    element gets one color. NaN means "not defined for this element" and is drawn gray.
    """

    title: str
    start: np.ndarray  # (time, element): value at each element's first end
    end: np.ndarray  # (time, element): value at its last end
    lo: float
    hi: float
    cmap: str = COLORMAP


def element_positions(
    piece: pv.PolyData, mesh: pv.UnstructuredGrid, at_start: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """For every point of a piece of geometry: its element, and its position along that element.

    The position is 0 at the element's first end and 1 at its last one: the point projected onto
    the line between them, clipped. With ``at_start`` every point gets 0.
    """
    element = np.rint(piece.cell_data_to_point_data()["element_index"]).astype(int)
    if at_start:
        return element, np.zeros(piece.n_points)
    ends = _cell_ends(mesh)[element]
    axis = ends[:, 1] - ends[:, 0]
    along = np.einsum("ij,ij->i", np.asarray(piece.points) - ends[:, 0], axis)
    length2 = np.einsum("ij,ij->i", axis, axis)
    return element, np.clip(along / np.where(length2 > 0, length2, 1.0), 0.0, 1.0)


def end_field_values(field: EndField, step: int, element: np.ndarray, position: np.ndarray) -> np.ndarray:
    """A field's values at points given by element and position, for one time step."""
    start, end = field.start[step, element], field.end[step, element]
    return start + position * (end - start)


def _field_range(field: EndField) -> tuple[float, float]:
    return field.lo, field.hi if field.hi > field.lo else field.lo + 1.0


def field_style(field: EndField) -> dict:
    """``add_mesh`` keyword arguments for geometry colored by a field (point array FIELD_ARRAY)."""
    return {
        "scalars": FIELD_ARRAY,
        "preference": "point",
        "cmap": field.cmap,
        "clim": _field_range(field),
        "nan_color": NOT_DEFINED_COLOR,
        "smooth_shading": True,
        "show_scalar_bar": False,
    }


def field_legend(field: EndField) -> dict | None:
    """Legend of a field: a gradient over its fixed range with ticks, and gray for no values."""
    lo, hi = _field_range(field)
    undefined = np.isnan(field.start).all(axis=0) & np.isnan(field.end).all(axis=0)
    return color_legend(ColorSpec(field.title, np.zeros(len(undefined)), undefined, lo, hi), field.cmap)


def _cell_ends(mesh: pv.UnstructuredGrid) -> np.ndarray:
    """First and last point of every cell, (cells, 2, 3)."""
    offsets = np.asarray(mesh.cell_offsets)
    connectivity = np.asarray(mesh.cell_connectivity)
    points = np.asarray(mesh.points, dtype=float)
    return np.stack([points[connectivity[offsets[:-1]]], points[connectivity[offsets[1:] - 1]]], axis=1)
