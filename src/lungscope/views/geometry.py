"""Geometry of the lung model for drawing: airway and terminal-unit tubes, joint spheres.

The model is a mesh of line elements (airways and terminal units, as in 4C's VTU input).
Airways become cylinders with the radius from the cell array ``radius`` (a thin default where
there is none) and get a sphere wherever they meet. Terminal units, the elements with a reference
volume (``v0`` in 4C input files, ``v_0`` in result files), become cylinders too: they continue
the airway they hang on, with its radius (their own ``radius`` is a placeholder zero in 4C's
input). Their reference volume is not drawn; the lung surface around the tree accounts for it.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pyvista as pv

if TYPE_CHECKING:
    import xarray as xr

SPHERE_SCALE = 1.3  # joint sphere radius relative to the thickest cylinder at a point
VOLUME_ARRAYS = ("v0", "v_0")  # reference volume of the terminal units: 4C input, 4C result
GEOMETRY_ARRAYS = ("block_id", "radius", "v0")  # input arrays that decide what gets drawn
LINE_CELLS = {pv.CellType.LINE, pv.CellType.POLY_LINE}


def entity_mesh(study: "xr.Dataset") -> pv.UnstructuredGrid:
    """One line cell per entity of a study (see ``data.load_study``), in entity order.

    Cells carry ``entity_id`` and the per-entity arrays that are the same in every sample
    (block_id, generation, radius, v0, ...), like the input mesh of a single run. A study whose
    drawn geometry (block_id, radius, v0) differs between samples is refused.
    """
    varying = [name for name in GEOMETRY_ARRAYS if name in study.data_vars]
    if varying:
        raise ValueError(f"{', '.join(varying)} differ between samples: such a study cannot be drawn yet")
    ends = np.asarray(study["endpoints"].values, dtype=float)
    points, inverse = np.unique(ends.reshape(-1, 3), axis=0, return_inverse=True)
    cells = np.column_stack([np.full(len(ends), 2), inverse.reshape(-1, 2)]).ravel()
    types = np.full(len(ends), pv.CellType.LINE, dtype=np.uint8)
    mesh = pv.UnstructuredGrid(cells, types, points)
    mesh.cell_data["entity_id"] = np.asarray(study["entity"].values)
    for name, coord in study.coords.items():
        if name != "entity" and coord.dims == ("entity",) and coord.dtype.kind in "biuf":
            mesh.cell_data[name] = np.asarray(coord.values)
    return mesh


def tube_radii(mesh: pv.UnstructuredGrid) -> np.ndarray:
    """Radius per element: ``radius`` where positive, otherwise a thin default."""
    if "radius" in mesh.cell_data:
        radius = np.asarray(mesh.cell_data["radius"], dtype=float)
    else:
        radius = np.zeros(mesh.n_cells)
    positive = np.isfinite(radius) & (radius > 0)
    thin = 0.35 * radius[positive].min() if positive.any() else 0.005 * mesh.length
    return np.where(positive, radius, thin)


def is_line_mesh(mesh: pv.UnstructuredGrid) -> bool:
    return mesh.n_cells > 0 and set(np.unique(mesh.celltypes)) <= LINE_CELLS


def unit_volumes(mesh: pv.UnstructuredGrid) -> np.ndarray | None:
    """Reference volume of each terminal unit, NaN for the other elements; None if there are none."""
    for name in VOLUME_ARRAYS:
        if name in mesh.cell_data:
            volume = np.asarray(mesh.cell_data[name], dtype=float)
            volume = np.where(np.isfinite(volume) & (volume > 0), volume, np.nan)
            if np.isfinite(volume).any():
                return volume
    return None


def unit_mask(mesh: pv.UnstructuredGrid) -> np.ndarray:
    """Which elements are terminal units: the elements with a reference volume."""
    volumes = unit_volumes(mesh)
    return np.isfinite(volumes) if volumes is not None else np.zeros(mesh.n_cells, dtype=bool)


def unit_radii(volumes: np.ndarray) -> np.ndarray:
    """Radius of a sphere with the given volume."""
    return np.cbrt(3.0 * np.asarray(volumes, dtype=float) / (4.0 * np.pi))


@dataclass
class Geometry:
    """What gets drawn; every cell carries ``element_index``, the element it shows."""

    airways: pv.PolyData | None  # cylinders and joint spheres, or the surface of a non-line mesh
    units: pv.PolyData | None  # one cylinder per terminal unit


def build_geometry(mesh: pv.UnstructuredGrid, n_sides: int = 24) -> Geometry:
    """Cylinders for the airways and the terminal units, spheres where airways meet.

    A joint sphere belongs to the thickest element at its point and is SPHERE_SCALE x its radius.
    Points that only terminal units reach (their free ends) get no joint sphere; the cylinders
    are capped there. A mesh that is not made of lines is shown as its surface.
    """
    mesh = mesh.copy()
    mesh.cell_data["element_index"] = np.arange(mesh.n_cells)
    if not is_line_mesh(mesh):
        return Geometry(surface(mesh), None)
    radii = tube_radii(mesh)
    is_unit = unit_mask(mesh)
    nodes = mesh_nodes(mesh, radii)
    pieces = []
    airways = np.flatnonzero(~is_unit)
    if airways.size:
        pieces.append(_cylinders(mesh.extract_cells(airways), radii, n_sides))
    joints = ~is_unit[nodes.owner]
    if joints.any():
        owner = nodes.owner[joints]
        pieces.append(_spheres(nodes.positions[joints], SPHERE_SCALE * radii[owner], owner))
    units = None
    if is_unit.any():
        ids = np.flatnonzero(is_unit)
        unit_tubes = radii.copy()
        unit_tubes[ids] = unit_tube_radii(nodes, radii, ids)
        units = _cylinders(mesh.extract_cells(ids), unit_tubes, n_sides)
    return Geometry(surface(pv.merge(pieces, merge_points=False)) if pieces else None, units)


def _cylinders(lines: pv.UnstructuredGrid, radii: np.ndarray, n_sides: int) -> pv.PolyData:
    separated = lines.separate_cells()  # no shared points, so each element keeps its own radius
    separated.cell_data["tube_radius"] = radii[np.asarray(separated.cell_data["element_index"])]
    polylines = surface(separated).cell_data_to_point_data(pass_cell_data=True)
    tubes = polylines.tube(scalars="tube_radius", absolute=True, n_sides=n_sides, capping=True)
    element_index = np.asarray(tubes.cell_data["element_index"])  # kept from the lines
    tubes.clear_data()
    tubes.cell_data["element_index"] = element_index
    return tubes


def _spheres(centres: np.ndarray, radii: np.ndarray, elements: np.ndarray) -> pv.PolyData:
    cloud = pv.PolyData(np.asarray(centres, dtype=float))
    cloud["scale"] = np.asarray(radii, dtype=float)
    template = pv.Sphere(radius=1.0, theta_resolution=24, phi_resolution=16)
    spheres = cloud.glyph(geom=template, scale="scale", orient=False)
    spheres.clear_data()
    spheres.cell_data["element_index"] = np.repeat(np.asarray(elements), template.n_cells)
    return spheres


def surface(data) -> pv.PolyData:
    """Outer surface as PolyData; line elements pass through as lines."""
    return data.extract_surface(pass_pointid=False, pass_cellid=False, algorithm="dataset_surface")


@dataclass
class Nodes:
    """Mesh points with coincident points merged (4C result files repeat them per element)."""

    positions: np.ndarray  # (n_nodes, 3)
    owner: np.ndarray  # thickest airway at each node (a terminal unit only where no airway is)
    ends: np.ndarray  # (n_cells, 2) node of the first and the last point of every element


def mesh_nodes(mesh: pv.UnstructuredGrid, radii: np.ndarray) -> Nodes:
    offsets = np.asarray(mesh.cell_offsets)
    connectivity = np.asarray(mesh.cell_connectivity)
    element_of_entry = np.repeat(np.arange(mesh.n_cells), np.diff(offsets))
    coords = np.asarray(mesh.points)[connectivity]
    keys = np.round(coords / max(mesh.length, 1e-12), 9)
    _, node_of_entry = np.unique(keys, axis=0, return_inverse=True)
    node_of_entry = node_of_entry.reshape(-1)
    # owner of a node: airways before terminal units (4C result files give both the same radius,
    # and the choice must not depend on the element order), then the thickest; per node, the
    # entry of the owner comes last after sorting by (node, priority)
    priority = np.where(unit_mask(mesh), -1.0, radii)
    order = np.lexsort((priority[element_of_entry], node_of_entry))
    last = np.r_[node_of_entry[order][1:] != node_of_entry[order][:-1], True]
    chosen = order[last]
    ends = np.stack([node_of_entry[offsets[:-1]], node_of_entry[offsets[1:] - 1]], axis=1)
    return Nodes(coords[chosen], element_of_entry[chosen], ends)


def unit_centres(nodes: Nodes, units: np.ndarray) -> np.ndarray:
    """Centre of each terminal unit's volume: at its free (alveolar) end, else at its middle."""
    degree = np.bincount(nodes.ends.ravel(), minlength=len(nodes.positions))
    a, b = nodes.ends[units].T
    free = np.where(degree[b] == 1, b, np.where(degree[a] == 1, a, -1))
    middle = 0.5 * (nodes.positions[a] + nodes.positions[b])
    return np.where((free >= 0)[:, None], nodes.positions[np.maximum(free, 0)], middle)


def unit_tube_radii(nodes: Nodes, radii: np.ndarray, units: np.ndarray) -> np.ndarray:
    """Tube radius of each terminal unit: the radius of the element that owns its inner end.

    The inner end is the one that is not free (a free end belongs to this element alone), so a
    terminal unit continues the thickest airway there. Where only terminal units meet, the owner
    is a terminal unit, and its own (thin default) radius is used.
    """
    degree = np.bincount(nodes.ends.ravel(), minlength=len(nodes.positions))
    first, last = nodes.ends[units].T
    inner = np.where((degree[first] == 1) & (degree[last] != 1), last, first)
    return radii[nodes.owner[inner]]
