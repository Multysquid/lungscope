"""The translucent lung drawn around the airway tree: two lungs and their lobes.

``lobe_surface`` draws real lobes from lobe masks (e.g. segmented from a CT), placed on the mesh.
Without masks, ``lung_surface`` draws an illustrative lung, styled like anatomical renderings:
two lungs separated by the mediastinum, the right one (on the left of the front view) split into
three lobes, the left one into two; the tree itself carries no lobe information.
"""

import itertools
from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import pyvista as pv

from lungscope.views.geometry import (
    SPHERE_SCALE,
    Nodes,
    is_line_mesh,
    mesh_nodes,
    surface,
    tube_radii,
    unit_centres,
    unit_mask,
    unit_radii,
    unit_volumes,
)

LUNG_COLOR = "#898781"
LUNG_OPACITY = 0.15
OBLIQUE_TILT = 45.0  # degrees from horizontal; the oblique fissure rises towards the back
HORIZONTAL_LEVEL = 0.2  # right lung: horizontal fissure this far above the centre (x half-height)
MASK_STEP = 2  # lobe masks are coarsened by this many voxels per axis before their surface is taken


def lobe_surface(
    masks: Sequence[np.ndarray], mesh_from_index: Sequence[np.ndarray], step: int = MASK_STEP
) -> pv.PolyData | None:
    """The lung from lobe masks: one closed, smoothed surface per lobe, in mesh coordinates.

    ``masks``: one boolean voxel mask per lobe, indexed (i, j, k); ``mesh_from_index``: 4 x 4 per
    mask, the mesh coordinates of the centre of voxel (i, j, k). Each mask is coarsened to the
    fraction of every step x step x step block that is inside, and its surface is where that
    fraction is one half. Neighbouring lobes touch, so their shared walls show the fissures.
    Returns None if no mask has a voxel inside.
    """
    sections = []
    for mask, matrix in zip(masks, mesh_from_index, strict=True):
        fraction, matrix = _coarsen(np.asarray(mask, dtype=bool), np.asarray(matrix, dtype=float), step)
        image = pv.ImageData(dimensions=fraction.shape)
        image.point_data["inside"] = fraction.ravel(order="F")
        lobe = image.contour([0.5], scalars="inside", method="flying_edges")
        if lobe.n_points == 0:
            continue
        lobe.clear_data()
        lobe = lobe.smooth_taubin(n_iter=20, pass_band=0.05)  # in voxels, before any stretching
        lobe.points = np.asarray(lobe.points) @ matrix[:3, :3].T + matrix[:3, 3]
        sections.append(lobe.compute_normals(auto_orient_normals=True))
    if not sections:
        return None
    return surface(pv.merge(sections, merge_points=False))


def _coarsen(mask: np.ndarray, matrix: np.ndarray, step: int) -> tuple[np.ndarray, np.ndarray]:
    """Fraction inside per step^3 block, with a layer of empty blocks around it, and its 4 x 4.

    Block (I, J, K) covers voxels step * I ... step * I + step - 1, so its centre is voxel
    step * I + (step - 1) / 2; the empty layer shifts the blocks by one.
    """
    padded = np.pad(mask, [(0, -n % step) for n in mask.shape])
    blocks = padded.reshape(padded.shape[0] // step, step, padded.shape[1] // step, step, -1, step)
    fraction = np.pad(blocks.mean(axis=(1, 3, 5), dtype=np.float32), 1)
    index_from_block = np.diag([step, step, step, 1.0])
    index_from_block[:3, 3] = (step - 1) / 2 - step
    return fraction, matrix @ index_from_block


def lung_regions(nodes: Nodes, radii: np.ndarray) -> tuple[list[np.ndarray], int | None, int | None]:
    """Elements of each lung: everything downstream of each branch at the first bifurcation.

    The root element (trachea, see _root_element) has a free end, the inlet; the node at its other
    end is the carina. Returns the regions (first entry of each: the main branch), the carina and
    the inlet.
    """
    degree = np.bincount(nodes.ends.ravel(), minlength=len(nodes.positions))
    free_end = np.flatnonzero((degree[nodes.ends] == 1).any(axis=1))
    if free_end.size == 0:
        return [], None, None
    root = _root_element(nodes, radii, free_end, degree)
    first, second = nodes.ends[root]
    carina, inlet = (second, first) if degree[first] == 1 else (first, second)
    incident = defaultdict(list)
    for element, (a, b) in enumerate(nodes.ends):
        incident[a].append(element)
        incident[b].append(element)
    regions = []
    for branch in incident[carina]:
        if branch == root:
            continue
        region, queue, seen = [], [branch], {root, branch}
        while queue:
            element = queue.pop(0)
            region.append(element)
            for node in nodes.ends[element]:
                if node == carina:
                    continue
                for neighbour in incident[node]:
                    if neighbour not in seen:
                        seen.add(neighbour)
                        queue.append(neighbour)
        regions.append(np.array(region))
    return regions, int(carina), int(inlet)


def _root_element(nodes: Nodes, radii: np.ndarray, free_end: np.ndarray, degree: np.ndarray) -> int:
    """The trachea among the elements with a free end.

    The uniquely thickest one if the radii tell (4C input files). Otherwise, e.g. in 4C result
    files, which carry no radius, the one whose free end is the most central node of the tree
    (smallest eccentricity): in a branching airway tree that is the inlet, every other free end
    is a terminal unit deep in the tree.
    """
    thickness = radii[free_end]
    if np.count_nonzero(thickness == thickness.max()) == 1:
        return int(free_end[np.argmax(thickness)])
    neighbours = defaultdict(list)
    for a, b in nodes.ends:
        neighbours[a].append(b)
        neighbours[b].append(a)

    def hops(start: int) -> dict:
        distance, queue = {start: 0}, deque([start])
        while queue:
            node = queue.popleft()
            for following in neighbours[node]:
                if following not in distance:
                    distance[following] = distance[node] + 1
                    queue.append(following)
        return distance

    leaf = {
        int(e): int(a if degree[a] == 1 else b)
        for e, (a, b) in zip(free_end, nodes.ends[free_end], strict=True)
    }
    # in a tree the farthest node from any node is an end of a longest path, so the eccentricity of
    # a node is its larger distance to the two ends of such a path
    sweep = hops(next(iter(leaf.values())))
    from_a = hops(max(sweep, key=sweep.get))
    from_b = hops(max(from_a, key=from_a.get))
    eccentricity = {e: max(from_a.get(n, np.inf), from_b.get(n, np.inf)) for e, n in leaf.items()}
    return min(eccentricity, key=eccentricity.get)


@dataclass
class Lobe:
    """One main branch: points the lung must enclose (each with a clearance radius), its bronchus."""

    points: np.ndarray
    reach: np.ndarray
    bronchus: np.ndarray  # unit direction of the main bronchus, from the carina outwards
    bronchus_radius: float

    @property
    def centre(self) -> np.ndarray:
        return self.points.mean(axis=0)


@dataclass
class Tree:
    """A tree split at the carina, with anatomical directions for an upright front view."""

    carina: np.ndarray
    inlet: np.ndarray
    carina_radius: float  # radius of the sphere at the carina
    up: np.ndarray  # along the trachea, towards the inlet
    lateral: np.ndarray  # from the right lung (first lobe) towards the left lung
    anterior: np.ndarray  # towards the viewer of the front view
    lobes: list[Lobe]  # two lobes: right lung first


def analyse_tree(mesh: pv.UnstructuredGrid) -> Tree | None:
    """Trachea axis, sides and branches of a tree of line elements; None for any other mesh."""
    if not is_line_mesh(mesh):
        return None
    radii = tube_radii(mesh)
    nodes = mesh_nodes(mesh, radii)
    regions, carina, inlet = lung_regions(nodes, radii)
    if not regions:
        return None
    positions = nodes.positions
    joint_radius = SPHERE_SCALE * radii[nodes.owner]
    is_unit = unit_mask(mesh)
    units = np.flatnonzero(is_unit)
    unit_centre = dict(zip(units, unit_centres(nodes, units), strict=True))
    unit_reach = dict(zip(units, unit_radii(unit_volumes(mesh)[units]) if units.size else [], strict=True))
    lobes = []
    for region in regions:
        main = region[0]
        start, end = nodes.ends[main]
        if end == carina:
            start, end = end, start
        points, reach = [], []
        for k, element in enumerate(region):
            if is_unit[element]:
                points.append(unit_centre[element])
                reach.append(unit_reach[element])
                continue
            a, b = (start, end) if k == 0 else nodes.ends[element]
            # samples along the airway; the first 30 % of the main bronchus stays outside
            for t in np.linspace(0.3 if k == 0 else 0.0, 1.0, 6):
                points.append(positions[a] + t * (positions[b] - positions[a]))
                reach.append(radii[element])
        joints = np.unique(nodes.ends[region].ravel())
        joints = joints[(joints != carina) & ~is_unit[nodes.owner[joints]]]
        points.extend(positions[joints])
        reach.extend(joint_radius[joints])
        bronchus = _unit(positions[end] - positions[start])
        lobes.append(Lobe(np.array(points), np.array(reach), bronchus, float(radii[main])))
    up = _unit(positions[inlet] - positions[carina])
    lobes = _right_first(lobes, up)
    lateral = _perp(lobes[-1].centre - (lobes[0].centre if len(lobes) > 1 else positions[carina]), up)
    if not np.linalg.norm(lateral) > 0:  # degenerate: any direction across the trachea
        lateral = _perp(np.eye(3)[np.argmin(np.abs(up))], up)
    lateral = _unit(lateral)
    return Tree(
        carina=positions[carina],
        inlet=positions[inlet],
        carina_radius=float(joint_radius[carina]),
        up=up,
        lateral=lateral,
        anterior=-np.cross(up, lateral),
        lobes=lobes,
    )


def _right_first(lobes: list[Lobe], up: np.ndarray) -> list[Lobe]:
    """Order two lobes as (right lung, left lung).

    The right main bronchus is steeper (closer to the trachea axis) and wider than the left one.
    If the tree does not tell (mirror-symmetric), the order follows the positions, so that it does
    not depend on how the elements are numbered.
    """
    if len(lobes) != 2:
        return lobes
    first, second = lobes
    steepness = [np.degrees(np.arccos(np.clip(-lobe.bronchus @ up, -1.0, 1.0))) for lobe in lobes]
    if abs(steepness[0] - steepness[1]) > 2.0:
        return lobes if steepness[0] < steepness[1] else [second, first]
    widest = max(first.bronchus_radius, second.bronchus_radius)
    if abs(first.bronchus_radius - second.bronchus_radius) > 0.02 * widest:
        return lobes if first.bronchus_radius > second.bronchus_radius else [second, first]
    key = [tuple(np.round(lobe.centre, 9)) for lobe in lobes]
    return lobes if key[0] <= key[1] else [second, first]


def front_view(mesh: pv.UnstructuredGrid) -> tuple[np.ndarray, np.ndarray] | None:
    """Viewing direction and view-up for an upright front view: trachea up, right lung on the left."""
    tree = analyse_tree(mesh)
    if tree is None:
        return None
    return -tree.anterior, tree.up


@dataclass(frozen=True)
class Egg:
    """Egg-shaped body in local coordinates: s outwards, h up (trachea axis), d depth."""

    s_centre: float
    h_centre: float
    a: float  # lateral semi-axis at the height of the centre
    b: float  # vertical semi-axis
    c: float  # depth semi-axis at the height of the centre
    taper: float = 0.3  # the base is (1 + taper) times, the apex region (1 - taper) times as wide

    def q(self, s, h, d):
        """Squared normalised radius: below 1 inside."""
        widen = 1 + self.taper * (self.h_centre - h) / self.b
        return (
            ((s - self.s_centre) / (self.a * widen)) ** 2
            + ((h - self.h_centre) / self.b) ** 2
            + (d / (self.c * widen)) ** 2
        )


def lung_surface(mesh: pv.UnstructuredGrid, resolution: int = 64) -> pv.PolyData | None:
    """Translucent lung around the tree: one lung per main branch, split into lobes.

    Each lung is an egg-shaped body with a narrow, rounded apex next to the trachea, widening to a
    broad, rounded base, scaled until its whole branch (spheres and cylinders included) is inside.
    Between the lungs the mediastinum stays free: a gap around an upright plane between the
    branches (see _separating_plane), as wide as the branches allow, and the space around the
    trachea and the carina. The right lung is split into upper, middle and lower lobe, the left
    one into upper and lower lobe, with thin fissures. Returns None for a mesh that is not a tree
    of line elements.
    """
    tree = analyse_tree(mesh)
    if tree is None:
        return None
    fissure = 0.004 * mesh.length  # gap between neighbouring lobes
    clearance = 0.04 * mesh.length  # space between the tree and the lung surface
    widest = 0.05 * mesh.length  # largest half-width of the mediastinum
    walls = {}
    for i, j in itertools.combinations(range(len(tree.lobes)), 2):
        plane = _separating_plane(tree, tree.lobes[i], tree.lobes[j])
        if plane is not None:
            normal, offset, room = plane
            walls[i, j] = (normal, offset, max(0.5 * fissure, min(widest, 0.5 * room)))
    sections = []
    for i, lobe in enumerate(tree.lobes):
        out = _perp(lobe.centre - tree.carina, tree.up)
        if not np.linalg.norm(out) > 0:
            continue
        out = _unit(out)
        frame = np.stack([out, tree.up, np.cross(tree.up, out)])  # rows: local s, h, d axes
        # medial walls as (normal, offset): this lung keeps (x - carina) . normal <= offset
        cuts = [(n, c - w) for (a, _), (n, c, w) in walls.items() if a == i]
        cuts += [(-n, -c - w) for (_, b), (n, c, w) in walls.items() if b == i]
        # the lung has to reach its walls, so its points' shadows on them must be inside too
        relative = lobe.points - tree.carina
        shadows = [lobe.points - (relative @ n - c)[:, None] * n for n, c in cuts]
        points = np.vstack([lobe.points, *shadows])
        reach = np.concatenate([lobe.reach + clearance, np.zeros(len(lobe.points) * len(cuts))])
        egg = _fit_egg((points - tree.carina) @ frame.T, reach)
        count = 3 if len(tree.lobes) == 2 and i == 0 else 2
        sections += _lung_lobes(egg, tree, frame, cuts, count, fissure, resolution)
    if not sections:
        return None
    return surface(pv.merge(sections, merge_points=False))


def _separating_plane(tree: Tree, lobe: Lobe, other: Lobe) -> tuple[np.ndarray, float, float] | None:
    """Upright plane between two branches as (normal, offset, room): (x - carina) . normal = offset.

    The normal is perpendicular to the trachea axis and points from ``lobe`` to ``other``. Of the
    directions within 60 degrees of the line between the branch centres, the one with the most
    room between the two branches (or, if they interleave, the least overlap, then room < 0) is
    taken, and the plane sits in the middle of that room. For a mirror-symmetric tree this is the
    plane through the carina that contains the trachea axis.
    """
    base = _perp(other.centre - lobe.centre, tree.up)
    if not np.linalg.norm(base) > 0:
        return None
    base = _unit(base)
    side = np.cross(tree.up, base)
    near, far = lobe.points - tree.carina, other.points - tree.carina
    best = None
    for degrees in sorted(range(-60, 61, 2), key=abs):  # straight across first: wins ties
        angle = np.radians(degrees)
        normal = np.cos(angle) * base + np.sin(angle) * side
        lobe_ends = (near @ normal + lobe.reach).max()  # how far the branch reaches towards the other
        other_starts = (far @ normal - other.reach).min()
        room = other_starts - lobe_ends
        if best is None or room > best[0] + 1e-9 * np.abs(room):
            best = (room, normal, 0.5 * (lobe_ends + other_starts))
    return best[1], float(best[2]), float(best[0])


def _fit_egg(local: np.ndarray, reach: np.ndarray) -> Egg:
    """Egg centred beside the trachea axis that contains every point with its reach.

    It also reaches across the axis; the medial walls and the trachea carve cut that part off.
    """
    s, h, d = local.T
    top, bottom, outer = (h + reach).max(), (h - reach).min(), (s + reach).max()
    egg = Egg(
        s_centre=0.5 * outer,
        h_centre=0.5 * (top + bottom),
        a=0.62 * outer,
        b=0.5 * (top - bottom),
        c=max(0.4 * outer, (np.abs(d) + reach).max()),
    )
    corners = np.array(list(itertools.product((-1.0, 1.0), repeat=3))) / np.sqrt(3)
    directions = np.vstack([np.zeros(3), np.eye(3), -np.eye(3), corners])
    test = (local[:, None, :] + reach[:, None, None] * directions[None]).reshape(-1, 3)
    for _ in range(50):
        worst = egg.q(*test.T).max()
        if worst <= 1:
            break
        grow = np.sqrt(worst) * 1.01
        egg = replace(egg, a=egg.a * grow, b=egg.b * grow, c=egg.c * grow)
    return egg


def _lung_lobes(
    egg: Egg, tree: Tree, frame: np.ndarray, cuts: list, count: int, fissure: float, resolution: int
) -> list[pv.PolyData]:
    """The lung (egg minus the trachea space, cut at its medial walls), split into lobes.

    Fissures run through the centre of the lung: the oblique one rises towards the back and
    separates the lower lobe; with three lobes a horizontal one separates the upper from the
    middle lobe.
    """
    pad = 0.1 * max(egg.a, egg.b, egg.c)
    wide = 1 + egg.taper
    # the grid must hold the whole egg (it reaches across the trachea axis): a contour that runs
    # into the grid boundary would leave a hole; the cuts below remove what is not wanted
    lo = np.array([egg.s_centre - egg.a * wide - pad, egg.h_centre - egg.b - pad, -egg.c * wide - pad])
    hi = np.array([egg.s_centre + egg.a * wide + pad, egg.h_centre + egg.b + pad, egg.c * wide + pad])
    spacing = (hi - lo).max() / resolution
    dims = np.ceil((hi - lo) / spacing).astype(int) + 1
    grid = pv.ImageData(dimensions=dims, spacing=(spacing,) * 3, origin=lo)
    local = np.asarray(grid.points)
    s, h, d = local.T
    body = (np.sqrt(egg.q(s, h, d)) - 1) * min(egg.a, egg.b, egg.c)  # about a distance, < 0 inside
    # keep the trachea (from the carina up to the inlet) and the carina sphere outside
    trachea_length = float(np.linalg.norm(tree.inlet - tree.carina))
    to_trachea = np.hypot(np.hypot(s, d), h - np.clip(h, 0.0, trachea_length))
    body = np.maximum(body, 1.15 * tree.carina_radius + fissure - to_trachea)
    relative = local @ frame  # position relative to the carina
    for normal, offset in cuts:
        body = np.maximum(body, relative @ normal - offset)
    centre = np.array([egg.s_centre, egg.h_centre, 0.0]) @ frame
    tilt = np.radians(OBLIQUE_TILT)
    oblique = (relative - centre) @ (np.cos(tilt) * tree.up + np.sin(tilt) * tree.anterior)
    horizontal = (relative - centre) @ tree.up - HORIZONTAL_LEVEL * egg.b
    g = 0.5 * fissure
    if count == 3:  # upper, middle, lower
        limits = [(g - oblique, g - horizontal), (g - oblique, horizontal + g), (oblique + g,)]
    else:  # upper, lower
        limits = [(g - oblique,), (oblique + g,)]
    lobes = []
    for bounds in limits:
        grid["field"] = np.max(np.vstack([body, *bounds]), axis=0)
        lobe = grid.contour([0.0], scalars="field")
        if lobe.n_points == 0:
            continue
        lobe.clear_data()
        lobe.points = tree.carina + np.asarray(lobe.points) @ frame
        lobes.append(lobe.smooth_taubin(n_iter=15, pass_band=0.1).compute_normals(auto_orient_normals=True))
    return lobes


def _perp(vector: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Component of a vector perpendicular to a unit axis."""
    return vector - vector.dot(axis) * axis


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)
