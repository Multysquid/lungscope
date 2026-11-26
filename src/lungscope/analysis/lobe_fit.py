"""Fit where lobe masks sit on a lung tree: the transform from image to mesh coordinates.

The lobe masks that come with a PneumoGen tree are in the coordinates of the image they were
segmented from (a CT's scanner coordinates, adult size); the tree is in other coordinates and of
another size, and the transform between the two is not given. ``fit_lobes`` looks for the affine
transform that puts the terminal units inside the lobes and lets them fill the lobes.

Method (iterative closest points, on the masks coarsened by ``step``):
- starts: each of the 48 axis orders and sign flips, scaled so that the spread of the units
  matches the spread of the lobes, and shifted onto the lobes' centre;
- every step pairs each unit outside the lobes with the nearest lobe boundary point, each of a
  sample of points inside the lobes with the nearest unit (so that the units fill the lobes
  instead of shrinking into them), and each unit inside with where it is; then it solves for the
  weighted least-squares transform;
- first a similarity transform (rotation, one scale, shift; mirrored if the start is) from every
  start, on a subset of the units; then from the best starts, with all units, a similarity and
  finally an affine transform. The transform that puts the most units inside a lobe wins; this,
  and the fractions reported, are counted on the full masks.
"""

import itertools
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

FILL_SAMPLES = 15_000  # points inside the lobes that pull their nearest unit towards them
FILL_WEIGHT = 0.2  # their total weight, per unit
ANCHOR_WEIGHT = 0.5  # weight of a unit inside a lobe staying where it is
START_UNITS = 6_000  # units used to compare the starts
START_STEPS = 20
REFINE_STEPS = 60
BEST_STARTS = 3
CONVERGED = 1e-4  # stop when no unit moves by more than this fraction of the units' extent


@dataclass(frozen=True)
class LobeFit:
    """The fitted transform and how well it fits."""

    mesh_from_world: np.ndarray  # 4 x 4 affine map from image to mesh coordinates
    inside: float  # fraction of the units inside a lobe
    similarity_inside: float  # the same with the best similarity transform
    per_lobe: tuple[int, ...]  # units inside each lobe


def terminal_unit_centres(endpoints: np.ndarray, reference_volume: np.ndarray) -> np.ndarray:
    """Midpoints (n, 3) of the terminal units: the elements with a positive reference volume."""
    v0 = np.asarray(reference_volume, dtype=float)
    units = np.isfinite(v0) & (v0 > 0)
    return np.asarray(endpoints, dtype=float)[units].mean(axis=1)


def fit_lobes(
    units: np.ndarray,
    masks: Sequence[np.ndarray],
    world_from_index: Sequence[np.ndarray],
    step: int = 2,
    seed: int = 0,
) -> LobeFit:
    """The affine transform from image to mesh coordinates that puts the units into the lobes.

    ``units``: terminal-unit positions (n, 3) in mesh coordinates; ``masks``: one boolean voxel
    mask per lobe, indexed (i, j, k); ``world_from_index``: 4 x 4 per mask, the image coordinates
    of the centre of voxel (i, j, k).
    """
    units = np.asarray(units, dtype=float)
    rng = np.random.default_rng(seed)
    lobes = _Lobes(masks, world_from_index, step, rng)
    subset = units[rng.choice(len(units), min(len(units), START_UNITS), replace=False)]
    scale = np.sqrt(lobes.spread / np.trace(np.cov(units.T)))
    starts = []
    for order in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            flip = np.zeros((3, 3))
            flip[range(3), order] = signs
            linear = scale * flip
            shift = lobes.centre - linear @ units.mean(axis=0)
            linear, shift = _icp(subset, linear, shift, lobes, "similarity", START_STEPS)
            starts.append((lobes.inside(_map(subset, linear, shift)), linear, shift))
    starts.sort(key=lambda start: -start[0])
    full = _Labels(masks, world_from_index, step=1)  # what is reported is counted on the full masks
    best_similarity, best = 0.0, None
    for _, linear, shift in starts[:BEST_STARTS]:
        linear, shift = _icp(units, linear, shift, lobes, "similarity", REFINE_STEPS)
        best_similarity = max(best_similarity, full.inside(_map(units, linear, shift)))
        linear, shift = _icp(units, linear, shift, lobes, "affine", REFINE_STEPS)
        inside = full.inside(_map(units, linear, shift))
        if best is None or inside > best[0]:
            best = (inside, linear, shift)
    inside, linear, shift = best
    world_from_mesh = np.eye(4)
    world_from_mesh[:3, :3], world_from_mesh[:3, 3] = linear, shift
    label = full.lobe_of(_map(units, linear, shift))
    per_lobe = tuple(int(np.count_nonzero(label == k + 1)) for k in range(len(full.masks)))
    return LobeFit(np.linalg.inv(world_from_mesh), float(inside), float(best_similarity), per_lobe)


class _Labels:
    """Which lobe a point is in, on the masks taken every ``step`` voxels."""

    def __init__(self, masks, world_from_index, step: int) -> None:
        self.masks, self.world_from_index, self.index_from_world = [], [], []
        for mask, matrix in zip(masks, world_from_index, strict=True):
            matrix = np.asarray(matrix, dtype=float) @ np.diag([step, step, step, 1.0])
            self.masks.append(np.asarray(mask, dtype=bool)[::step, ::step, ::step])
            self.world_from_index.append(matrix)
            self.index_from_world.append(np.linalg.inv(matrix))

    def lobe_of(self, world: np.ndarray) -> np.ndarray:
        """Which lobe each point is in: 1, 2, ... in mask order; 0 for none."""
        label = np.zeros(len(world), dtype=int)
        for k, (mask, matrix) in enumerate(zip(self.masks, self.index_from_world, strict=True)):
            index = np.rint(_map(world, matrix[:3, :3], matrix[:3, 3])).astype(int)
            fits = np.all((index >= 0) & (index < mask.shape), axis=1) & (label == 0)
            hit = np.zeros(len(world), dtype=bool)
            hit[fits] = mask[tuple(index[fits].T)]
            label[hit] = k + 1
        return label

    def inside(self, world: np.ndarray) -> float:
        return float(np.mean(self.lobe_of(world) > 0))


class _Lobes(_Labels):
    """The coarsened masks: which lobe a point is in, boundary points and points inside."""

    def __init__(self, masks, world_from_index, step: int, rng: np.random.Generator) -> None:
        super().__init__(masks, world_from_index, step)
        boundary, inside, volume = [], [], []
        for mask, matrix in zip(self.masks, self.world_from_index, strict=True):
            padded = np.pad(mask, 1)
            core = mask.copy()  # voxels whose six neighbours are inside too
            for axis in range(3):
                for direction in (-1, 1):
                    core &= np.roll(padded, direction, axis)[1:-1, 1:-1, 1:-1]
            boundary.append(_points(np.argwhere(mask & ~core), matrix))
            filled = np.argwhere(mask)
            inside.append(_points(filled, matrix))
            volume.append(np.full(len(filled), abs(np.linalg.det(matrix[:3, :3]))))
        if not any(len(points) for points in inside):
            raise ValueError("the lobe masks are empty")
        self.boundary = np.vstack(boundary)
        self.boundary_tree = cKDTree(self.boundary)
        inside, volume = np.vstack(inside), np.concatenate(volume)
        weights = volume / volume.sum()
        self.centre = weights @ inside
        self.spread = float(np.sum(weights @ (inside - self.centre) ** 2))  # trace of the covariance
        count = min(FILL_SAMPLES, len(inside))
        self.fill = inside[rng.choice(len(inside), count, replace=False, p=weights)]


def _points(index: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return _map(index, matrix[:3, :3], matrix[:3, 3])


def _map(points: np.ndarray, linear: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """linear @ point + shift for every point (n, 3).

    The products of the fit go through einsum, not BLAS: on many cores BLAS spreads each of these
    small products over threads, which made the fit three times slower.
    """
    return np.einsum("ij,nj->ni", linear, points) + shift


def _icp(units, linear, shift, lobes: _Lobes, kind: str, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Improve world = linear @ unit + shift by iterative closest points (see the module)."""
    extent = float(np.ptp(units, axis=0).max()) or 1.0
    corners = np.array(list(itertools.product(*zip(units.min(axis=0), units.max(axis=0)))))
    handedness = np.sign(np.linalg.det(linear))
    fill_weight = FILL_WEIGHT * len(units) / len(lobes.fill)
    for _ in range(steps):
        world = _map(units, linear, shift)
        outside = lobes.lobe_of(world) == 0
        _, nearest_boundary = lobes.boundary_tree.query(world[outside])
        _, nearest_unit = cKDTree(world).query(lobes.fill)
        source = np.vstack([units[outside], units[nearest_unit], units[~outside]])
        target = np.vstack([lobes.boundary[nearest_boundary], lobes.fill, world[~outside]])
        weight = np.concatenate(
            [
                np.ones(np.count_nonzero(outside)),
                np.full(len(nearest_unit), fill_weight),
                np.full(np.count_nonzero(~outside), ANCHOR_WEIGHT),
            ]
        )
        new_linear, new_shift = _solve(source, target, weight, kind, handedness)
        moved = np.abs(_map(corners, new_linear - linear, new_shift - shift)).max()
        linear, shift = new_linear, new_shift
        if moved < CONVERGED * extent * np.cbrt(abs(np.linalg.det(linear))):
            break
    return linear, shift


def _solve(source, target, weight, kind: str, handedness: float) -> tuple[np.ndarray, np.ndarray]:
    """Weighted least squares: target ~ linear @ source + shift.

    "affine": any linear map; "similarity": scale times a rotation (a reflection if
    ``handedness`` is negative), as in Umeyama's method.
    """
    weight = weight / weight.sum()
    source_centre, target_centre = np.einsum("n,ni->i", weight, source), np.einsum("n,ni->i", weight, target)
    s, t = source - source_centre, target - target_centre
    cross = np.einsum("n,ni,nj->ij", weight, s, t)  # see _map for why not BLAS
    if kind == "affine":
        linear = np.linalg.solve(np.einsum("n,ni,nj->ij", weight, s, s), cross).T
    else:
        u, sigma, vt = np.linalg.svd(cross)
        signs = np.ones(3)
        signs[2] = handedness * np.sign(np.linalg.det(vt.T @ u.T))
        rotation = vt.T @ np.diag(signs) @ u.T
        linear = (sigma @ signs) / np.einsum("n,ni,ni->", weight, s, s) * rotation
    return linear, target_centre - linear @ source_centre
