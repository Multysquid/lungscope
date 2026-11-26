"""Lobe masks of a lung model (NIfTI) and where they sit on its mesh.

A lobe folder holds one binary mask per lobe, ``lobe_*.nii.gz`` (non-zero inside the lobe), in
the coordinates of the image they were segmented from (for a CT: scanner coordinates in mm), and
``lobes.json``, which places them on the lung model's mesh:

    {"mesh_from_world": [[...], [...], [...], [0, 0, 0, 1]],   4 x 4, rows
     "fit": {...}}                                              how it was found

``mesh_from_world`` maps image coordinates to mesh coordinates. PneumoGen trees come without it;
``lungscope-fit-lobes`` fits one (``analysis.fit_lobes``), and a known transform can replace it.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkIOImage import vtkNIFTIImageReader

MASK_PATTERN = "lobe_*.nii.gz"
TRANSFORM_FILE = "lobes.json"
MARGIN = 2  # empty voxels kept around each lobe when it is cropped


@dataclass(frozen=True)
class Lobe:
    """One lobe mask, cropped to the lobe with a margin of empty voxels."""

    name: str  # the file name without .nii.gz, e.g. "lobe_0"
    # bool, indexed (i, j, k) as VTK stores the image: for a mirrored image the slices may run in
    # reverse; world_from_index always belongs to these indices
    mask: np.ndarray
    world_from_index: np.ndarray  # 4 x 4: image coordinates (mm) of the centre of voxel (i, j, k)


@dataclass(frozen=True)
class LobeMasks:
    """The lobes of a lobe folder and, if lobes.json is there, where they sit on the mesh."""

    folder: Path
    lobes: tuple[Lobe, ...]
    mesh_from_world: np.ndarray | None = None  # 4 x 4 from lobes.json
    fit: dict = field(default_factory=dict)  # how mesh_from_world was found, as lobes.json says

    def mesh_from_index(self, lobe: Lobe) -> np.ndarray:
        """4 x 4: mesh coordinates of the centre of a lobe's voxel (i, j, k)."""
        if self.mesh_from_world is None:
            raise ValueError(f"{self.folder}: no {TRANSFORM_FILE}, so the lobes have no place on the mesh")
        return self.mesh_from_world @ lobe.world_from_index


def load_lobes(folder: str | Path, placed: bool = True) -> LobeMasks:
    """Read the lobe masks of a folder (in file name order) and its lobes.json.

    With ``placed`` (the default) lobes.json must be there; it is read before the masks, so that a
    missing one is reported at once. Without, a missing lobes.json leaves ``mesh_from_world`` None.
    """
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"{folder}: no such folder")
    files = sorted(folder.glob(MASK_PATTERN))
    if not files:
        raise FileNotFoundError(f"{folder}: no lobe masks ({MASK_PATTERN})")
    transform = folder / TRANSFORM_FILE
    mesh_from_world, fit = None, {}
    if transform.is_file():
        mesh_from_world, fit = read_lobe_transform(transform)
    elif placed:
        raise FileNotFoundError(
            f"{transform} is missing: it places the lobes on the mesh (lungscope-fit-lobes fits one)"
        )
    lobes = tuple(read_mask(path) for path in files)
    return LobeMasks(folder, lobes, mesh_from_world, fit)


def read_mask(path: Path) -> Lobe:
    """One NIfTI mask, cropped to its non-zero voxels.

    The image coordinates come from the sform, or the qform without one, or the voxel spacing
    alone without either.
    """
    reader = vtkNIFTIImageReader()
    reader.SetFileName(str(path))
    reader.Update()
    image = reader.GetOutput()
    dims = image.GetDimensions()
    if image.GetPointData().GetScalars() is None or 0 in dims:
        raise ValueError(f"{path}: not a readable NIfTI image")
    # VTK stores x fastest: reshaped (k, j, i), transposed to (i, j, k)
    inside = (vtk_to_numpy(image.GetPointData().GetScalars()) != 0).reshape(dims[::-1]).transpose()
    if not inside.any():
        raise ValueError(f"{path}: the mask is empty")
    # VTK's point (x, y, z) = origin + spacing * (i, j, k); the s/qform maps that point to the image
    # coordinates
    form = reader.GetSFormMatrix() or reader.GetQFormMatrix()
    world_from_vtk = np.eye(4)
    if form is not None:
        world_from_vtk = np.array([[form.GetElement(row, col) for col in range(4)] for row in range(4)])
    vtk_from_index = np.diag([*image.GetSpacing(), 1.0])
    vtk_from_index[:3, 3] = image.GetOrigin()
    lo, hi = [], []
    for axis in range(3):
        filled = np.flatnonzero(inside.any(axis=tuple(a for a in range(3) if a != axis)))
        lo.append(max(int(filled[0]) - MARGIN, 0))
        hi.append(min(int(filled[-1]) + MARGIN + 1, inside.shape[axis]))
    crop = inside[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]].copy()
    shift = np.eye(4)
    shift[:3, 3] = lo
    name = path.name.removesuffix(".gz").removesuffix(".nii")
    return Lobe(name, crop, world_from_vtk @ vtk_from_index @ shift)


def read_lobe_transform(path: Path) -> tuple[np.ndarray, dict]:
    """``mesh_from_world`` and ``fit`` of a lobes.json; the matrix must be an invertible affine map."""
    content = json.loads(Path(path).read_text())
    try:
        matrix = np.array(content["mesh_from_world"], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{path}: needs mesh_from_world, a 4 x 4 matrix") from error
    if matrix.shape != (4, 4) or not np.allclose(matrix[3], [0, 0, 0, 1]):
        raise ValueError(f"{path}: mesh_from_world must be a 4 x 4 matrix with last row 0 0 0 1")
    if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix[:3, :3])) < 1e-12:
        raise ValueError(f"{path}: mesh_from_world is not invertible")
    return matrix, content.get("fit", {})


def save_lobe_transform(folder: str | Path, mesh_from_world: np.ndarray, fit: dict) -> Path:
    """Write lobes.json into a lobe folder; returns its path."""
    path = Path(folder).expanduser().resolve() / TRANSFORM_FILE
    content = {"mesh_from_world": np.asarray(mesh_from_world, dtype=float).tolist(), "fit": fit}
    path.write_text(json.dumps(content, indent=2) + "\n")
    return path
