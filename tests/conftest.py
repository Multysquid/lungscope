"""Lobe masks for the tests: two ellipsoids, and NIfTI files written by hand after the NIfTI-1 standard."""

import gzip
import struct
from pathlib import Path

import numpy as np
import pytest

SHAPE = (48, 40, 36)
# image coordinates (mm) of voxel (i, j, k): x and y run backwards, as a CT scanner writes them
WORLD_FROM_INDEX = np.array(
    [[-1.5, 0.0, 0.0, 120.0], [0.0, -1.5, 0.0, 90.0], [0.0, 0.0, 2.0, 500.0], [0.0, 0.0, 0.0, 1.0]]
)


def ellipsoid(centre, radii, shape=SHAPE) -> np.ndarray:
    index = np.indices(shape).reshape(3, -1).T
    return (np.sum(((index - np.asarray(centre)) / np.asarray(radii)) ** 2, axis=1) <= 1).reshape(shape)


def lobe_masks() -> list[np.ndarray]:
    """A big and a small lobe that touch; together they have no symmetry."""
    big = ellipsoid((16, 20, 18), (12, 14, 15))
    small = ellipsoid((36, 14, 22), (8, 7, 10)) & ~big
    return [big, small]


def write_nifti(path: Path, mask: np.ndarray, world_from_index: np.ndarray) -> None:
    """A gzipped NIfTI-1 file of a uint8 mask whose sform (srow) is world_from_index."""
    header = bytearray(348)
    struct.pack_into("<i", header, 0, 348)  # sizeof_hdr
    struct.pack_into("<8h", header, 40, 3, *mask.shape, 1, 1, 1, 1)  # dim
    struct.pack_into("<hh", header, 70, 2, 8)  # datatype uint8, bitpix
    spacing = np.linalg.norm(world_from_index[:3, :3], axis=0)
    struct.pack_into("<8f", header, 76, 1.0, *spacing, 1.0, 1.0, 1.0, 1.0)  # pixdim (qfac first)
    struct.pack_into("<f", header, 108, 352.0)  # vox_offset: header and an empty extension flag
    struct.pack_into("<ff", header, 112, 1.0, 0.0)  # scl_slope, scl_inter
    header[123] = 2  # xyzt_units: mm
    struct.pack_into("<hh", header, 252, 0, 1)  # qform_code 0, sform_code 1
    struct.pack_into("<12f", header, 280, *np.asarray(world_from_index[:3], dtype=float).ravel())  # srow_x/y/z
    header[344:348] = b"n+1\0"
    voxels = np.asarray(mask, dtype=np.uint8).ravel(order="F").tobytes()  # i runs fastest
    path.write_bytes(gzip.compress(bytes(header) + bytes(4) + voxels))


@pytest.fixture
def lobe_folder(tmp_path) -> Path:
    """A folder with lobe_0.nii.gz and lobe_1.nii.gz (see lobe_masks), without lobes.json."""
    folder = tmp_path / "lobes"
    folder.mkdir()
    for k, mask in enumerate(lobe_masks()):
        write_nifti(folder / f"lobe_{k}.nii.gz", mask, WORLD_FROM_INDEX)
    return folder
