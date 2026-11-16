"""Read a single mesh file, e.g. a 4C input .vtu, for ``lungscope --mesh``.
"""

from pathlib import Path

import pyvista as pv


def load_mesh(path: Path) -> pv.UnstructuredGrid:
    """Read a mesh file; a multi-block file is merged into one grid."""
    data = pv.read(path)
    if isinstance(data, pv.MultiBlock):
        data = data.combine()
    if not isinstance(data, pv.UnstructuredGrid):
        data = data.cast_to_unstructured_grid()
    return data
