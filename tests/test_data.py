"""Result cells are matched to input cells by both endpoints, and a study is loaded from its files."""

import json
from pathlib import Path

import numpy as np
import pytest
import pyvista as pv

from lungscope.data import load_study
from lungscope.data.study import match_cells

# three line cells: a trachea and two branches
ENDS = np.array(
    [[[0, 0, 0], [0, 0, 1]], [[0, 0, 1], [1, 0, 2]], [[0, 0, 1], [-1, 0, 2]]],
    dtype=float,
)


def line_cells(arrays: dict[str, np.ndarray]) -> pv.UnstructuredGrid:
    """The three cells of ENDS with two points each, as 4C writes them, and the given cell arrays."""
    cells = np.array([[2, 2 * i, 2 * i + 1] for i in range(len(ENDS))]).ravel()
    grid = pv.UnstructuredGrid(cells, np.full(len(ENDS), pv.CellType.LINE, dtype=np.uint8), ENDS.reshape(-1, 3))
    for name, values in arrays.items():
        grid.cell_data[name] = values
    return grid


def write_study(folder: Path, inputs: dict[str, np.ndarray], steps: list[dict[str, np.ndarray]]) -> Path:
    """A one-sample study as run_uq.py writes it: study.json, the input mesh and a 4C result series."""
    sample = folder / "0"
    (sample / "result-vtk-files").mkdir(parents=True)
    line_cells(inputs).save(sample / "lung.vtu")
    entries = []
    for k, arrays in enumerate(steps):
        file = f"result-vtk-files/reduced_lung-{k:05d}.vtu"
        line_cells(arrays).save(sample / file)
        entries.append(f'<DataSet timestep="{0.1 * (k + 1)}" group="" part="0" file="{file}"/>')
    (sample / "result-reduced_lung.pvd").write_text(
        '<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n<Collection>\n'
        + "\n".join(entries)
        + "\n</Collection>\n</VTKFile>\n"
    )
    record = {"sample_id": 0, "status": "complete", "parameters": {"xi_000": 0.5}}
    record |= {"input_vtu": "0/lung.vtu", "result_pvd": "0/result-reduced_lung.pvd"}
    (folder / "study.json").write_text(json.dumps({"samples": [record]}))
    return folder / "study.json"


def test_an_input_array_that_4c_writes_as_well_is_taken_from_the_results(tmp_path):
    v_0 = np.array([np.nan, 1.0, 2.0])  # PneumoGen meshes carry v_0, and 4C writes it too
    inputs = {"block_id": np.array([1, 2, 2]), "radius": np.array([1.0, 0.0, 0.0]), "v_0": v_0}
    steps = [{"p_1": np.array([3.0, 2.0, 2.0]) + k, "v_0": v_0} for k in range(2)]
    study = load_study(write_study(tmp_path, inputs, steps))
    assert study["v_0"].dims == ("realization", "time", "entity")
    np.testing.assert_array_equal(study["v_0"].values[0, 1], v_0)
    assert study["v_0"].attrs["units"] == "mm^3"
    assert study["radius"].dims == ("entity",)  # the other input arrays stay per element
    np.testing.assert_array_equal(study["p_1"].values[0, :, 0], [3.0, 4.0])


def test_same_order():
    assert match_cells(ENDS, ENDS.copy(), scale=2.0).tolist() == [0, 1, 2]


def test_reordered_cells():
    result = ENDS[[2, 0, 1]]  # 4C may write the cells in another order
    assert match_cells(ENDS, result, scale=2.0).tolist() == [1, 2, 0]


def test_swapped_ends_are_refused():
    result = ENDS.copy()
    result[1] = result[1][::-1]
    with pytest.raises(ValueError, match="swapped"):
        match_cells(ENDS, result, scale=2.0)


def test_missing_and_unknown_cells_are_refused():
    with pytest.raises(ValueError):
        match_cells(ENDS, ENDS[:2], scale=2.0)
    moved = ENDS.copy()
    moved[0, 1, 2] = 5.0
    with pytest.raises(ValueError, match="no result cell"):
        match_cells(ENDS, moved, scale=2.0)
