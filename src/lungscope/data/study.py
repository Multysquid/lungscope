"""Load a forward-UQ study (QUEENS + 4C) into one labelled dataset.

Reads ``study.json`` (the index the supplied ``run_uq.py`` writes), each completed sample's input
mesh and its 4C result series, links them, checks the links and returns an ``xarray.Dataset``:

    dims    realization (sample id), time (s), entity (element id); coefficient, end, xyz
    coords  per entity: the input cell arrays that are the same in every sample (block_id,
            generation, radius, v0, ...) and ``endpoints`` (entity, end, xyz). End 0 is the
            element's first point, where 4C reports p_1 and q_in (the side toward the inlet).
            per realization: status and the file paths from study.json
    data    every cell array of the 4C results          (realization, time, entity)
            input cell arrays that differ between samples (realization, entity), e.g. elasticity_e
            the random-field coefficients               (realization, coefficient)
            numeric results recorded in study.json      (realization), e.g. the QUEENS response

NaN means "not defined for this element": 4C writes NaN, e.g. volume on airways. Only samples
with status "complete" are loaded. Result cells are matched to input cells by their two
endpoints, because 4C may reorder cells and writes no ids. An input array that 4C writes as well
(``v_0`` in PneumoGen meshes) is taken from the results.
"""

import json
from pathlib import Path

import numpy as np
import pyvista as pv
import xarray as xr

UNITS = {
    "p_1": "Pa",
    "p_2": "Pa",
    "q_in": "mm^3/s",
    "q_out": "mm^3/s",
    "volume": "mm^3",
    "v_0": "mm^3",
    "v0": "mm^3",
    "elasticity_e": "Pa",
    "elastic_pressure": "Pa",
    "radius": "mm",
    "endpoints": "mm",
    "time": "s",
}
SAMPLE_KEYS = {"sample_id", "parameters", "status", "input_vtu", "result_pvd", "error"}
TIME_TOLERANCE = 1e-9  # s; 4C writes times like 0.9400000000000001


def load_study(study_json: str | Path) -> xr.Dataset:
    """Read study.json and every file it names, link and check them, return the dataset."""
    path = Path(study_json).expanduser().resolve()
    index = json.loads(path.read_text())
    samples = [s for s in index.get("samples", []) if s.get("status") == "complete"]
    if not samples:
        raise ValueError(f"{path}: no sample with status 'complete'")
    folder = path.parent

    inputs = [read_input(folder / s["input_vtu"]) for s in samples]
    ends = inputs[0][0]
    for (other, _), sample in zip(inputs, samples, strict=True):
        if other.shape != ends.shape or not np.array_equal(other, ends):
            raise ValueError(f"sample {sample['sample_id']}: the geometry differs from the first sample")
    scale = float(np.ptp(ends.reshape(-1, 3), axis=0).max()) or 1.0

    times, outputs = None, {}
    for sample in samples:
        series_times, fields = read_results(folder / sample["result_pvd"], ends, scale)
        if times is None:
            times = series_times
        elif series_times.shape != times.shape or not np.allclose(series_times, times, rtol=0, atol=TIME_TOLERANCE):
            raise ValueError(f"sample {sample['sample_id']}: output times differ from the first sample")
        if outputs and set(fields) != set(outputs):
            raise ValueError(f"sample {sample['sample_id']}: result arrays differ from the first sample")
        for name, values in fields.items():
            outputs.setdefault(name, []).append(values)

    first = inputs[0][1]
    entity = np.asarray(first["entity_id"]) if "entity_id" in first else np.arange(len(ends))
    if len(np.unique(entity)) != len(entity):
        raise ValueError("entity_id is not unique")

    coords = {
        "realization": [int(s["sample_id"]) for s in samples],
        "time": times,
        "entity": entity,
        "endpoints": (("entity", "end", "xyz"), ends),
        "status": ("realization", [s["status"] for s in samples]),
        "input_vtu": ("realization", [s["input_vtu"] for s in samples]),
        "result_pvd": ("realization", [s["result_pvd"] for s in samples]),
    }
    data_vars = {name: (("realization", "time", "entity"), np.stack(v)) for name, v in outputs.items()}
    for name in first:
        if name == "entity_id" or name in outputs:
            continue
        stacked = np.stack([arrays[name] for _, arrays in inputs])
        if all(np.array_equal(row, stacked[0], equal_nan=stacked.dtype.kind == "f") for row in stacked):
            coords[name] = ("entity", stacked[0])
        else:
            data_vars[name] = (("realization", "entity"), stacked)

    names = index.get("prior", {}).get("coefficient_names") or list(samples[0].get("parameters", {}))
    if names:
        coords["coefficient"] = names
        xi = [[float(s["parameters"][n]) for n in names] for s in samples]
        data_vars["xi"] = (("realization", "coefficient"), np.asarray(xi))
    for key in samples[0]:
        if key not in SAMPLE_KEYS and all(isinstance(s.get(key), int | float) for s in samples):
            data_vars[key] = ("realization", np.asarray([float(s[key]) for s in samples]))

    ds = xr.Dataset(data_vars, coords)
    for name, unit in UNITS.items():
        if name in ds.variables:
            ds[name].attrs["units"] = unit
    ds.attrs["source"] = str(path)
    for key in ("seed", "requested_samples"):
        if key in index:
            ds.attrs[key] = index[key]
    for key, value in index.get("prior", {}).items():
        if isinstance(value, int | float | str):
            ds.attrs[f"prior_{key}"] = value
    return ds


def read_input(vtu: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Endpoints (cells, 2, 3) and the 1-D cell arrays of an input mesh of line elements."""
    mesh = _grid(pv.read(vtu))
    arrays = {name: np.asarray(mesh.cell_data[name]) for name in mesh.cell_data}
    return cell_endpoints(mesh), {name: a for name, a in arrays.items() if a.ndim == 1}


def read_results(pvd: Path, ends: np.ndarray, scale: float) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Times (s) and the cell arrays of a 4C result series, as (time, entity) in the input order."""
    reader = pv.get_reader(str(pvd))
    times = np.asarray(reader.time_values, dtype=float)
    fields: dict[str, list[np.ndarray]] = {}
    order, last_ends = None, None
    for k in range(len(times)):
        reader.set_active_time_point(k)
        grid = _grid(reader.read())
        result_ends = cell_endpoints(grid)
        if last_ends is None or not np.array_equal(result_ends, last_ends):
            order = match_cells(ends, result_ends, scale)
            last_ends = result_ends
        for name in grid.cell_data:
            values = np.asarray(grid.cell_data[name])
            if values.ndim == 1:
                fields.setdefault(name, []).append(values[order])
    return times, {name: np.stack(v) for name, v in fields.items()}


def cell_endpoints(mesh: pv.UnstructuredGrid) -> np.ndarray:
    """First and last point of every cell, (cells, 2, 3)."""
    offsets = np.asarray(mesh.cell_offsets)
    connectivity = np.asarray(mesh.cell_connectivity)
    points = np.asarray(mesh.points, dtype=float)
    return np.stack([points[connectivity[offsets[:-1]]], points[connectivity[offsets[1:] - 1]]], axis=1)


def match_cells(ends: np.ndarray, result_ends: np.ndarray, scale: float) -> np.ndarray:
    """Index of the result cell for every input cell, by both endpoints; one-to-one or ValueError.

    A result cell whose endpoints are swapped is refused: which end p_1, p_2, q_in and q_out then
    belong to is not known.
    """
    if len(result_ends) != len(ends):
        raise ValueError(f"{len(result_ends)} result cells for {len(ends)} input cells")

    def keys(e: np.ndarray) -> list[tuple]:
        rounded = np.round(e / scale, 9) + 0.0  # + 0.0 turns -0.0 into 0.0
        return [(tuple(a), tuple(b)) for a, b in rounded]

    lookup = {key: i for i, key in enumerate(keys(result_ends))}
    order = np.empty(len(ends), dtype=int)
    for cell, (a, b) in enumerate(keys(ends)):
        if (a, b) in lookup:
            order[cell] = lookup[(a, b)]
        elif (b, a) in lookup:
            raise ValueError(f"result cell for input cell {cell} has its ends swapped")
        else:
            raise ValueError(f"no result cell has the endpoints of input cell {cell}")
    if len(np.unique(order)) != len(order):
        raise ValueError("several input cells match the same result cell")
    return order


def _grid(data) -> pv.UnstructuredGrid:
    """One unstructured grid from what a reader returns (a multi-block holds one grid per file)."""
    if isinstance(data, pv.MultiBlock):
        blocks = [block for block in data if block is not None]
        data = blocks[0] if len(blocks) == 1 else data.combine()
    if not isinstance(data, pv.UnstructuredGrid):
        data = data.cast_to_unstructured_grid()
    return data
