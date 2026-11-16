"""Study data and importers for 4C results and QUEENS metadata.

The only part of lungscope that opens files. It imports neither ``analysis`` nor ``views``.

Public interface
- ``load_study(study_json) -> xarray.Dataset``: reads ``study.json`` (the index the supplied
  ``run_uq.py`` writes), each completed sample's input mesh and its 4C result series, matches the
  result cells to the input cells by their endpoints, and checks geometry and output times.
  Dimensions: realization (sample id), time (s), entity (element id), plus coefficient, end, xyz.

  - per entity: the input arrays that are the same in every sample (block_id, generation,
    radius, v0, ...) and ``endpoints`` (entity, end, xyz); end 0 is where 4C reports p_1, q_in;
  - outputs (realization, time, entity): every 4C result array (p_1, p_2, q_in, q_out,
    volume, v_0, ...);
  - inputs that differ between samples (realization, entity), e.g. elasticity_e;
  - ``xi`` (realization, coefficient): the random-field coefficients;
  - numeric results recorded in study.json (realization), e.g. the QUEENS response.

  NaN means "not defined for this element"; units are in each variable's ``units`` attribute.
- ``load_mesh(path) -> pyvista.UnstructuredGrid``: one mesh file; a multi-block file is merged.
"""

from lungscope.data.mesh_file import load_mesh
from lungscope.data.study import load_study

__all__ = ["load_mesh", "load_study"]
