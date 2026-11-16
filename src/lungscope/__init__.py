"""Interactive visualization of reduced lung simulations.

The code is in three layers, one subpackage each with its public interface documented in its
``__init__``, and ``app.py`` on top (see docs/project-guide.md):

- ``data``: reads the study outputs; the only code that opens files. A study (``study.json``
  and the VTK files it names) becomes one ``xarray.Dataset``, a single mesh file a PyVista grid.
- ``analysis``: statistics on that dataset, such as ensemble means, standard deviations and the
  quantities that can be shown over time. Independent of trame and rendering; no file access.
- ``views``: the 3D scene and the plots, drawn for the selections they are given. They never
  open files and never import ``data``.
- ``app.py``: the command line, the trame layout, and the selections it coordinates between the
  views (coloring, time step, clicked element).

Dependencies point one way: ``views`` may use ``analysis``, ``data`` uses neither, and only
``app.py`` uses all three.
"""
