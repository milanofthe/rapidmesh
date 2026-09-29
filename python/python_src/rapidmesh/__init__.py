"""rapidmesh: pure-Rust conforming tetrahedral mesher for EM FEM.

.. code-block:: python

    import rapidmesh as rm

    g = rm.Geometry(maxh=0.9)
    air = g.box(4, 4, 4)
    diel = g.box(2, 2, 1, position=(1, 1, 1), maxh=0.45)
    mesh = g.mesh()
"""

from .geometry import (Geometry, Mesh, SurfaceMesh, Solid, Sheet, Spline, Mesh2D, Region2D,
                       load_msh,
                       mesh_2d, mesh_layers, union_regions, overlay_regions)
from . import adapt
from .adapt import dorfler_mark, refine_dorfler
from ._native import set_log_level

try:
    from importlib.metadata import version as _version
    __version__ = _version("rapidmesh")
except Exception:  # not installed as a package (a source checkout)
    __version__ = "0.0.0"
__all__ = [
    "Geometry",
    "Mesh",
    "SurfaceMesh",
    "Solid",
    "Sheet",
    "load_msh",
    "Spline",
    "mesh_2d",
    "mesh_layers",
    "union_regions",
    "overlay_regions",
    "Mesh2D",
    "Region2D",
    "adapt",
    "dorfler_mark",
    "refine_dorfler",
    "set_log_level",
]
