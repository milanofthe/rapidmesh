"""rapidmesh: pure-Rust conforming tetrahedral mesher for EM FEM.

.. code-block:: python

    import rapidmesh as rm

    g = rm.Geometry(maxh=0.9)
    air = g.box(4, 4, 4)
    diel = g.box(2, 2, 1, position=(1, 1, 1), maxh=0.45)
    mesh = g.mesh()
"""

from .geometry import Geometry, Mesh, SurfaceMesh, Solid
from . import adapt
from .adapt import dorfler_mark, refine_dorfler

__version__ = "0.1.0"
__all__ = ["Geometry", "Mesh", "SurfaceMesh", "Solid", "adapt", "dorfler_mark", "refine_dorfler"]
