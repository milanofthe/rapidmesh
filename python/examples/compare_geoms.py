"""The comparison-showcase geometry registry: one geometry, three meshers.

Each :class:`CompareGeom` describes a single shape by its concrete dimensions
and gives two builders that produce the *same* geometry at the *same* target
size:

- ``build_rapidmesh()``  -> an unmeshed :class:`rapidmesh.Geometry`
- ``build_gmsh()``       -> builds into the live gmsh OCC model (the caller
  owns ``gmsh.initialize``/``finalize`` and the mesh-size options)

tetgen has no CAD kernel, so it is not listed here: the exporter feeds it
gmsh's *surface* triangulation of the same geometry (see ``compare_showcase``).

The set spans the capability range the landing page advertises: single
primitives (analytic curved surfaces), simple booleans (intersection curves),
mechanical parts (extrusion + bores), and one organic imported surface (a
metaball iso-surface meshed by all three from the same STL).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

FIXTURES = Path(__file__).resolve().parent / "fixtures"
BLOB_STL = FIXTURES / "blob.stl"


# ----------------------------------------------------------- shared profiles


def _gear_profile(n_teeth: int = 12, r_root: float = 1.0, r_tip: float = 1.25):
    """Closed tooth boundary polygon (xy), shared by both meshers so the gear
    outline is identical. Trapezoidal teeth, no keyway/lightening holes (a
    single central bore keeps the gmsh build tractable)."""
    pts: list[tuple[float, float]] = []
    T = 2 * math.pi / n_teeth
    for i in range(n_teeth):
        a = i * T
        for f, r in ((0.00, r_root), (0.32, r_root), (0.42, r_tip),
                     (0.68, r_tip), (0.78, r_root)):
            pts.append((r * math.cos(a + f * T), r * math.sin(a + f * T)))
    return pts


def _circle(n: int, r: float):
    return [(r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n))
            for i in range(n)]


# ------------------------------------------------------------------ blob STL


def ensure_blob_stl() -> Path:
    """Generate the organic blob iso-surface once (pyvista metaballs ->
    marching cubes) and cache it as ``fixtures/blob.stl``. Returns the path."""
    if BLOB_STL.exists():
        return BLOB_STL
    import pyvista as pv

    FIXTURES.mkdir(parents=True, exist_ok=True)
    # A few overlapping Gaussian blobs sampled on a grid, contoured at a level
    # that fuses them into one smooth organic body.
    n = 64
    lo, hi = -2.0, 2.0
    xs = np.linspace(lo, hi, n)
    grid = pv.ImageData(
        dimensions=(n, n, n),
        spacing=((hi - lo) / (n - 1),) * 3,
        origin=(lo, lo, lo),
    )
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    centers = [(-0.6, -0.4, 0.0, 0.9), (0.7, 0.3, 0.2, 1.0),
               (0.0, 0.7, -0.5, 0.7), (0.2, -0.7, 0.5, 0.6)]
    field = np.zeros_like(X)
    for cx, cy, cz, w in centers:
        field += w * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2) / 0.5)
    grid["f"] = field.flatten(order="F")
    surf = grid.contour([0.9], scalars="f").triangulate().clean()
    # Decimate to a web-friendly triangle count and keep it watertight.
    surf = surf.decimate(0.5).clean()
    surf.save(str(BLOB_STL))
    return BLOB_STL


def _read_stl_arrays(path: Path):
    """STL surface as ``(verts (n,3) float64, tris (m,3) int)``."""
    import pyvista as pv

    surf = pv.read(str(path)).triangulate().clean()
    verts = np.asarray(surf.points, dtype=np.float64)
    faces = surf.faces.reshape(-1, 4)[:, 1:]  # drop the leading "3" count
    return verts, np.asarray(faces, dtype=np.int64)


# ------------------------------------------------------------- gmsh builders
# Each operates on the already-initialized gmsh model via its OCC kernel; the
# caller synchronizes, sets the size, and runs generate(3).


def _g_sphere(occ):
    occ.addSphere(0, 0, 0, 1.0)


def _g_cylinder(occ):
    occ.addCylinder(0, 0, 0, 0, 0, 2.0, 0.7)


def _g_torus(occ):
    occ.addTorus(0, 0, 0, 1.0, 0.35)


def _g_drilled_block(occ):
    box = occ.addBox(-1.0, -1.0, -0.5, 2.0, 2.0, 1.0)
    bore = occ.addCylinder(0, 0, -0.6, 0, 0, 1.2, 0.5)
    occ.cut([(3, box)], [(3, bore)])


def _g_fused_spheres(occ):
    s1 = occ.addSphere(-0.55, 0, 0, 0.7)
    s2 = occ.addSphere(0.55, 0, 0, 0.7)
    occ.fuse([(3, s1)], [(3, s2)])


def _g_bracket(occ):
    plate = occ.addBox(-1.0, -0.6, 0.0, 2.0, 1.2, 0.3)
    bores = []
    for x in (-0.6, 0.6):
        bores.append((3, occ.addCylinder(x, 0, -0.1, 0, 0, 0.5, 0.18)))
    occ.cut([(3, plate)], bores)


def _g_gear(occ):
    pts = _gear_profile()
    ptags = [occ.addPoint(x, y, 0.0) for (x, y) in pts]
    ltags = [occ.addLine(ptags[i], ptags[(i + 1) % len(ptags)])
             for i in range(len(ptags))]
    loop = occ.addCurveLoop(ltags)
    surf = occ.addPlaneSurface([loop])
    ext = occ.extrude([(2, surf)], 0, 0, 0.5)
    vol = [d for d in ext if d[0] == 3]
    bore = occ.addCylinder(0, 0, -0.1, 0, 0, 0.7, 0.35)
    occ.cut(vol, [(3, bore)])


def _g_core_shell(occ):
    box = occ.addBox(-1, -1, -1, 2, 2, 2)
    ball = occ.addSphere(0, 0, 0, 0.6)
    occ.fragment([(3, box)], [(3, ball)])


def _g_layered_substrate(occ):
    air = occ.addBox(-1.5, -1.5, 0, 3, 3, 1.5)
    sub = occ.addBox(-1.5, -1.5, 0, 3, 3, 0.5)
    occ.fragment([(3, air)], [(3, sub)])


def _g_nested_spheres(occ):
    outer = occ.addSphere(0, 0, 0, 1.0)
    inner = occ.addSphere(0, 0, 0, 0.55)
    occ.fragment([(3, outer)], [(3, inner)])


def _g_blob(occ):
    # gmsh cannot CAD an organic blob; it remeshes the STL surface as a
    # discrete geometry, then tetrahedralizes the volume it bounds.
    import gmsh

    gmsh.merge(str(ensure_blob_stl()))
    # classify the discrete surface into a single watertight closed shell
    gmsh.model.mesh.classifySurfaces(math.pi, True, True, math.pi / 6)
    gmsh.model.mesh.createGeometry()
    surfs = gmsh.model.getEntities(2)
    loop = gmsh.model.geo.addSurfaceLoop([s[1] for s in surfs])
    gmsh.model.geo.addVolume([loop])
    gmsh.model.geo.synchronize()


# --------------------------------------------------------- rapidmesh builders


def _r_sphere():
    import rapidmesh as rm
    g = rm.Geometry(maxh=0.28)
    # icosphere (geodesic): isotropic, no pole rings, like gmsh/tetgen
    g.label(g.icosphere(1.0, subdivisions=2), "sphere")
    return g


def _r_cylinder():
    import rapidmesh as rm
    g = rm.Geometry(maxh=0.25)
    # uniform barrel grid (even surface distribution) vs ring strips
    g.label(g.cylinder(0.7, 2.0, segments=24, uniform=True), "cylinder")
    return g


def _r_torus():
    import rapidmesh as rm
    g = rm.Geometry(maxh=0.25)
    g.label(g.torus(1.0, 0.35, segments=28, tube_segments=14), "torus")
    return g


def _r_drilled_block():
    import rapidmesh as rm
    g = rm.Geometry(maxh=0.25)
    g.label(g.box(2.0, 2.0, 1.0, position=(-1.0, -1.0, -0.5)), "block")
    g.label(g.cylinder(0.5, 1.2, position=(0, 0, -0.6), segments=20,
                       uniform=True, void=True), "bore")
    return g


def _r_fused_spheres():
    import rapidmesh as rm
    g = rm.Geometry(maxh=0.30)
    g.label(g.icosphere(0.7, position=(-0.55, 0, 0), subdivisions=2), "a")
    g.label(g.icosphere(0.7, position=(0.55, 0, 0), subdivisions=2), "b")
    return g


def _r_bracket():
    import rapidmesh as rm
    g = rm.Geometry(maxh=0.15)
    g.label(g.box(2.0, 1.2, 0.3, position=(-1.0, -0.6, 0.0)), "plate")
    for x in (-0.6, 0.6):
        g.label(g.cylinder(0.18, 0.5, position=(x, 0, -0.1), segments=16,
                           uniform=True, void=True), "bores")
    return g


def _r_gear():
    import rapidmesh as rm
    g = rm.Geometry(maxh=0.16)
    pts = _gear_profile()
    bore = _circle(32, 0.35)
    g.label(g.prism(pts, 0.5, holes=[bore]), "gear")
    return g


def _r_blob():
    import rapidmesh as rm
    g = rm.Geometry(maxh=0.16)
    verts, tris = _read_stl_arrays(ensure_blob_stl())
    g.label(g.mesh_solid(verts, tris), "blob")
    return g


# multi-region: a later solid carves its region out of the earlier one, leaving
# a conformal material interface (rapidmesh's core capability). gmsh gets the
# same shared interface via OCC fragment; tetgen meshes the interface but has no
# material model (single region).

def _r_core_shell():
    import rapidmesh as rm
    g = rm.Geometry(maxh=0.28)
    g.label(g.box(2, 2, 2, position=(-1, -1, -1)), "shell")
    g.label(g.icosphere(0.6, subdivisions=2), "core")
    return g


def _r_layered_substrate():
    import rapidmesh as rm
    g = rm.Geometry(maxh=0.30)
    g.label(g.box(3, 3, 1.5, position=(-1.5, -1.5, 0)), "air")
    g.label(g.box(3, 3, 0.5, position=(-1.5, -1.5, 0)), "substrate")
    return g


def _r_nested_spheres():
    import rapidmesh as rm
    g = rm.Geometry(maxh=0.28)
    g.label(g.icosphere(1.0, subdivisions=2), "shell")
    g.label(g.icosphere(0.55, subdivisions=2), "core")
    return g


# ----------------------------------------------------------------- registry


@dataclass(frozen=True)
class CompareGeom:
    id: str
    name: str
    category: str
    target_h: float
    build_rapidmesh: Callable[[], "rm.Geometry"]
    build_gmsh: Callable[[object], None]


GEOMS: list[CompareGeom] = [
    CompareGeom("sphere", "Sphere", "Primitives", 0.28, _r_sphere, _g_sphere),
    CompareGeom("cylinder", "Cylinder", "Primitives", 0.25, _r_cylinder, _g_cylinder),
    CompareGeom("torus", "Torus", "Primitives", 0.25, _r_torus, _g_torus),
    CompareGeom("drilled_block", "Drilled Block", "Booleans", 0.25,
                _r_drilled_block, _g_drilled_block),
    CompareGeom("fused_spheres", "Fused Spheres", "Booleans", 0.30,
                _r_fused_spheres, _g_fused_spheres),
    CompareGeom("bracket", "Bracket", "Mechanical", 0.15, _r_bracket, _g_bracket),
    CompareGeom("gear", "Spur Gear", "Mechanical", 0.16, _r_gear, _g_gear),
    CompareGeom("blob", "Organic Blob", "Organic", 0.16, _r_blob, _g_blob),
    CompareGeom("core_shell", "Core + Shell", "Multi-Region", 0.28,
                _r_core_shell, _g_core_shell),
    CompareGeom("layered_substrate", "Layered Substrate", "Multi-Region", 0.30,
                _r_layered_substrate, _g_layered_substrate),
    CompareGeom("nested_spheres", "Nested Spheres", "Multi-Region", 0.28,
                _r_nested_spheres, _g_nested_spheres),
]
