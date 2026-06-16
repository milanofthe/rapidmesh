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
BUNNY_STL = FIXTURES / "bunny.stl"


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


def ensure_bunny_stl() -> Path:
    """Fetch the Stanford bunny once (pyvista example data), repair it to a
    watertight surface (fill the base hole), decimate to a web-friendly size,
    normalize to a ~2-unit body centred at the origin, and cache it."""
    if BUNNY_STL.exists():
        return BUNNY_STL
    import numpy as np
    import pyvista as pv
    from pyvista import examples

    FIXTURES.mkdir(parents=True, exist_ok=True)
    raw = examples.download_bunny()
    # The scanned bunny is an OPEN surface (hole at the base). Poisson-style
    # reconstruct_surface turns the points into a watertight shell (0 open
    # edges) that all three meshers can take, then decimate to a web size.
    surf = (raw.reconstruct_surface(nbr_sz=20).extract_largest()
            .clean().triangulate())
    # the scanned bunny is +Y up; the viewer is Z up -> stand it upright
    surf = surf.rotate_x(90, inplace=False)
    surf = surf.decimate(0.4).clean().triangulate()
    # normalize: centre at origin, bbox diagonal -> 2.0
    surf.points -= np.asarray(surf.center)
    diag = float(np.linalg.norm(
        np.asarray(surf.bounds[1::2]) - np.asarray(surf.bounds[0::2])))
    if diag > 0:
        surf.points *= 2.0 / diag
    surf.save(str(BUNNY_STL))
    return BUNNY_STL


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
    import gmsh
    box = occ.addBox(-1, -1, -1, 2, 2, 2)
    ball = occ.addSphere(0, 0, 0, 0.6)
    _, dim_tag_map = occ.fragment([(3, box)], [(3, ball)])
    occ.synchronize()
    # The tool (ball) volumes appear in both dtmap[0] and dtmap[1].
    # Shell = box-only volumes (dtmap[0] minus dtmap[1]); core = all tool volumes.
    tool_tags = {t[1] for t in dim_tag_map[1]}
    shell_vols = [t[1] for t in dim_tag_map[0] if t[1] not in tool_tags]
    core_vols = [t[1] for t in dim_tag_map[1]]
    gmsh.model.addPhysicalGroup(3, shell_vols, tag=1, name="shell")
    gmsh.model.addPhysicalGroup(3, core_vols, tag=2, name="core")


def _g_layered_substrate(occ):
    import gmsh
    air = occ.addBox(-1.5, -1.5, 0, 3, 3, 1.5)
    sub = occ.addBox(-1.5, -1.5, 0, 3, 3, 0.5)
    _, dim_tag_map = occ.fragment([(3, air)], [(3, sub)])
    occ.synchronize()
    # Sub (tool) volumes appear in both maps; air = air-only (dtmap[0] minus dtmap[1]).
    tool_tags = {t[1] for t in dim_tag_map[1]}
    air_vols = [t[1] for t in dim_tag_map[0] if t[1] not in tool_tags]
    sub_vols = [t[1] for t in dim_tag_map[1]]
    gmsh.model.addPhysicalGroup(3, air_vols, tag=1, name="air")
    gmsh.model.addPhysicalGroup(3, sub_vols, tag=2, name="substrate")


def _g_nested_spheres(occ):
    import gmsh
    outer = occ.addSphere(0, 0, 0, 1.0)
    inner = occ.addSphere(0, 0, 0, 0.55)
    _, dim_tag_map = occ.fragment([(3, outer)], [(3, inner)])
    occ.synchronize()
    # Inner (tool) volumes appear in both maps; shell = outer-only (dtmap[0] minus dtmap[1]).
    tool_tags = {t[1] for t in dim_tag_map[1]}
    shell_vols = [t[1] for t in dim_tag_map[0] if t[1] not in tool_tags]
    core_vols = [t[1] for t in dim_tag_map[1]]
    gmsh.model.addPhysicalGroup(3, shell_vols, tag=1, name="shell")
    gmsh.model.addPhysicalGroup(3, core_vols, tag=2, name="core")


def _g_stl(stl_path):
    # gmsh cannot CAD an organic surface; it remeshes the STL as a discrete
    # geometry, then tetrahedralizes the volume it bounds.
    import gmsh

    gmsh.merge(str(stl_path))
    gmsh.model.mesh.classifySurfaces(math.pi, True, True, math.pi / 6)
    gmsh.model.mesh.createGeometry()
    surfs = gmsh.model.getEntities(2)
    loop = gmsh.model.geo.addSurfaceLoop([s[1] for s in surfs])
    gmsh.model.geo.addVolume([loop])
    gmsh.model.geo.synchronize()


def _g_blob(occ):
    _g_stl(ensure_blob_stl())


def _g_bunny(occ):
    _g_stl(ensure_bunny_stl())


def _g_box(occ):
    occ.addBox(-0.8, -0.8, -0.8, 1.6, 1.6, 1.6)


def _g_cone(occ):
    occ.addCone(0, 0, -1.0, 0, 0, 2.0, 0.8, 0.4)


def _g_via(occ):
    import gmsh
    box = occ.addBox(-1, -1, -0.5, 2, 2, 1)
    pin = occ.addCylinder(0, 0, -0.7, 0, 0, 1.4, 0.3)
    _, dim_tag_map = occ.fragment([(3, box)], [(3, pin)])
    occ.synchronize()
    # Pin protrudes outside the box; fragment splits the pin into bottom stub,
    # middle section (inside box), and top stub. The middle section appears in
    # both dtmap[0] (box space) and dtmap[1] (pin space).
    # Substrate = box-only (dtmap[0] minus dtmap[1]); conductor = all pin pieces.
    pin_tags = {t[1] for t in dim_tag_map[1]}
    substrate_vols = [t[1] for t in dim_tag_map[0] if t[1] not in pin_tags]
    conductor_vols = [t[1] for t in dim_tag_map[1]]
    gmsh.model.addPhysicalGroup(3, substrate_vols, tag=1, name="substrate")
    gmsh.model.addPhysicalGroup(3, conductor_vols, tag=2, name="conductor")


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


def _r_box():
    import rapidmesh as rm
    g = rm.Geometry(maxh=0.30)
    g.label(g.box(1.6, 1.6, 1.6, position=(-0.8, -0.8, -0.8)), "box")
    return g


def _r_cone():
    import rapidmesh as rm
    g = rm.Geometry(maxh=0.25)
    # gentle taper (a steep tip would re-introduce apex slivers)
    g.label(g.cone(0.8, 0.4, 2.0, position=(0, 0, -1.0), segments=24,
                   uniform=True), "cone")
    return g


def _r_via():
    import rapidmesh as rm
    g = rm.Geometry(maxh=0.25)
    g.label(g.box(2, 2, 1, position=(-1, -1, -0.5)), "substrate")
    g.label(g.cylinder(0.3, 1.4, position=(0, 0, -0.7), segments=20,
                       uniform=True), "conductor")
    return g


def _r_bunny():
    import rapidmesh as rm
    g = rm.Geometry(maxh=0.14)
    verts, tris = _read_stl_arrays(ensure_bunny_stl())
    g.label(g.mesh_solid(verts, tris), "bunny")
    return g


def _naca0012_points(chord: float, n: int):
    """NACA 0012 outline as a polyline: upper trailing->leading edge, then
    lower leading->trailing (blunt TE), cosine x-spacing toward the LE."""
    def yt(x):
        return 0.6 * (0.2969 * x ** 0.5 - 0.1260 * x - 0.3516 * x * x
                      + 0.2843 * x ** 3 - 0.1015 * x ** 4)
    # LE-only clustering (x = s^2): dense at the high-curvature nose, coarse at
    # the near-straight tail (cosine would cluster the tail too and fan it).
    pts = []
    for i in range(n + 1):
        s = i / n
        x = (1.0 - s) ** 2  # 1 (TE) -> 0 (LE), dense near LE
        pts.append((x * chord, yt(x) * chord))
    for i in range(1, n + 1):
        s = i / n
        x = s ** 2  # 0 (LE) -> 1 (TE), dense near LE
        pts.append((x * chord, -yt(x) * chord))
    return pts


# A NACA 0012 wing as a through-hole in an air box (the external-aero / wind-
# tunnel setup): the air is meshed AROUND the airfoil, whose curved skin is the
# internal boundary. rapidmesh sweeps its analytic spline profile and refines the
# air by the profile curvature (fine nose, coarse far field); gmsh cuts the same
# OCC-spline airfoil from the box at a uniform size; tetgen meshes gmsh's surface
# with a hole point inside the airfoil column.
def _r_naca():
    import rapidmesh as rm
    # grading 1.0: steeper than the 0.5 default so the fine LE/TE bands coarsen
    # fast into the far field (few tets) instead of a wide fine halo.
    g = rm.Geometry(maxh=0.4, grading=1.0)
    g.label(g.box(3, 2, 0.5, position=(-1, -1, 0)), "air")
    g.airfoil_naca0012(1.0, 0.5, position=(0, 0, 0), n_seg=140, void=True)
    return g


def _g_naca(occ):
    box = occ.addBox(-1, -1, 0, 3, 2, 0.5)
    pts = _naca0012_points(1.0, 40)
    ptags = [occ.addPoint(x, y, 0.0) for (x, y) in pts]
    spline = occ.addSpline(ptags)
    close = occ.addLine(ptags[-1], ptags[0])
    loop = occ.addCurveLoop([spline, close])
    surf = occ.addPlaneSurface([loop])
    ext = occ.extrude([(2, surf)], 0, 0, 0.5)
    vol = [d for d in ext if d[0] == 3]
    occ.cut([(3, box)], vol)


# ----------------------------------------- additional edge / curved examples
# Chosen to stress the boundary representation the mesher consumes: a circular
# intersection edge (hemisphere), nested barrels with ring edges (tube), tangent
# curved patches (capsule), a concave planar edge (L-bracket), and sharp dihedral
# edges (wedge).


def _r_hemisphere():
    import rapidmesh as rm
    g = rm.Geometry(maxh=0.2)
    g.label(g.icosphere(1.0, subdivisions=3), "dome")
    # carve the lower half: a circular edge appears between cap and flat disk
    g.box(3, 3, 1.2, position=(-1.5, -1.5, -1.2), void=True)
    return g


def _g_hemisphere(occ):
    s = occ.addSphere(0, 0, 0, 1.0)
    b = occ.addBox(-1.5, -1.5, -1.2, 3, 3, 1.2)
    occ.cut([(3, s)], [(3, b)])


def _r_tube():
    import rapidmesh as rm
    g = rm.Geometry(maxh=0.18)
    g.label(g.cylinder(0.8, 2.0, position=(0, 0, -1.0), segments=32, uniform=True), "tube")
    g.cylinder(0.45, 2.2, position=(0, 0, -1.1), segments=28, uniform=True, void=True)
    return g


def _g_tube(occ):
    o = occ.addCylinder(0, 0, -1.0, 0, 0, 2.0, 0.8)
    i = occ.addCylinder(0, 0, -1.1, 0, 0, 2.2, 0.45)
    occ.cut([(3, o)], [(3, i)])


def _r_capsule():
    import rapidmesh as rm
    g = rm.Geometry(maxh=0.2)
    g.label(g.cylinder(0.6, 1.2, position=(0, 0, -0.6), segments=28, uniform=True), "cap")
    g.icosphere(0.6, position=(0, 0, -0.6), subdivisions=3)
    g.icosphere(0.6, position=(0, 0, 0.6), subdivisions=3)
    return g


def _g_capsule(occ):
    c = occ.addCylinder(0, 0, -0.6, 0, 0, 1.2, 0.6)
    s1 = occ.addSphere(0, 0, -0.6, 0.6)
    s2 = occ.addSphere(0, 0, 0.6, 0.6)
    occ.fuse([(3, c)], [(3, s1), (3, s2)])


_L_POLY = [(0.0, 0.0), (2.0, 0.0), (2.0, 0.6), (0.6, 0.6), (0.6, 2.0), (0.0, 2.0)]


def _r_l_bracket():
    import rapidmesh as rm
    g = rm.Geometry(maxh=0.13)
    g.label(g.prism(_L_POLY, 0.6, position=(-1.0, -1.0, 0.0)), "lbracket")
    return g


def _g_l_bracket(occ):
    ptags = [occ.addPoint(x - 1.0, y - 1.0, 0.0) for (x, y) in _L_POLY]
    ltags = [occ.addLine(ptags[i], ptags[(i + 1) % len(ptags)]) for i in range(len(ptags))]
    loop = occ.addCurveLoop(ltags)
    surf = occ.addPlaneSurface([loop])
    occ.extrude([(2, surf)], 0, 0, 0.6)


def _r_wedge():
    import rapidmesh as rm
    g = rm.Geometry(maxh=0.16)
    g.label(g.wedge(2.0, 1.0, 1.0, position=(-1.0, -0.5, 0.0), top_x=0.0), "wedge")
    return g


def _g_wedge(occ):
    # triangular cross-section in xz at y=-0.5, extruded +y by 1.0
    p0 = occ.addPoint(-1.0, -0.5, 0.0)
    p1 = occ.addPoint(1.0, -0.5, 0.0)
    p2 = occ.addPoint(-1.0, -0.5, 1.0)
    ls = [occ.addLine(p0, p1), occ.addLine(p1, p2), occ.addLine(p2, p0)]
    loop = occ.addCurveLoop(ls)
    surf = occ.addPlaneSurface([loop])
    occ.extrude([(2, surf)], 0, 1.0, 0)


# ------------------------------------------------- more fused-sphere variants
# Sphere-sphere intersections (circular intersection edges, recovered analytically
# by the B-rep): different counts, sizes and overlaps.


def _r_fused_unequal():
    import rapidmesh as rm
    g = rm.Geometry(maxh=0.18)
    g.label(g.icosphere(0.8, position=(-0.4, 0, 0), subdivisions=3), "a")
    g.icosphere(0.5, position=(0.55, 0, 0), subdivisions=3)
    return g


def _g_fused_unequal(occ):
    a = occ.addSphere(-0.4, 0, 0, 0.8)
    b = occ.addSphere(0.55, 0, 0, 0.5)
    occ.fuse([(3, a)], [(3, b)])


def _r_tri_fused():
    import rapidmesh as rm
    g = rm.Geometry(maxh=0.18)
    g.label(g.icosphere(0.7, position=(0.0, 0.0, 0), subdivisions=3), "a")
    g.icosphere(0.7, position=(0.9, 0.0, 0), subdivisions=3)
    g.icosphere(0.7, position=(0.45, 0.78, 0), subdivisions=3)
    return g


def _g_tri_fused(occ):
    s = [occ.addSphere(0.0, 0.0, 0, 0.7), occ.addSphere(0.9, 0.0, 0, 0.7),
         occ.addSphere(0.45, 0.78, 0, 0.7)]
    occ.fuse([(3, s[0])], [(3, s[1]), (3, s[2])])


def _r_fused_chain():
    import rapidmesh as rm
    g = rm.Geometry(maxh=0.16)
    g.label(g.icosphere(0.6, position=(0.0, 0, 0), subdivisions=3), "a")
    for x in (0.8, 1.6, 2.4):
        g.icosphere(0.6, position=(x, 0, 0), subdivisions=3)
    return g


def _g_fused_chain(occ):
    s = [occ.addSphere(x, 0, 0, 0.6) for x in (0.0, 0.8, 1.6, 2.4)]
    occ.fuse([(3, s[0])], [(3, s[i]) for i in (1, 2, 3)])


def _r_fused_deep():
    import rapidmesh as rm
    g = rm.Geometry(maxh=0.16)
    g.label(g.icosphere(0.8, position=(-0.3, 0, 0), subdivisions=3), "a")
    g.icosphere(0.8, position=(0.3, 0, 0), subdivisions=3)
    return g


def _g_fused_deep(occ):
    a = occ.addSphere(-0.3, 0, 0, 0.8)
    b = occ.addSphere(0.3, 0, 0, 0.8)
    occ.fuse([(3, a)], [(3, b)])


# ----------------------------------------------------------------- registry


@dataclass(frozen=True)
class CompareGeom:
    id: str
    name: str
    category: str
    target_h: float
    build_rapidmesh: Callable[[], "rm.Geometry"]
    build_gmsh: Callable[[object], None]
    # Interior seed points for tetgen's -A (regionattrib) mechanism.
    # Each entry is (x, y, z, material_id). One seed per topological sub-volume
    # of the PLC; multiple entries may share the same material_id (e.g. the via
    # conductor is 3 separate OCC solids after fragment but one material).
    # Empty tuple means single-region geometry: all tets get label 1.
    region_seeds: tuple = ()
    # Hole points (one inside each cavity to be EXCLUDED), for tetgen's
    # add_hole: an airfoil-shaped void in an air box is a hole, not a region.
    hole_points: tuple = ()
    # Enable rapidmesh local-feature-size refinement (thin features like a
    # trailing edge): grades the volume out of the thin part, no sliver fan.
    density_weighted: bool = False
    # gmsh curvature-adaptive sizing (elements per 2*pi of curvature; 0 = off,
    # uniform). The fair counterpart to rapidmesh's adaptive mode for curved
    # geometry (gmsh's native strength).
    gmsh_curvature: float = 0.0


GEOMS: list[CompareGeom] = [
    CompareGeom("sphere", "Sphere", "Primitives", 0.28, _r_sphere, _g_sphere),
    CompareGeom("cylinder", "Cylinder", "Primitives", 0.25, _r_cylinder, _g_cylinder),
    CompareGeom("box", "Box", "Primitives", 0.30, _r_box, _g_box),
    CompareGeom("cone", "Cone", "Primitives", 0.25, _r_cone, _g_cone),
    CompareGeom("torus", "Torus", "Primitives", 0.25, _r_torus, _g_torus),
    CompareGeom("wedge", "Wedge", "Primitives", 0.16, _r_wedge, _g_wedge),
    CompareGeom("hemisphere", "Hemisphere", "Booleans", 0.2,
                _r_hemisphere, _g_hemisphere, gmsh_curvature=30.0),
    CompareGeom("tube", "Hollow Tube", "Booleans", 0.18,
                _r_tube, _g_tube, gmsh_curvature=30.0),
    CompareGeom("capsule", "Capsule", "Booleans", 0.2,
                _r_capsule, _g_capsule, gmsh_curvature=30.0),
    CompareGeom("l_bracket", "L-Bracket", "Mechanical", 0.13,
                _r_l_bracket, _g_l_bracket),
    CompareGeom("drilled_block", "Drilled Block", "Booleans", 0.25,
                _r_drilled_block, _g_drilled_block),
    CompareGeom("fused_spheres", "Fused Spheres", "Booleans", 0.30,
                _r_fused_spheres, _g_fused_spheres),
    CompareGeom("fused_unequal", "Fused (unequal)", "Booleans", 0.18,
                _r_fused_unequal, _g_fused_unequal, gmsh_curvature=30.0),
    CompareGeom("tri_fused", "Three Fused", "Booleans", 0.18,
                _r_tri_fused, _g_tri_fused, gmsh_curvature=30.0),
    CompareGeom("fused_chain", "Fused Chain", "Booleans", 0.16,
                _r_fused_chain, _g_fused_chain, gmsh_curvature=30.0),
    CompareGeom("fused_deep", "Fused (deep overlap)", "Booleans", 0.16,
                _r_fused_deep, _g_fused_deep, gmsh_curvature=30.0),
    CompareGeom("bracket", "Bracket", "Mechanical", 0.15, _r_bracket, _g_bracket),
    CompareGeom("gear", "Spur Gear", "Mechanical", 0.16, _r_gear, _g_gear),
    CompareGeom("blob", "Organic Blob", "Organic", 0.16, _r_blob, _g_blob),
    CompareGeom("bunny", "Stanford Bunny", "Organic", 0.14, _r_bunny, _g_bunny),
    CompareGeom("naca0012", "NACA 0012 Wing", "Organic", 0.4, _r_naca, _g_naca,
                hole_points=((0.3, 0.0, 0.25),), density_weighted=True,
                gmsh_curvature=30.0),
    CompareGeom("core_shell", "Core + Shell", "Multi-Region", 0.28,
                _r_core_shell, _g_core_shell,
                region_seeds=((0.8, 0.0, 0.0, 1), (0.0, 0.0, 0.0, 2))),
    CompareGeom("layered_substrate", "Layered Substrate", "Multi-Region", 0.30,
                _r_layered_substrate, _g_layered_substrate,
                region_seeds=((0.0, 0.0, 1.0, 1), (0.0, 0.0, 0.25, 2))),
    CompareGeom("nested_spheres", "Nested Spheres", "Multi-Region", 0.28,
                _r_nested_spheres, _g_nested_spheres,
                region_seeds=((0.0, 0.0, 0.77, 1), (0.0, 0.0, 0.0, 2))),
    CompareGeom("via", "Via (substrate + pin)", "Multi-Region", 0.25,
                _r_via, _g_via,
                region_seeds=((-0.7, 0.0, 0.0, 1), (0.0, 0.0, 0.0, 2),
                               (0.0, 0.0, -0.6, 2), (0.0, 0.0, 0.6, 2))),
]
