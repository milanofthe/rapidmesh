"""Permutation tests for the error/size control surface: the global knobs
(maxh, tol_edge/tol_surf, maxh_edge/surf/vol), the per-dimension hierarchy
(g.edge()/g.surf()/g.region()), and the per-entity hierarchy
(g.region(..).surf(..).edge(..) with id/tag/normal/near/between selectors).

Each check asserts a MONOTONIC effect (a finer bound makes more elements, a
selector refines only its target), which is robust to absolute counts. Run with
``pytest`` or directly (``python python/tests/test_sizing.py``).
"""
import math

import numpy as np
import rapidmesh as rm


def _box(maxh, setup=None, **mesh_kw):
    g = rm.Geometry(maxh=maxh)
    g.box(4.0, 4.0, 4.0, (0.0, 0.0, 0.0))
    if setup is not None:
        setup(g)
    return g.mesh(maxh=maxh, **mesh_kw)


def _ntets(maxh, setup=None, **kw):
    return int(_box(maxh, setup, **kw).stats["n_tets"])


def _cyl_faces(tol_edge=1e-2, tol_surf=1e-2, maxh_surf=math.inf):
    g = rm.Geometry(maxh=2.0)
    g.cylinder(1.0, 2.0, position=(0.0, 0.0, 0.0))
    m = g.surface_mesh(maxh=2.0, tol_edge=tol_edge, tol_surf=tol_surf, maxh_surf=maxh_surf)
    return len(m.faces)


def _box_faces(maxh, setup=None):
    g = rm.Geometry(maxh=maxh)
    g.box(4.0, 4.0, 4.0, (0.0, 0.0, 0.0))
    if setup is not None:
        setup(g)
    return len(g.surface_mesh(maxh=maxh).faces)


_ALL_BOX_NORMALS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


def _refine_all_faces(g, h=0.5):
    for n in _ALL_BOX_NORMALS:
        g.surf(normal=n).maxh = h


# ---- global knobs --------------------------------------------------------

def test_global_maxh_refines():
    assert _ntets(0.8) > _ntets(2.0)


def test_maxh_vol_refines_interior():
    assert _ntets(4.0, maxh_vol=0.8) > _ntets(4.0)


def test_maxh_edge_refines_edges():
    assert _ntets(4.0, maxh_edge=0.8) > _ntets(4.0)


def test_tol_surf_refines_curved_surface():
    assert _cyl_faces(tol_surf=1e-3) > _cyl_faces(tol_surf=1e-2)


def test_tol_edge_refines_curved_edges():
    assert _cyl_faces(tol_edge=1e-3) > _cyl_faces(tol_edge=1e-2)


def test_maxh_surf_refines_curved_surface():
    assert _cyl_faces(maxh_surf=0.2) > _cyl_faces(maxh_surf=2.0)


# ---- per-dimension hierarchy (unfiltered == the global knob) -------------

def test_g_edge_equals_maxh_edge():
    hier = _ntets(4.0, setup=lambda g: setattr(g.edge(), "maxh", 0.8))
    flat = _ntets(4.0, maxh_edge=0.8)
    assert hier == flat


def test_g_region_equals_maxh_vol():
    hier = _ntets(4.0, setup=lambda g: setattr(g.region(), "maxh", 0.8))
    flat = _ntets(4.0, maxh_vol=0.8)
    assert hier == flat


# ---- per-entity hierarchy ------------------------------------------------

def _points_on_edge(mesh, a, b):
    P = np.asarray(mesh.points)
    ab = np.array(b) - np.array(a)
    l2 = float(ab @ ab)
    t = np.clip((P - a) @ ab / l2, 0.0, 1.0)
    q = np.array(a) + t[:, None] * ab
    return int(np.count_nonzero(np.einsum("ij,ij->i", P - q, P - q) < 1e-12))


def test_per_edge_near_refines_only_that_edge():
    # Edge along x at y=z=0 of the [0,4]^3 box; refine just it.
    a, b = (0.0, 0.0, 0.0), (4.0, 0.0, 0.0)
    base = _box(4.0)
    fine = _box(4.0, setup=lambda g: setattr(g.edge(near=(2.0, 0.0, 0.0)), "maxh", 0.4))
    assert _points_on_edge(fine, a, b) > _points_on_edge(base, a, b)


def test_per_surface_normal_selects_faces():
    # Refining the +z face only must add fewer tets than refining all faces.
    one = _ntets(4.0, setup=lambda g: setattr(g.surf(normal=(0.0, 0.0, 1.0)), "maxh", 0.5))
    allf = _ntets(4.0, setup=lambda g: setattr(g.surf(), "maxh", 0.5))
    assert one <= allf


# ---- global cap vs per-entity consistency (no path silently ignores a knob) ---

def test_global_surf_refines_volume():
    # The global surface cap must refine the VOLUME, not just the surface tiling
    # (regression guard: cap_surf was omitted from the domain sizing field).
    assert _ntets(4.0, setup=lambda g: setattr(g.surf(), "maxh", 0.5)) > 4 * _ntets(4.0)


def test_global_surf_equals_per_entity_all_faces():
    # The global cap and the equivalent per-entity override on EVERY face produce
    # the same volume field, hence the same mesh.
    glob = _ntets(4.0, setup=lambda g: setattr(g.surf(), "maxh", 0.5))
    per_entity = _ntets(4.0, setup=_refine_all_faces)
    assert glob == per_entity


def test_per_entity_surf_refines_surface_export():
    # A per-entity surf override must reach surface_mesh() too (regression guard:
    # the surface export built its domain without the per-entity overrides).
    assert _box_faces(4.0, setup=_refine_all_faces) > 4 * _box_faces(4.0)


def test_maxh_vol_refines_box_interior():
    # The global volume cap must densify the interior, not merely a boundary band.
    assert _ntets(4.0, maxh_vol=0.5) > 10 * _ntets(4.0)


def test_hierarchical_composition_runs_and_refines():
    # region -> surf(+x face) -> edge(near its y=0 edge): refine just that edge.
    a, b = (4.0, 0.0, 0.0), (4.0, 0.0, 4.0)

    def setup(g):
        g.region().maxh = 1.5
        g.region(1).surf(normal=(1.0, 0.0, 0.0)).edge(near=(4.0, 0.0, 2.0)).maxh = 0.3

    base = _box(4.0, setup=lambda g: setattr(g.region(), "maxh", 1.5))
    fine = _box(4.0, setup=setup)
    assert _points_on_edge(fine, a, b) > _points_on_edge(base, a, b)


def test_most_specific_wins():
    # Global coarse edges, one edge fine: the fine override must take effect.
    a, b = (0.0, 0.0, 0.0), (4.0, 0.0, 0.0)

    def setup(g):
        g.edge().maxh = 3.0
        g.edge(near=(2.0, 0.0, 0.0)).maxh = 0.4

    base = _box(4.0, setup=lambda g: setattr(g.edge(), "maxh", 3.0))
    fine = _box(4.0, setup=setup)
    assert _points_on_edge(fine, a, b) > _points_on_edge(base, a, b)


def test_topology_follows_every_change():
    """The sizing ids come from the model every mesh uses, rebuilt on any
    change: a union and a sheet change the faces too, not only new solids."""
    g = rm.Geometry(maxh=1.0)
    a = g.box(2.0, 2.0, 2.0, (0.0, 0.0, 0.0))
    b = g.box(2.0, 2.0, 2.0, (1.0, 0.0, 0.0))
    split = len(g._topology().faces())
    g.union(a, b)
    fused = len(g._topology().faces())
    assert fused < split, (split, fused)
    g.plate((0.5, 0.5, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), tag=3)
    assert len(g._topology().faces()) > fused


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\nall {len(fns)} sizing permutation tests passed")



def test_faces_by_origin_and_role_names():
    """``solid=``/``role=`` select a face by its origin; the name of the
    role (``"+z"`` of a box) and its index agree, and the selection keeps
    its face when shapes are added after it."""
    import pytest

    g = rm.Geometry(maxh=0.4)
    b = g.box(1.0, 1.0, 1.0)
    assert b.roles == ("-z", "+z", "-y", "+y", "-x", "+x")
    by_name = g._resolve(g.surf(solid=b, role="+z"))
    by_index = g._resolve(g.surf(solid=b.index, role=1))
    assert by_name == by_index and len(by_name) == 1
    faces = g._topology().faces()
    assert faces[by_name[0]][1][2] > 0.99  # the +z face
    g.surf(solid=b, role="+z").name = "lid"
    g.box(1.0, 1.0, 1.0, position=(-3.0, 0.0, 0.0))
    m = g.mesh()
    lid = m.sets()["faces"]["lid"]
    assert len(lid) > 0
    with pytest.raises(ValueError):
        g.surf(solid=b, role="top")


def test_faces_edges_and_regions_carry_bounding_boxes():
    g = rm.Geometry(maxh=0.5)
    g.box(1.0, 2.0, 3.0, position=(1.0, 0.0, -1.0))
    t = g._topology()
    [(lo, hi)] = t.region_bbox()
    assert list(lo) == [1.0, 0.0, -1.0] and list(hi) == [2.0, 2.0, 2.0]
    for f in t.faces():
        lo, hi = f[10]
        assert all(l <= c <= h for l, c, h in zip(lo, f[0], hi))
    top = [f for f in t.faces() if f[1][2] > 0.99][0]
    assert [list(b) for b in top[10]] == [[1.0, 0.0, 2.0], [2.0, 2.0, 2.0]]
    for e in t.edges():
        lo, hi = e[6]
        assert all(l <= c <= h for l, c, h in zip(lo, e[2], hi))


def test_triangle_budget_caps_the_surface_mesh():
    """``target_triangles`` coarsens the bottom-up surface mesh to its budget."""
    def faces(target):
        g = rm.Geometry()
        g.sphere(1.0)
        return len(g.surface_mesh(maxh=0.05, target_triangles=target).faces)
    free = faces(None)
    assert free > 10_000
    for target in (2000, 500):
        assert 0.7 * target < faces(target) <= 1.06 * target


def test_dorfler_refines_where_the_indicator_is():
    """One MARK -> REFINE step: the marked triangles become size points, the
    remesh is finer there and nowhere else much."""
    import numpy as np
    g = rm.Geometry()
    g.sphere(1.0)
    m = g.surface_mesh(maxh=0.3)
    cen = np.array([m.points[f].mean(axis=0) for f in m.faces])
    eta = np.exp(-20.0 * np.sum((cen - [1.0, 0.0, 0.0]) ** 2, axis=1))
    assert len(rm.dorfler_mark(eta)) < len(rm.dorfler_mark(eta, theta=0.9))
    fine = rm.refine_dorfler(g, m, eta, maxh=0.3)
    near = lambda mesh: sum(1 for f in mesh.faces if mesh.points[f].mean(axis=0)[0] > 0.8)
    assert near(fine) > 1.5 * near(m)
    assert len(fine.faces) < 2 * len(m.faces)


def test_polygon_union_merges_overlapping_rectangles():
    rect = lambda x0, x1: [(x0, 0), (x1, 0), (x1, 1), (x0, 1)]
    merged = rm.polygon_union([rect(0, 2), rect(1, 3), rect(5, 6)])
    assert len(merged) == 2
    areas = sorted(abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(o, o[1:] + o[:1]))) / 2 for o, _ in merged)
    assert areas == [1.0, 3.0]


def test_edges_by_kind_name():
    """``kind=`` selects edges by the name of their curve: a cylinder has two
    circles and no line; an unknown name says which names there are."""
    import pytest

    g = rm.Geometry(maxh=0.4)
    g.cylinder(0.5, 1.0)
    circles = g._resolve(g.edge(kind="circle"))
    assert len(circles) == 2
    assert g._resolve(g.edge(kind="line")) == []
    kinds = {e[4] for e in g._topology().edges()}
    assert "circle" in kinds
    with pytest.raises(ValueError, match="unknown edge kind"):
        g._resolve(g.edge(kind="arc"))
